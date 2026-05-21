"""
ufs2arco source for MRMS QPE patch extraction driven by a patch manifest JSON.

Each sample is a single 256×256 tile from the MRMS MultiSensor_QPE_01H_Pass2
product at one valid_time.  The manifest is produced by
superres_precip.data.patch_sampler.build_manifest and guarantees a uniform
patch count across all valid_times (required by the ufs2arco DataMover).
"""
import gzip
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from .base import Source

logger = logging.getLogger(__name__)

_BUCKET = "noaa-mrms-pds"
_PRODUCT = "MultiSensor_QPE_01H_Pass2_00.00"

# MRMS grid constants (0.01° lat/lon, top-left origin)
_MRMS_LAT0 =  54.995
_MRMS_LON0 = 230.005
_MRMS_DLAT =  -0.01
_MRMS_DLON =   0.01


def _mrms_patch_latlon(y0: int, x0: int, ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return 2-D lat/lon arrays (ny, nx) for an MRMS patch."""
    rows = np.arange(y0, y0 + ny)
    cols = np.arange(x0, x0 + nx)
    lats = _MRMS_LAT0 + rows * _MRMS_DLAT         # (ny,)
    lons_e = _MRMS_LON0 + cols * _MRMS_DLON       # (nx,)  East lon 230-300°
    lons = np.where(lons_e > 180, lons_e - 360.0, lons_e)  # convert to -180..180
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    return lat2d.astype(np.float64), lon2d.astype(np.float64)


class AWSMRMSPatches(Source):
    """
    MRMS QPE patch source driven by a pre-built manifest JSON.

    Manifest format (list of dicts, one per patch):
        valid_time, patch_idx, mrms_y0, mrms_x0, hrrr_y0, hrrr_x0,
        lat_c, lon_c, source, intensity, trajectory_id

    All valid_times must have the same patch count (enforced at manifest build
    time by superres_precip.data.patch_sampler.build_manifest).

    ufs2arco DataMover iterates the Cartesian product:
        valid_time × patch_step  →  open_sample_dataset(dims, ...)
    """

    sample_dims = ("valid_time", "patch_step")
    horizontal_dims = ("y", "x")
    available_levels = ()
    statics_vary_per_sample = True
    STORED_FREQ = "1h"

    def __init__(
        self,
        manifest_path: str,
        variables: tuple | list = ("qpe_01h",),
        levels=None,
        use_nearest_levels: bool = False,
        slices: dict | None = None,
        patch_size: int = 256,
    ) -> None:
        # Must be set before super().__init__() which calls __str__ → sample_dims properties
        self._valid_time_list: list[pd.Timestamp] = []
        self._n_patches: int = 0
        super().__init__(
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices,
        )
        self.patch_size = patch_size

        with open(manifest_path) as f:
            entries = json.load(f)

        # group by valid_time (preserve manifest order within each group)
        self._manifest_by_time: dict[pd.Timestamp, list[dict]] = {}
        for e in entries:
            t = pd.Timestamp(e["valid_time"])
            self._manifest_by_time.setdefault(t, []).append(e)

        counts = {t: len(v) for t, v in self._manifest_by_time.items()}
        unique_counts = set(counts.values())
        if len(unique_counts) != 1:
            raise ValueError(
                f"AWSMRMSPatches: manifest has non-uniform patch counts per "
                f"valid_time: {unique_counts}. Re-build with a fixed "
                f"n_patches_per_time."
            )
        self._n_patches = unique_counts.pop()
        self._valid_time_list = sorted(self._manifest_by_time.keys())

        # trajectory_id index: global ordinal → trajectory_id string
        self._trajectory_ids = [
            e["trajectory_id"]
            for t in self._valid_time_list
            for e in self._manifest_by_time[t]
        ]

        # per-time QPE cache (avoids re-downloading for each patch_step)
        self._qpe_cache: dict[pd.Timestamp, np.ndarray] = {}
        self._fs: s3fs.S3FileSystem | None = None

        logger.info(
            f"AWSMRMSPatches: {len(self._valid_time_list)} times × "
            f"{self._n_patches} patches = {len(entries)} total"
        )

    # ------------------------------------------------------------------
    # DataMover-facing properties
    # ------------------------------------------------------------------

    @property
    def valid_time(self) -> list[pd.Timestamp]:
        return self._valid_time_list

    @property
    def patch_step(self) -> list[int]:
        return list(range(self._n_patches))

    @property
    def available_variables(self) -> tuple:
        return ("qpe_01h",)

    @property
    def trajectory_ids(self) -> list:
        return self._trajectory_ids

    @property
    def n_samples(self) -> int:
        return len(self._valid_time_list) * self._n_patches

    @property
    def valid_times(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([
            t
            for t in self._valid_time_list
            for _ in range(self._n_patches)
        ])

    # ------------------------------------------------------------------
    # Sample fetching
    # ------------------------------------------------------------------

    def open_sample_dataset(
        self,
        dims: dict,
        open_static_vars: bool = True,
        cache_dir: str | None = None,
    ) -> xr.Dataset:
        t: pd.Timestamp = dims["valid_time"]
        k: int = dims["patch_step"]

        entry = self._manifest_by_time[t][k]
        y0, x0 = int(entry["mrms_y0"]), int(entry["mrms_x0"])
        ps = self.patch_size

        qpe = self._get_qpe(t, cache_dir)
        tile = qpe[y0:y0 + ps, x0:x0 + ps]            # (ps, ps) float32

        lat2d, lon2d = _mrms_patch_latlon(y0, x0, ps, ps)

        time_idx = self._valid_time_list.index(t)
        sample_idx = time_idx * self._n_patches + k

        # latitude, longitude, and valid_time must be data variables (not just
        # coordinates) so that xds.expand_dims({"ensemble": [0]}) in the Anemoi
        # target broadcasts them, making lat.isel(ensemble=0) and
        # valid_time.squeeze("ensemble") work downstream.
        xds = xr.Dataset(
            {
                "qpe_01h":    (["time", "y", "x"], tile[np.newaxis]),
                "latitude":   (["y", "x"], lat2d),
                "longitude":  (["y", "x"], lon2d),
                "valid_time": (["time"], pd.DatetimeIndex([t])),
            },
            coords={
                "time": pd.DatetimeIndex([t]),
            },
        )
        xds.attrs["_sample_index"] = sample_idx

        return xds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_fs(self) -> s3fs.S3FileSystem:
        if self._fs is None:
            self._fs = s3fs.S3FileSystem(anon=True)
        return self._fs

    def _get_qpe(
        self, t: pd.Timestamp, cache_dir: str | None
    ) -> np.ndarray:
        if t in self._qpe_cache:
            return self._qpe_cache[t]

        date = f"{t.year:04d}{t.month:02d}{t.day:02d}"
        hms  = f"{t.hour:02d}0000"
        fname = f"MRMS_{_PRODUCT}_{date}-{hms}.grib2.gz"
        s3_key = f"{_BUCKET}/CONUS/{_PRODUCT}/{date}/{fname}"

        # optionally keep the .gz on disk to avoid re-download on restart
        cached_gz = None
        if cache_dir:
            cached_gz = Path(cache_dir) / fname
            cached_gz.parent.mkdir(parents=True, exist_ok=True)

        tmp_gz = tmp_grib = None
        try:
            if cached_gz and cached_gz.exists():
                gz_path = str(cached_gz)
            else:
                with tempfile.NamedTemporaryFile(suffix=".grib2.gz", delete=False) as f:
                    tmp_gz = f.name
                self._get_fs().get(s3_key, tmp_gz)
                gz_path = tmp_gz
                if cached_gz:
                    shutil.copy2(tmp_gz, str(cached_gz))

            with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
                tmp_grib = f.name
            with gzip.open(gz_path, "rb") as fin, open(tmp_grib, "wb") as fout:
                shutil.copyfileobj(fin, fout)

            import cfgrib
            datasets = cfgrib.open_datasets(
                tmp_grib,
                backend_kwargs={"indexpath": ""},
            )
            # QPE field: typeOfLevel=heightAboveSea; cfgrib names it 'unknown'
            ds = next(
                d for d in datasets
                if "heightAboveSea" in d.coords or any(
                    d.coords[c].attrs.get("GRIB_typeOfLevel") == "heightAboveSea"
                    for c in d.coords
                )
            )
            vname = list(ds.data_vars)[0]
            qpe = ds[vname].values.astype(np.float32)

        except Exception as exc:
            raise RuntimeError(
                f"AWSMRMSPatches: failed to fetch QPE for {t}: {exc}"
            ) from exc
        finally:
            for p in (tmp_gz, tmp_grib):
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

        # keep only the most recent time in cache to bound memory
        self._qpe_cache = {t: qpe}
        return qpe
