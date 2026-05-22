"""
ufs2arco source for HRRR patch extraction from NOAA GRIB2 files.

Reads from s3://noaa-hrrr-bdp-pds/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcf{fhr:02d}.grib2.

For each unique init time (t0) a single CONUS GRIB2 file is downloaded once,
all needed variable arrays are extracted and cached in memory, and each patch
call slices from the cache.  This eliminates the 120× redundant chunk downloads
that occur with the UofUtah Zarr source (chunks are (48, 150, 150) so reading
any single forecast hour downloads all 48).

Each wrfsfcf{fhr}.grib2 file contains two tp records:
  stepRange=0-{fhr}  (running total from forecast start)
  stepRange={fhr-1}-{fhr}  (native 1-hr bucket)
APCP uses the native 1-hr record directly — no f{fhr-1} download needed.

filter_by_keys reference (verified against hrrr.t12z.wrfsfcf02.grib2 2021-07-15):
  APCP     : typeOfLevel=surface, stepType=accum, paramId=228228, stepRange=1-{fhr} → var=tp
  CAPE_ML  : typeOfLevel=pressureFromGroundLayer, paramId=59, level=25500  → var=cape (0-255 hPa layer)
  RH_2M    : typeOfLevel=heightAboveGround, paramId=260242          → var=r2
  UGRD_700 : typeOfLevel=isobaricInhPa, level=700, shortName=u      → var=u
  VGRD_700 : typeOfLevel=isobaricInhPa, level=700, shortName=v      → var=v
  UGRD_10M : typeOfLevel=heightAboveGround, level=10, shortName=10u → var=u10  (OROLIFT only)
  VGRD_10M : typeOfLevel=heightAboveGround, level=10, shortName=10v → var=v10  (OROLIFT only)
  MAXREF   : typeOfLevel=heightAboveGround, stepType=max, paramId=0, level=1000 → var=unknown
  MAXUVV   : typeOfLevel=pressureFromGroundLayer, stepType=max, paramId=0 → var=unknown
  HGT      : typeOfLevel=surface, paramId=228002                    → var=orog  (static)
"""
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

_BUCKET = "noaa-hrrr-bdp-pds"

# filter_by_keys for each project variable name.
# Variables that cfgrib decodes as 'unknown' (NOAA local params) are identified
# by typeOfLevel + stepType + paramId=0 + level (where applicable).
_GRIB_FILTERS: dict[str, dict] = {
    "APCP":     {"typeOfLevel": "surface",                  "stepType": "accum", "paramId": 228228},
    "CAPE_ML":  {"typeOfLevel": "pressureFromGroundLayer",  "paramId": 59,       "level": 25500},
    "RH_2M":    {"typeOfLevel": "heightAboveGround",        "paramId": 260242},
    "UGRD_700": {"typeOfLevel": "isobaricInhPa",            "level": 700,        "shortName": "u"},
    "VGRD_700": {"typeOfLevel": "isobaricInhPa",            "level": 700,        "shortName": "v"},
    "UGRD_10M": {"typeOfLevel": "heightAboveGround",        "level": 10,         "shortName": "10u"},
    "VGRD_10M": {"typeOfLevel": "heightAboveGround",        "level": 10,         "shortName": "10v"},
    "MAXREF":   {"typeOfLevel": "heightAboveGround",        "stepType": "max",   "paramId": 0, "level": 1000},
    "MAXUVV":   {"typeOfLevel": "pressureFromGroundLayer",  "stepType": "max",   "paramId": 0},
    "HGT":      {"typeOfLevel": "surface",                  "paramId": 228002},
}

# module-level caches for static fields (shared across instances in same process)
_LATLON_CACHE: tuple[np.ndarray, np.ndarray] | None = None  # (lat2d, lon2d) in degrees
_HGT_CACHE:    np.ndarray | None = None                     # terrain height, metres


def _s3_path(t0: pd.Timestamp, fhr: int) -> str:
    date = f"{t0.year:04d}{t0.month:02d}{t0.day:02d}"
    fname = f"hrrr.t{t0.hour:02d}z.wrfsfcf{fhr:02d}.grib2"
    return f"{_BUCKET}/hrrr.{date}/conus/{fname}"


def _download(s3_key: str, fs: s3fs.S3FileSystem, cache_dir: str | None) -> str:
    """Download s3_key to a local path; return that path.

    If cache_dir is given the file is kept there permanently (by filename).
    Otherwise a temp file is used — caller is responsible for unlinking it.
    """
    fname = os.path.basename(s3_key)
    if cache_dir:
        local = os.path.join(cache_dir, fname)
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(local):
            fs.get(s3_key, local)
        return local

    tmp = tempfile.mktemp(suffix=".grib2")
    fs.get(s3_key, tmp)
    return tmp


def _read_var(grib_path: str, fbk: dict) -> np.ndarray:
    """Open one variable slice from a GRIB2 file; return float32 array (ny, nx)."""
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        filter_by_keys=fbk,
        backend_kwargs={"indexpath": ""},
    )
    var = list(ds.data_vars)[0]
    return ds[var].values.astype(np.float32)


def _read_latlon(grib_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat2d, lon2d) float64 arrays from any single GRIB2 message."""
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        filter_by_keys=_GRIB_FILTERS["HGT"],
        backend_kwargs={"indexpath": ""},
    )
    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return lat, lon


class AWSHRRRPatchesGrib2(Source):
    """
    HRRR f0X patch source that reads NOAA GRIB2 from s3://noaa-hrrr-bdp-pds/.

    Iterates over manifest entries (sample_dims = ("sample",)) exactly like
    AWSHRRRPatches.  The key difference is that on the first call for a given
    init time (t0) the full CONUS wrfsfc GRIB2 is downloaded once and all
    variable arrays are cached in memory.  Subsequent calls for the same t0
    (i.e., other patches at the same valid_time) slice from the in-memory cache
    with no S3 I/O.

    Requires MPIDataMover to use contiguous block assignment so that a rank's
    consecutive calls share the same t0.  See datamover.MPIDataMover.get_batch_indices.

    APCP uses the native 1-hr accumulation bucket (stepRange="{fhr-1}-{fhr}") present
    in every wrfsfcf{fhr}.grib2 — only one GRIB2 file is downloaded per init time.
    """

    sample_dims = ("sample",)
    horizontal_dims = ("y", "x")
    available_levels = ()
    statics_vary_per_sample = True
    STORED_FREQ = "1h"

    def __init__(
        self,
        manifest_path: str,
        variables: tuple | list = ("APCP", "CAPE_ML", "RH_2M", "UGRD_700", "VGRD_700", "MAXREF", "MAXUVV", "OROLIFT"),
        forecast_hours: tuple | list = (2,),
        levels=None,
        use_nearest_levels: bool = False,
        slices: dict | None = None,
        patch_size: int = 96,
    ) -> None:
        self._base_vars = list(variables)
        self._entries: list[dict] = []
        super().__init__(
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices,
        )
        self.patch_size = patch_size
        self.forecast_hours = list(forecast_hours)

        with open(manifest_path) as f:
            self._entries = json.load(f)

        # per-t0 cache: {t0: {varname: np.ndarray(1059, 1799)}}
        # keep only the most recent t0 to bound memory (~500 MB for all vars)
        self._hrrr_cache: dict[pd.Timestamp, dict[str, np.ndarray]] = {}
        self._fs: s3fs.S3FileSystem | None = None

        logger.info(
            f"AWSHRRRPatchesGrib2: {len(self._entries)} samples, "
            f"vars={self._base_vars}, fhrs={self.forecast_hours}"
        )

    # ------------------------------------------------------------------
    # DataMover-facing properties
    # ------------------------------------------------------------------

    @property
    def sample(self) -> list[int]:
        return list(range(len(self._entries)))

    @property
    def available_variables(self) -> tuple:
        return tuple(self._base_vars)

    @property
    def trajectory_ids(self) -> list:
        return [e["trajectory_id"] for e in self._entries]

    @property
    def n_samples(self) -> int:
        return len(self._entries)

    @property
    def valid_times(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([pd.Timestamp(e["valid_time"]) for e in self._entries])

    # ------------------------------------------------------------------
    # Sample fetching
    # ------------------------------------------------------------------

    def open_sample_dataset(
        self,
        dims: dict,
        open_static_vars: bool = True,
        cache_dir: str | None = None,
    ) -> xr.Dataset:
        i: int = dims["sample"]
        entry = self._entries[i]

        t = pd.Timestamp(entry["valid_time"])
        y0, x0 = int(entry["hrrr_y0"]), int(entry["hrrr_x0"])
        ps = self.patch_size
        y_sl = slice(y0, y0 + ps)
        x_sl = slice(x0, x0 + ps)

        needs_orolift = "OROLIFT" in self._base_vars
        fetch_vars = [v for v in self._base_vars if v != "OROLIFT"]

        data_vars: dict = {}
        for fhr in self.forecast_hours:
            t0 = t - pd.Timedelta(hours=fhr)
            conus = self._get_conus_arrays(t0, fhr, cache_dir)

            tiles: dict[str, np.ndarray] = {}
            for vname in fetch_vars:
                tiles[vname] = conus[vname][y_sl, x_sl]

            if needs_orolift:
                hgt = self._get_hgt_conus(t0, fhr, cache_dir)
                dz_dy, dz_dx = np.gradient(hgt[y_sl, x_sl].astype(np.float64), 3000.0, 3000.0)
                tiles["OROLIFT"] = (
                    conus["UGRD_10M"][y_sl, x_sl] * dz_dx
                    + conus["VGRD_10M"][y_sl, x_sl] * dz_dy
                ).astype(np.float32)

            for vname in self._base_vars:
                data_vars[f"{vname}_f{fhr:02d}"] = (["y", "x"], tiles[vname])

        lat2d, lon2d = self._get_latlon_conus(t0, fhr, cache_dir)

        timed_vars = {k: (["time", "y", "x"], v[1][np.newaxis]) for k, v in data_vars.items()}
        timed_vars["latitude"]   = (["y", "x"], lat2d[y_sl, x_sl])
        timed_vars["longitude"]  = (["y", "x"], lon2d[y_sl, x_sl])
        timed_vars["valid_time"] = (["time"], pd.DatetimeIndex([t]))

        xds = xr.Dataset(timed_vars, coords={"time": pd.DatetimeIndex([t])})
        xds.attrs["_sample_index"] = i
        return xds

    # ------------------------------------------------------------------
    # CONUS array cache
    # ------------------------------------------------------------------

    def _get_conus_arrays(
        self, t0: pd.Timestamp, fhr: int, cache_dir: str | None
    ) -> dict[str, np.ndarray]:
        """Return cached CONUS arrays for t0; download and parse GRIB2 on miss."""
        if t0 in self._hrrr_cache:
            return self._hrrr_cache[t0]

        global _HGT_CACHE, _LATLON_CACHE
        needs_orolift = "OROLIFT" in self._base_vars
        fetch_vars = [v for v in self._base_vars if v not in ("OROLIFT",)]

        fs = self._get_fs()
        fhr_path = _download(_s3_path(t0, fhr), fs, cache_dir)

        try:
            arrays: dict[str, np.ndarray] = {}
            for vname in fetch_vars:
                if vname == "APCP":
                    # Each wrfsfcf{fhr} contains a native 1-hr bucket: stepRange="{fhr-1}-{fhr}"
                    apcp_fbk = {**_GRIB_FILTERS["APCP"], "stepRange": f"{fhr - 1}-{fhr}"}
                    arrays["APCP"] = _read_var(fhr_path, apcp_fbk)
                else:
                    arrays[vname] = _read_var(fhr_path, _GRIB_FILTERS[vname])

            # Fetch 10m winds for OROLIFT (not output as standalone vars)
            if needs_orolift:
                arrays["UGRD_10M"] = _read_var(fhr_path, _GRIB_FILTERS["UGRD_10M"])
                arrays["VGRD_10M"] = _read_var(fhr_path, _GRIB_FILTERS["VGRD_10M"])

            # Populate static caches while file is open — avoids redundant S3 downloads
            if _HGT_CACHE is None:
                _HGT_CACHE = _read_var(fhr_path, _GRIB_FILTERS["HGT"])
            if _LATLON_CACHE is None:
                lat, lon = _read_latlon(fhr_path)
                _LATLON_CACHE = (lat, lon)
        finally:
            # clean up temp file (cache_dir files are kept on disk by _download)
            if cache_dir is None and os.path.exists(fhr_path):
                try:
                    os.unlink(fhr_path)
                except OSError:
                    pass

        # keep only the most recent t0 to bound memory
        self._hrrr_cache = {t0: arrays}
        return arrays

    def _get_hgt_conus(
        self, t0: pd.Timestamp, fhr: int, cache_dir: str | None
    ) -> np.ndarray:
        """Return static terrain height (1059, 1799).

        Normally populated by _get_conus_arrays (called first in open_sample_dataset).
        Falls back to a standalone download if the cache is cold (e.g. called in isolation).
        """
        global _HGT_CACHE
        if _HGT_CACHE is not None:
            return _HGT_CACHE

        fs = self._get_fs()
        f02_path = _download(_s3_path(t0, fhr), fs, cache_dir)
        try:
            _HGT_CACHE = _read_var(f02_path, _GRIB_FILTERS["HGT"])
        finally:
            if cache_dir is None and os.path.exists(f02_path):
                try:
                    os.unlink(f02_path)
                except OSError:
                    pass
        return _HGT_CACHE

    def _get_latlon_conus(
        self, t0: pd.Timestamp, fhr: int, cache_dir: str | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lat2d, lon2d) float64 arrays.

        Normally populated by _get_conus_arrays.  Falls back to standalone download.
        """
        global _LATLON_CACHE
        if _LATLON_CACHE is not None:
            return _LATLON_CACHE

        fs = self._get_fs()
        f02_path = _download(_s3_path(t0, fhr), fs, cache_dir)
        try:
            lat, lon = _read_latlon(f02_path)
            _LATLON_CACHE = (lat, lon)
        finally:
            if cache_dir is None and os.path.exists(f02_path):
                try:
                    os.unlink(f02_path)
                except OSError:
                    pass
        return _LATLON_CACHE

    def _get_fs(self) -> s3fs.S3FileSystem:
        if self._fs is None:
            self._fs = s3fs.S3FileSystem(anon=True)
        return self._fs
