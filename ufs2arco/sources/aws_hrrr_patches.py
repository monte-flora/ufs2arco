"""
ufs2arco source for HRRR patch extraction driven by a patch manifest JSON.

Each sample fetches a 96×96-cell halo from the UofUtah HRRR Zarr archive
(s3://hrrrzarr/sfc/) for each requested forecast hour and stacks them as
separate channels — e.g. PRATE_f01, PRATE_f02, PRATE_f03.

The manifest is the same JSON produced by
superres_precip.data.patch_sampler.build_manifest that drives AWSMRMSPatches,
ensuring that both sources produce identical time axes and trajectory_ids.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr

from .base import Source

logger = logging.getLogger(__name__)

_BASE_URL = "hrrrzarr/sfc"

# HRRR v4 LCC projection parameters (for lat/lon computation)
_PROJ_PARAMS = {
    "proj": "lcc",
    "lon_0": 262.5,
    "lat_0": 38.5,
    "lat_1": 38.5,
    "lat_2": 38.5,
    "a": 6371229.0,
    "b": 6371229.0,
}

# Maps project-level variable names to (level_group, zarr_varname) in hrrrzarr.
# OROLIFT is derived (U10·∂HGT/∂x + V10·∂HGT/∂y) using 10m winds; not a direct zarr fetch.
_VARIABLE_PATHS = {
    "APCP":     ("surface",                 "APCP_1hr_acc_fcst"),    # 1-hr QPF, mm
    "CAPE_ML":  ("255_0mb_above_ground",    "CAPE"),                 # 0-255mb ML CAPE, J/kg
    "RH_2M":    ("2m_above_ground",         "RH"),                   # 2-m RH, %
    "UGRD_700": ("700mb",                   "UGRD"),                 # 700-mb U-wind, m/s
    "VGRD_700": ("700mb",                   "VGRD"),                 # 700-mb V-wind, m/s
    "UGRD_10M": ("10m_above_ground",        "UGRD"),                 # 10-m U-wind, m/s (OROLIFT)
    "VGRD_10M": ("10m_above_ground",        "VGRD"),                 # 10-m V-wind, m/s (OROLIFT)
    "MAXREF":   ("1000m_above_ground",      "MAXREF_1hr_max_fcst"),  # 1-hr max refl., dBZ
    "MAXUVV":   ("100_1000mb_above_ground", "MAXUVV_1hr_max_fcst"),  # 1-hr max upward vel., m/s
}


def _get_hrrr_latlon() -> tuple[np.ndarray, np.ndarray]:
    """Return cached (1059, 1799) lat/lon arrays for the HRRR v4 grid."""
    # Use functools.lru_cache via module-level singleton
    global _HRRR_LATLON_CACHE
    if _HRRR_LATLON_CACHE is not None:
        return _HRRR_LATLON_CACHE

    try:
        from pyproj import Proj
    except ImportError as e:
        raise ImportError("pyproj is required for HRRR lat/lon computation") from e

    fs = s3fs.S3FileSystem(anon=True)
    # projection_x/y_coordinate arrays live at the variable group level (surface/PRATE),
    # NOT inside the inner surface/PRATE/surface subgroup which only contains the data array.
    coord_key = "hrrrzarr/sfc/20210715/20210715_12z_fcst.zarr/surface/PRATE"
    coord_store = s3fs.S3Map(coord_key, s3=fs)
    grp = zarr.open_group(coord_store, mode="r")

    x = grp["projection_x_coordinate"][:]   # metres, shape (1799,)
    y = grp["projection_y_coordinate"][:]   # metres, shape (1059,)

    p = Proj(**_PROJ_PARAMS)
    xx, yy = np.meshgrid(x, y)
    lon2d, lat2d = p(xx, yy, inverse=True)

    _HRRR_LATLON_CACHE = (lat2d.astype(np.float64), lon2d.astype(np.float64))
    return _HRRR_LATLON_CACHE


_HRRR_LATLON_CACHE = None


class AWSHRRRPatches(Source):
    """
    HRRR f0X patch source driven by a pre-built manifest JSON.

    Iterates directly over manifest entries (sample_dims = ("sample",)).
    The DataMover calls open_sample_dataset({"sample": i}) for each entry,
    bypassing the Cartesian-product constraint that required a uniform patch
    count per valid_time.

    For each manifest entry the source:
      1. Derives t0 = valid_time - fhr hours for each requested forecast_hour
      2. Opens the UofUtah HRRR Zarr at s3://hrrrzarr/sfc/YYYYMMDD/...
      3. Extracts a 96×96-cell halo at (hrrr_y0, hrrr_x0)
      4. Returns one channel per (variable, forecast_hour): APCP_f02, CAPE_ML_f02, …
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
        # _entries must exist before super().__init__() which calls __str__ → sample property
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

        self._zarr_cache: dict[tuple, zarr.Array] = {}
        self._fs: s3fs.S3FileSystem | None = None
        self._hgt_full: np.ndarray | None = None

        logger.info(
            f"AWSHRRRPatches: {len(self._entries)} samples, "
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
        orolift_deps = {"UGRD_10M", "VGRD_10M"} if needs_orolift else set()

        data_vars = {}
        for fhr in self.forecast_hours:
            t0 = t - pd.Timedelta(hours=fhr)
            fhr_idx = fhr - 1  # hrrrzarr: forecast_period index 0 = f01

            tiles = {}
            all_fetch = list(fetch_vars) + [v for v in orolift_deps if v not in fetch_vars]
            for vname in all_fetch:
                if vname not in _VARIABLE_PATHS:
                    raise ValueError(f"AWSHRRRPatches: unknown variable '{vname}'")
                level_group, zarr_vname = _VARIABLE_PATHS[vname]
                arr = self._get_zarr_array(t0, level_group, zarr_vname)
                tiles[vname] = arr[fhr_idx, y_sl, x_sl].astype(np.float32)

            if needs_orolift:
                hgt_patch = self._get_hgt_patch(y_sl, x_sl)
                dz_dy, dz_dx = np.gradient(hgt_patch.astype(np.float64), 3000.0, 3000.0)
                tiles["OROLIFT"] = (
                    tiles["UGRD_10M"] * dz_dx + tiles["VGRD_10M"] * dz_dy
                ).astype(np.float32)
                for dep in orolift_deps - set(fetch_vars):
                    del tiles[dep]

            for vname in self._base_vars:
                data_vars[f"{vname}_f{fhr:02d}"] = (["y", "x"], tiles[vname])

        hrrr_lat, hrrr_lon = _get_hrrr_latlon()
        lat2d = hrrr_lat[y_sl, x_sl].astype(np.float64)
        lon2d = hrrr_lon[y_sl, x_sl].astype(np.float64)

        # latitude, longitude, and valid_time must be data variables so that
        # expand_dims({"ensemble": [0]}) in the Anemoi target broadcasts them.
        timed_vars = {k: (["time", "y", "x"], v[1][np.newaxis]) for k, v in data_vars.items()}
        timed_vars["latitude"]   = (["y", "x"], lat2d)
        timed_vars["longitude"]  = (["y", "x"], lon2d)
        timed_vars["valid_time"] = (["time"], pd.DatetimeIndex([t]))

        xds = xr.Dataset(timed_vars, coords={"time": pd.DatetimeIndex([t])})
        xds.attrs["_sample_index"] = i
        return xds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_fs(self) -> s3fs.S3FileSystem:
        if self._fs is None:
            self._fs = s3fs.S3FileSystem(anon=True)
        return self._fs

    def _get_hgt_patch(self, y_sl: slice, x_sl: slice) -> np.ndarray:
        """Return the static terrain-height patch; downloads HGT once per process."""
        if not hasattr(self, "_hgt_full") or self._hgt_full is None:
            path = (
                f"{_BASE_URL}/20210101/20210101_00z_fcst.zarr"
                "/surface/HGT/surface/HGT"
            )
            store = s3fs.S3Map(path, s3=self._get_fs())
            arr = zarr.open_array(store, mode="r")
            self._hgt_full = arr[0].astype(np.float32)  # f01; HGT is static
        return self._hgt_full[y_sl, x_sl]

    def _get_zarr_array(
        self, t0: pd.Timestamp, level_group: str, varname: str
    ) -> zarr.Array:
        cache_key = (t0, level_group, varname)
        if cache_key in self._zarr_cache:
            return self._zarr_cache[cache_key]

        date = f"{t0.year:04d}{t0.month:02d}{t0.day:02d}"
        init = f"{t0.hour:02d}z"
        path = (
            f"{_BASE_URL}/{date}/{date}_{init}_fcst.zarr"
            f"/{level_group}/{varname}/{level_group}/{varname}"
        )
        store = s3fs.S3Map(path, s3=self._get_fs())
        arr = zarr.open_array(store, mode="r")

        # bound cache size: evict oldest entry beyond 10 open arrays
        if len(self._zarr_cache) >= 10:
            oldest = next(iter(self._zarr_cache))
            del self._zarr_cache[oldest]

        self._zarr_cache[cache_key] = arr
        return arr
