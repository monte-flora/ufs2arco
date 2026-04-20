# pylint: disable=line-too-long,trailing-whitespace
# Author: monte-flora
# Email: monte.flora@weather.com

# A source class for loading the regridded
# TWCo GRAF Reforecast dataset (2D structured grid)
# compatible with the UFS2ARCO repo

import logging
from typing import Optional, Literal
import os

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4

from ufs2arco.sources import Source
from ufs2arco.transforms.destagger import destagger
from ufs2arco.transforms.temporal_aggregation import temporal_aggregation

from .aws_graf_reforecast import AWSGRAFArchive
from .graf_utils import subsample_by_month, compute_reflectivity_3d, apply_raymond_filter_to_dataset

logger = logging.getLogger("ufs2arco")

# 51 staggered zeta levels (m) from GRAF's hybrid vertical coordinate (Table 2)
GRAF_ZETA_LEVELS_STAGGERED = np.array([
    0, 65.9983, 146.143, 248.677, 379.614, 545.524, 752.977, 1008.55,
    1318.8, 1690.31, 2129.64, 2643.37, 3238.07, 3920.31, 4696.65,
    5573.68, 6544, 7528, 8512, 9496, 10480, 11464, 12448, 13432,
    14416, 15400, 16384, 17368, 18352, 19336, 20320, 21304, 22288,
    23272, 24256, 25240, 26224, 27208, 28192, 29176, 30160, 31144,
    32128, 33112, 34096, 35080, 36064, 37048, 38032, 39016, 40000,
], dtype=np.float32)

# 50 cell-centered zeta values (midpoints of adjacent staggered levels)
GRAF_ZETA_CENTERS = 0.5 * (
    GRAF_ZETA_LEVELS_STAGGERED[:-1] + GRAF_ZETA_LEVELS_STAGGERED[1:]
)
GRAF_ZTOP = 40000.0


class AWSGRAFRegriddedArchive(AWSGRAFArchive):
    """
    Access for the TWCo GRAF Reforecast dataset regridded to a structured
    2D grid (y=1308, x=1524) covering eastern 2/3 CONUS.

    Unlike the parent AWSGRAFArchive:
    - Data is on a regular 2D grid, not an unstructured MPAS mesh
    - Time is ordered (no permutation files needed)
    - Data is on a local FUSE mount, not S3
    - Geographic subsetting uses 2D y,x index slicing instead of 1D cell masking
    """

    BUCKET = "s3://graf-reforecast-regrid/"
    horizontal_dims = ("y", "x")
    available_variables = AWSGRAFArchive.available_variables + ("physical_height", "refl_3d")

    label = BUCKET.replace("//", "").replace(":", "-").replace("/", "")
    GRAF_CASES_FILE = "/home/mflora/graf-reforecast-conus-interp/data/graf_reforecast_cases.csv"

    def __init__(
        self,
        file_freqstr: Literal["05m", "15m"],
        init_times: dict[Literal["start", "stop"], str],
        lead_times: dict[Literal["start", "stop"], str],
        variables: Optional[list] = None,
        static_variables: list = None,
        levels: Optional[list | tuple] = None,
        geographic_extent=dict[Literal["lat_min", "lat_max", "lon_min", "lon_max"], float],
        static_regridded_file_path: str = None,
        cases_csv_path: str = None,
        destagger_kwargs=None,
        temporal_aggregation_kwargs: dict = None,
        subsample_by_month_kwargs: dict = {"frac": 1.0, "seed": 42},
        raymond_filter_kwargs: dict = None,
    ) -> None:
        """
        Args:
            file_freqstr: "15m" or "05m" for 15-min or 5-min data
            init_times: dict with start/stop or dates for case selection
            lead_times: dict with start/stop timedelta strings
            variables: list of variable names to extract
            static_variables: list of static variable names from the regridded file
            levels: vertical levels to extract
            geographic_extent: dict with lat_min, lat_max, lon_min, lon_max (360° convention)
            static_regridded_file_path: path to pre-regridded static netCDF file
            destagger_kwargs: if provided, used to destagger variables
            temporal_aggregation_kwargs: if provided, used for temporal aggregation
            subsample_by_month_kwargs: random subsampling per month
        """
        self.file_freqstr = file_freqstr
        self._storage_options = {"anon": False} if self.BUCKET.startswith("s3://") else {}

        # Load GRAF cases CSV (same logic as parent)
        csv_path = cases_csv_path if cases_csv_path is not None else self.GRAF_CASES_FILE
        graf_cases_df = pd.read_csv(
            csv_path,
            index_col="init_time",
            parse_dates=["init_time"],
        )
        if "dates" in init_times:
            target_dates = pd.to_datetime(init_times["dates"])
            graf_cases_df_sub = graf_cases_df.loc[target_dates]
        else:
            graf_cases_df_sub = graf_cases_df.loc[init_times["start"]:init_times["stop"]]
        self.init_times_df = subsample_by_month(graf_cases_df_sub, **subsample_by_month_kwargs)

        self.init_time = self.init_times_df["case_str"].values
        self.init_time_dts = pd.to_datetime(self.init_times_df.index)
        self._init_time_dt_map = {
            ic: pd.to_datetime(ic.split("_")[0], format="%Y%m%d%H")
            for ic in self.init_time
        }

        self.variables = variables
        self.static_variables = static_variables
        self.levels = levels
        self.lead_times = lead_times
        self.geographic_extent = geographic_extent
        self.static_regridded_file_path = static_regridded_file_path
        self.static_file_path = static_regridded_file_path  # alias for parent __str__
        self.destagger_kwargs = destagger_kwargs
        self.temporal_aggregation_kwargs = temporal_aggregation_kwargs
        self.raymond_filter_kwargs = raymond_filter_kwargs

        self._load_static_file_regridded(static_regridded_file_path)
        self._cache_geo_extent_2d()

        self._ocean_mask = None

        # Build slices (same logic as parent, minus cell-based geo slicing)
        slices = {
            "isel": {},
            "sel": {},
        }
        if levels:
            slices["isel"]["level"] = levels
            if "smois" in variables:
                slices["isel"]["nSoilLevels"] = 0

        # Call grandparent Source.__init__() directly, skipping parent
        Source.__init__(self, variables, levels=None, use_nearest_levels=False, slices=slices)

        self._compute_valid_times()

        self.forecast_step = np.arange(self.n_steps, dtype=int) + self.forecast_offset

        self._zarr_cache_per_init_time = {}
        self._base_xds_cache_per_init_time = {}

    def _load_static_file_regridded(self, static_regridded_file_path):
        """Load pre-regridded static vars and 2D lat/lon from zarr."""
        # Load static vars from the pre-regridded netCDF
        self.static_vars = {}
        if self.static_variables:
            with netCDF4.Dataset(static_regridded_file_path, "r") as static_ds:
                for v in self.static_variables:
                    name = self.STATIC_VAR_RENAMER.get(v, v)
                    vals = np.array(static_ds.variables[v][:])
                    self.static_vars[name] = (["y", "x"], vals)
                    self.variables.append(name)

        # Load 2D lat/lon from the first case's zarr file
        # (all cases have identical lat/lon grids)
        first_case = self.init_time[0]
        zarr_path = f"{self.BUCKET}{first_case}/mpasout_{self.file_freqstr}.zarr"
        ds = xr.open_zarr(zarr_path, consolidated=False, storage_options=self._storage_options)

        lat = ds["lat"].values  # (y, x)
        lon = ds["lon"].values  # (y, x)
        ds.close()

        # Convert negative longitude to 360° convention
        lon = np.where(lon < 0, lon + 360.0, lon)

        self.lat_lon = {
            "latitude": (["y", "x"], lat),
            "longitude": (["y", "x"], lon),
        }

        self.variables += list(self.lat_lon)

    def _cache_geo_extent_2d(self):
        """
        Pre-compute 2D geographic extent using bounding y,x index ranges
        and permanently slice static variables and lat/lon to the extent.
        """
        b = self.geographic_extent
        lat = self.lat_lon["latitude"][-1]  # (y, x) array
        lon = self.lat_lon["longitude"][-1]  # (y, x) already in 360°

        mask = (
            (lat >= b["lat_min"])
            & (lat <= b["lat_max"])
            & (lon >= b["lon_min"])
            & (lon <= b["lon_max"])
        )

        # Find bounding y,x index ranges from the 2D boolean mask
        y_indices = np.where(mask.any(axis=1))[0]
        x_indices = np.where(mask.any(axis=0))[0]

        self._y_slice = slice(int(y_indices[0]), int(y_indices[-1]) + 1)
        self._x_slice = slice(int(x_indices[0]), int(x_indices[-1]) + 1)

        # Slice lat_lon and static_vars to the geographic extent
        for k, v in self.lat_lon.items():
            self.lat_lon[k] = (v[0], v[1][self._y_slice, self._x_slice])

        if self.static_vars:
            for k, v in self.static_vars.items():
                self.static_vars[k] = (v[0], v[1][self._y_slice, self._x_slice])

    def rename_coords(self, xds: xr.Dataset) -> xr.Dataset:
        """Rename coords for regridded 2D data (no nCells→cell rename)."""
        rename = {"Time": "time"}
        if "nVertLevels" in xds.dims:
            rename["nVertLevels"] = "level"
        return xds.rename(rename)

    def _build_path(self, init_time: str) -> str:
        """Build the path to a regridded GRAF zarr store."""
        return f"{self.BUCKET}{init_time}/mpasout_{self.file_freqstr}.zarr"

    def _open_zarr(self, dims: dict):
        """Open a zarr store (local or S3)."""
        zarr_path = self._build_path(dims["init_time"])
        xds = xr.open_zarr(zarr_path, consolidated=False, storage_options=self._storage_options)
        # Drop zarr-native coords that conflict with our processing:
        # - lat/lon: replaced by our latitude/longitude data vars (360° convention)
        drop_vars = [v for v in ("lat", "lon") if v in xds.coords]
        if drop_vars:
            xds = xds.drop_vars(drop_vars)
        # Reset x/y index coords to bare dims — prevents merge conflicts
        # when Anemoi target stacks (y, x) → cell2d during grid flattening
        idx_to_reset = [d for d in ("y", "x") if d in xds.indexes]
        if idx_to_reset:
            xds = xds.reset_index(idx_to_reset, drop=True)
        return xds

    def _prepare_base_xds(self, xds: xr.Dataset) -> xr.Dataset:
        """Override parent to add refl_3d component dependencies."""
        if 'refl_3d' in self.variables:
            # Temporarily inject components so parent's subsetting keeps them;
            # they get dropped later by xds[self.variables] in open_sample_dataset.
            saved = list(self.variables)
            self.variables = list(set(self.variables) | {'pressure', 'temperature', 'qs', 'qr', 'qg'})
            xds = super()._prepare_base_xds(xds)
            self.variables = saved
            return xds
        return super()._prepare_base_xds(xds)

    def add_static_vars(self, xds: xr.Dataset) -> xr.Dataset:
        """Add static variables with ["y", "x"] dims."""
        for v in self.lat_lon:
            xds[v] = self.lat_lon[v]
        for v in self.static_vars:
            xds[v] = self.static_vars[v]
        return xds

    def _compute_physical_height(self, xds: xr.Dataset) -> xr.Dataset:
        """Compute 3D physical height from MPAS hybrid zeta coordinate + terrain.

        Formula: height[k, y, x] = zeta_center[k] + surface_elevation[y, x] * (1 - zeta_center[k] / ztop)

        At the surface (zeta=0): height ≈ surface_elevation (terrain-following).
        At the top (zeta=40000): height ≈ 40000 (pure geometric).
        """
        surface_elev = xds["surface_elevation"]  # (y, x)
        n_levels = xds.sizes["level"]
        zeta_centers = xr.DataArray(
            GRAF_ZETA_CENTERS[:n_levels],
            dims=["level"],
            coords={"level": xds.coords["level"]},
        )
        height = zeta_centers + surface_elev * (1.0 - zeta_centers / GRAF_ZTOP)
        xds["physical_height"] = height.astype(np.float32)
        return xds

    def open_sample_dataset(
        self,
        dims: dict,
        open_static_vars: bool,
        cache_dir: Optional[str] = None,
    ) -> xr.Dataset:
        """Lazily open a regridded GRAF zarr and process a single forecast time step.

        Returns:
            xr.Dataset with data for the requested timestep.
        """
        step = dims["forecast_step"]
        init_time = dims["init_time"]

        if init_time in self._base_xds_cache_per_init_time:
            xds = self._base_xds_cache_per_init_time[init_time]
        else:
            xds = self._open_zarr(dims)
            xds = self._prepare_base_xds(xds)
            self._base_xds_cache_per_init_time[init_time] = xds

        valid_time = self.get_valid_time(xds, **dims)

        # Time selection (ordered data, no permutation needed)
        xds = self.select_time(xds, step)

        # Geographic subset using 2D index slices
        xds = xds.isel(y=self._y_slice, x=self._x_slice)

        # Add static variables (already pre-sliced to geo extent)
        xds = self.add_static_vars(xds)

        if self.destagger_kwargs is not None:
            xds = destagger(xds, **self.destagger_kwargs)

        # Raymond filter: remove grid-scale MPAS mesh artifacts from 3D fields
        # Must run BEFORE diagnostic variables so filtered hydrometeors
        # produce clean reflectivity.
        if self.raymond_filter_kwargs is not None:
            xds = apply_raymond_filter_to_dataset(xds, **self.raymond_filter_kwargs)

        # Diagnostic variables (comp_refl, refl_3d, geopot, etc.)
        diag_vars = {"comp_refl", "refl_3d", "geopot", "rho", "theta_m"}
        if diag_vars & set(self.variables):
            xds = self.add_diagnostic_variables(xds, self.variables)
            if "refl_3d" in self.variables and "refl_3d" not in xds.data_vars:
                xds = compute_reflectivity_3d(xds)
            xds = xds[self.variables]

        xds = self.apply_slices(xds)
        xds = self.maybe_soil_impute_over_ocean(xds)
        xds = self.maybe_rename_wind_vars(xds)

        if self.temporal_aggregation_kwargs:
            xds = temporal_aggregation(xds, **self.temporal_aggregation_kwargs)

        xds = self.maybe_rescale_cloud_cover(xds)
        xds = self.add_valid_time_to_dataset(xds, valid_time)

        # Set stack_order for Anemoi target flattening (y, x) → cell
        xds.attrs["stack_order"] = ["y", "x"]
        xds.attrs["init_time"] = dims["init_time"]
        xds.attrs["forecast_step"] = dims["forecast_step"]

        return xds
