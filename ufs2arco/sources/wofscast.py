# pylint: disable=line-too-long,trailing-whitespace
# Author: monte-flora
# Email: monte.flora@weather.com

# A source class for loading WoFSCast zarr files
# compatible with the UFS2ARCO repo

import logging
import re
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ufs2arco.sources.base import Source

logger = logging.getLogger("ufs2arco")


class WoFSCastArchive(Source):
    """
    Source class for WoFSCast zarr datasets.

    Each zarr file contains N timesteps. Timestep 0 is the initial
    condition and is skipped. Timesteps 1..N-1 are used as forecast
    outputs, giving N-1 samples per file. The timestep count is
    auto-detected from the first zarr file.

    Note:
        RAIN_AMOUNT has dim order (lat, lon, time) and must be
        transposed before time selection.
        XLAND has non-integer values from WRF compression and
        is rounded to integers.
        Longitude is stored as positive values representing
        West longitude — negate via config flag.
    """

    available_variables = (
        "T",
        "U",
        "V",
        "W",
        "QVAPOR",
        "GEOPOT",
        "COMPOSITE_REFL_10CM",
        "T2",
        "UP_HELI_MAX",
        "RAIN_AMOUNT",
        # Static variables
        "HGT",
        "XLAND",
        # Coordinate variables added by this source
        "latitude",
        "longitude",
    )

    available_levels = list(range(17))

    sample_dims = ("file_index", "time_index")
    horizontal_dims = ("cell",)

    FREQ = "10min"
    STORED_FREQ = "10m"

    VAR_RENAMER = {
        "T": "temperature",
        "QVAPOR": "qv",
        "GEOPOT": "geopot",
        "COMPOSITE_REFL_10CM": "comp_refl",
        "T2": "t2m",
        "UP_HELI_MAX": "up_heli_max",
        "RAIN_AMOUNT": "rain_amount",
        "HGT": "surface_elevation",
        "XLAND": "land_sea_mask",
        "U": "u",
        "V": "v",
        "W": "w",
    }

    # Pattern to extract datetime from filename:
    # wrfwof_2019-04-30_200000_to_2019-04-30_203000__10min__ens_mem_01.zarr
    _FILENAME_RE = re.compile(r"wrfwof_(\d{4}-\d{2}-\d{2}_\d{6})_to_")

    def __init__(
        self,
        data_dir: str,
        years: list,
        variables: Optional[list] = None,
        levels: Optional[list] = None,
        convert_lon_to_360: bool = False,
        static_variables: Optional[list] = None,
    ) -> None:
        """
        Args:
            data_dir : str
                Root directory containing year subdirectories with zarr files.
            years : list
                Years to include (e.g., [2019, 2020]).
            variables : list, optional
                Variables to load. If None, loads all available.
            levels : list, optional
                Vertical levels to select.
            convert_lon_to_360 : bool
                If True, convert degrees-west longitude to the 0-360
                convention via ``360 - lon``.
            static_variables : list, optional
                Static variables (e.g., ["HGT", "XLAND"]) to load once
                and attach to every sample.
        """
        self.data_dir = data_dir
        self.years = years
        self.convert_lon_to_360 = convert_lon_to_360
        self.static_variable_names = static_variables or []

        # Discover all zarr file paths
        file_paths = []
        for year in sorted(years):
            pattern = f"{data_dir}/{year}/wrfwof_*.zarr"
            paths = sorted(glob(pattern))
            file_paths.extend(paths)

        if len(file_paths) == 0:
            raise FileNotFoundError(
                f"No zarr files found for years={years} in {data_dir}"
            )

        self._file_paths = file_paths
        n_files = len(file_paths)

        # Auto-detect number of timesteps from first file
        ref_ds = xr.open_zarr(file_paths[0])
        n_timesteps = len(ref_ds.time)

        logger.info(
            f"WoFSCastArchive: Found {n_files} zarr files across years {years}, "
            f"{n_timesteps} timesteps per file ({n_timesteps - 1} usable)"
        )

        # Sample dims: file_index x time_index
        # Skip timestep 0 (initial condition), use 1..n_timesteps-1
        self.file_index = np.arange(n_files, dtype=int)
        self.time_index = np.arange(1, n_timesteps, dtype=int)

        # Parse start datetime from each filename (avoid opening files)
        self._start_datetimes = []
        for fp in file_paths:
            fname = fp.split("/")[-1]
            m = self._FILENAME_RE.search(fname)
            if m is None:
                raise ValueError(f"Could not parse datetime from filename: {fname}")
            dt_str = m.group(1)  # e.g., "2019-04-30_200000"
            dt = pd.Timestamp(dt_str.replace("_", "T")[:len("2019-04-30T20:00:00")].replace(
                dt_str[11:], dt_str[11:13] + ":" + dt_str[13:15] + ":" + dt_str[15:17]
            ))
            self._start_datetimes.append(dt)

        # Compute valid_times: for each file, n_timesteps-1 datetimes
        freq = pd.to_timedelta(self.FREQ)
        all_valid_times = []
        for start_dt in self._start_datetimes:
            for ti in self.time_index:
                all_valid_times.append(start_dt + ti * freq)

        self.valid_times = pd.DatetimeIndex(all_valid_times)
        self.n_samples = len(self.valid_times)

        # Trajectory IDs: each file is one trajectory with n_timesteps-1 usable timesteps
        self.trajectory_ids = np.repeat(np.arange(n_files, dtype=int), n_timesteps - 1)
        self.trajectory_id_dict = {i: i for i in range(n_files)}

        # Forecast stepping info
        self.n_steps = n_timesteps - 1
        self.forecast_offset = 1

        # Use ref_ds (already opened above) to read lat/lon and static variables
        lat_1d = ref_ds.coords["lat"].values
        lon_1d = ref_ds.coords["lon"].values

        # The raw WoFS zarr lon is stored as degrees-west (abs of negative
        # WRF lon), with values increasing westward (79.5 → 84.2). Data
        # arrays have index 0 at the western edge. Reverse lon to align
        # with the data order, then convert to 0-360 convention.
        lon_1d = lon_1d[::-1]

        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

        if convert_lon_to_360:
            lon_2d = 360.0 - lon_2d

        # Flatten 2D meshgrid to 1D cell arrays (like GRAF source)
        self._lat_cell = lat_2d.astype(np.float32).ravel()
        self._lon_cell = lon_2d.astype(np.float32).ravel()
        self._grid_shape = lat_2d.shape  # (nlat, nlon) for reshaping data

        # Cache static variables from first file (take timestep 0), flattened to cell
        self._static_data = {}
        for sv in self.static_variable_names:
            if sv not in ref_ds.data_vars:
                raise ValueError(f"Static variable '{sv}' not found in zarr file")
            data = ref_ds[sv].isel(time=0).values.copy()
            if sv == "XLAND":
                data = np.round(data).astype(np.int32)
            self._static_data[sv] = data.ravel()

        ref_ds.close()

        # Build the variable list including static + lat/lon
        if variables is None:
            variables = list(self.available_variables)
        else:
            variables = list(variables)

        # Add static variables and lat/lon to the variable list
        for sv in self.static_variable_names:
            if sv not in variables:
                variables.append(sv)
        if "latitude" not in variables:
            variables.append("latitude")
        if "longitude" not in variables:
            variables.append("longitude")

        # Level slicing
        slices = {}
        if levels is not None:
            slices["isel"] = {"level": levels}

        super().__init__(variables, levels=None, use_nearest_levels=False, slices=slices)

        # Cache for open zarr datasets (avoid reopening same file)
        self._zarr_cache = {}

    def __len__(self):
        return len(self.valid_times)

    def __str__(self) -> str:
        title = f"Source: {self.name}"
        msg = f"\n{title}\n" + "-" * len(title) + "\n"
        msg += f"{'data_dir':<22s}: {self.data_dir}\n"
        msg += f"{'years':<22s}: {self.years}\n"
        msg += f"{'n_files':<22s}: {len(self._file_paths)}\n"
        msg += f"{'n_samples':<22s}: {self.n_samples}\n"
        msg += f"{'variables':<22s}: {self.variables}\n"
        msg += f"{'static_variables':<22s}: {self.static_variable_names}\n"
        msg += f"{'convert_lon_to_360':<22s}: {self.convert_lon_to_360}\n"
        return msg

    def get_valid_time(self, file_index: int, time_index: int) -> pd.Timestamp:
        """Compute valid datetime from file start time and time index."""
        start_dt = self._start_datetimes[file_index]
        freq = pd.to_timedelta(self.FREQ)
        return start_dt + time_index * freq

    def _flatten_grid(self, xds: xr.Dataset) -> xr.Dataset:
        """
        Flatten (lat, lon) -> (cell,)

        (time, level, lat, lon) -> (time, level, cell)
        (time, lat, lon) -> (time, cell)

        Args:
            xds (xr.Dataset): with expanded grid

        Returns:
            xds (xr.Dataset): with grid flattened to "cell"
        """
        nds = xds.stack(cell2d=xds.attrs["stack_order"])
        nds["cell"] = xr.DataArray(
            np.arange(len(nds["cell2d"])),
            coords=nds["cell2d"].coords,
            dims=nds["cell2d"].dims,
            attrs={
                "description": "logical index for 'cell2d', which is a flattened lon x lat array",
            },
        )
        nds = nds.swap_dims({"cell2d": "cell"})

        # For some reason, there's a failure when trying to store this multi-index
        # it's not needed in Anemoi, so no need to keep it anyway.
        nds = nds.drop_vars("cell2d")

        return nds

    def open_sample_dataset(
        self,
        dims: dict,
        open_static_vars: bool,
        cache_dir: Optional[str] = None,
    ) -> xr.Dataset:
        """Open a single sample from a WoFSCast zarr file.

        Args:
            dims : dict
                Must contain "file_index" and "time_index" keys.
            open_static_vars : bool
                Whether to include static variables.
            cache_dir : str, optional
                Not used for local zarr files.

        Returns:
            xr.Dataset with selected timestep, variables, and coordinates.
        """
        file_idx = dims["file_index"]
        time_idx = dims["time_index"]
        file_path = self._file_paths[file_idx]

        # Open zarr with caching (same file accessed for 3 timesteps)
        if file_path in self._zarr_cache:
            ds = self._zarr_cache[file_path]
        else:
            # Evict old cache entries to limit memory
            if len(self._zarr_cache) > 10:
                self._zarr_cache.clear()
            ds = xr.open_zarr(file_path)
            self._zarr_cache[file_path] = ds

        # Transpose RAIN_AMOUNT from (lat, lon, time) to (time, lat, lon)
        if "RAIN_AMOUNT" in ds.data_vars:
            if ds["RAIN_AMOUNT"].dims == ("lat", "lon", "time"):
                ds["RAIN_AMOUNT"] = ds["RAIN_AMOUNT"].transpose("time", "lat", "lon")

        # Select single timestep
        xds = ds.isel(time=[time_idx])

        # Select requested dynamic variables (exclude static/coord vars for now)
        dynamic_vars = [
            v for v in self.variables
            if v in xds.data_vars and v not in self.static_variable_names
        ]
        xds = xds[dynamic_vars]

        # Apply level slicing
        xds = self.apply_slices(xds)

        # Flatten (lat, lon) grid to (cell,) using xarray stack
        xds.attrs["stack_order"] = ["lat", "lon"]
        xds = self._flatten_grid(xds)

        # Drop residual lat/lon coordinates from the stacked multiindex
        # to avoid merge conflicts in the Anemoi target
        xds = xds.drop_vars(["lat", "lon"], errors="ignore")

        # Rename variables to standard names
        rename_map = {
            k: v for k, v in self.VAR_RENAMER.items()
            if k in xds.data_vars and k not in self.static_variable_names
        }
        if rename_map:
            xds = xds.rename(rename_map)

        # Add static variables from cached data (already flattened to cell)
        for sv in self.static_variable_names:
            if sv in self._static_data:
                renamed = self.VAR_RENAMER.get(sv, sv)
                xds[renamed] = xr.DataArray(
                    self._static_data[sv],
                    dims=("cell",),
                )

        # Add 1D cell latitude/longitude (flattened from 2D meshgrid)
        xds["latitude"] = xr.DataArray(self._lat_cell, dims=("cell",))
        xds["longitude"] = xr.DataArray(self._lon_cell, dims=("cell",))

        # Compute and add valid_time coordinate
        valid_time = self.get_valid_time(file_idx, time_idx)
        xds["valid_time"] = xr.DataArray(
            [valid_time],
            dims=("time",),
            attrs={"description": "Forecast valid time"},
        )

        # Metadata for trajectory tracking
        xds.attrs["file_index"] = file_idx
        xds.attrs["time_index"] = time_idx

        # Global sample index for unique time slot assignment
        # (valid_times are non-unique across ensemble members)
        n_timesteps = len(self.time_index)
        xds.attrs["_sample_index"] = file_idx * n_timesteps + list(self.time_index).index(time_idx)

        return xds
