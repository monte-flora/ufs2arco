# pylint: disable=line-too-long,trailing-whitespace
# Author: monte-flora 
# Email: monte.flora@weather.com 

# A source class for loading the 
# TWCo GRAF Reforecast dataset 
# compatiable with the UFS2ARCO repo

import logging 
from typing import Optional, Literal
import os 
import time 
from copy import copy 

import numpy as np
import pandas as pd 
import xarray as xr
import netCDF4
from pathlib import Path
import gc

from ufs2arco.sources import Source 
from ufs2arco.transforms.destagger import destagger 
from .graf_utils import (parse_order_file, 
                    get_expected_times, 
                    add_missing_times_with_nans,
                    times_with_nans, 
                    spherical_to_lat_lon,
                    subsample_by_month,
                    compute_composite_reflectivity,
                    compute_geopotential,
                    load_times_dict_json,
                    load_missing_times_parquet,
                    load_missing_indices_json     
                   )

from ufs2arco.transforms.temporal_aggregation import temporal_aggregation

# Required to load the new compressed bucket
import sys, os
sys.path.append("/data3/mflora/graf-ai/")
from grafai.utils import scale_codec

logger = logging.getLogger("ufs2arco")

    
class AWSGRAFArchive(Source):
    """
    Access for the TWCo GRAF Reforecast Zarr dataset on AWS. 
    
    Note: 
        Atmospheric data is stored with a 15-min frequency, but 
        preciptation data is stored separately with a 5-min frequency 
    """
    # Note: this list includes variables from the 15m, 05m, and the static files. 
    available_variables = (
        # 15m variables
        "bli", "cape", "ceiling_agl", "cin", "cpoice", "cporain", "cposnow",
        "dewpoint_2m", "echotop", "fwi", "hpbl", "lcl", "lh", "lh01h", "mslp",
        "olrtoa", "pdi", "precipw", "pressure", "q2", "qc", "qg", "qi", "qr",
        "qs", "qv", "sh2o", "skintemp", "smois", "snow_ratio", "snowh", "swdnb",
        "swdnb01h", "swdnbdn", "swdnbdn01h", "t2m", "temperature", "theta",
        "tpi", "tslb", "u10", "uReconstructMeridional", "uReconstructZonal",
        "v10", "visibility", "w", "windgust10m", "comp_refl",
        "geopot",
        
        # 05min variables
        "apcp_bucket", "conv_bucket",
        "prate", "ptype", "rain_bucket", "snow_bucket", "total_cloud_cover",
        "zrain_bucket",
        
        # static variables
        'surface_elevation', 'land_sea_mask', 'climo_soiltemp', 'latitude', 'longitude'
    )
    
    available_levels = list(range(50))
    
    # pd.timedelta and anemoi does not like FREQ
    # so created this constant for the anemoi.py 
    # to use to store frequency. Unfortunate, 
    # frequency cannot be directly determined from 
    # the forecast datetimes :( 
    FREQ = "15min"
    STORED_FREQ = "15m"
    
    # Deprecated bucket (time shuffled version)
    BUCKET = "s3://twc-graf-reforecast/"
    
    label = BUCKET.replace("//", "").replace(":", "-").replace("/", "")
    
    #BUCKET = "s3://twc-nvidia/graf/recompress/"
    GRAF_CASES_FILE = f"/data3/mflora/graf-ai/grafai/data/graf_reforecast_cases_{label}.csv"
    
    # JSON and parquet file containing the expected times 
    # and missing times for the old time shuffled bucket 
    expected_times_path = "/data3/mflora/graf-ai/grafai/data/expected_graf_reforecast_times.json"
    missing_times_path = "/data3/mflora/graf-ai/grafai/data/missing_graf_reforecast_times.parquet"
    missing_indices_path = "/data3/mflora/graf-ai/grafai/data/missing_indices_graf_reforecast.json"
    
    # When initializing this class, ensure that the variables
    # in sample_dims are class attributes (self.init_time=, self.forecast_step).
    # uf2arco.datamover.DataMover uses those attributes to determine sample indices,
    # batch size, etc. They are also used to determine the 
    # zarr store size. 
    # For the GRAF dataset, though init_time 
    # is not a dim of the final dataset, data is stored by 
    # init_time. The zarr files for a given init time
    # is chunked by time, so we can process it individually. 
    sample_dims = ("init_time", "forecast_step")
    horizontal_dims = ("cell",)
    
    FREQ_RENAMER = {
        "15m" : "15min",
        "05m" : "05min"
             }
    
    STATIC_VAR_RENAMER = {
        'ter' : 'surface_elevation', 
        'landmask' : 'land_sea_mask', 
        'soiltemp' : 'climo_soiltemp'
    } 
    
    WIND_VAR_RENAMER = {
      'uReconstructMeridional' : 'v',
      'uReconstructZonal' : 'u', 
    }
    
    TIMESTEP_RATIO = 15 // 5 
    
    # Imputed value for the soil moisture variables over the ocean 
    SOIL_VARS = ["smois", "tslb", "climo_soiltemp"]
    SOIL_IMPUTED_MEAN_VALUES = { 
        'climo_soiltemp' : 287.9046,
        'smois' : 0.2675,
        'tslb' : 284.4978
    }
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    def rename_coords(self, xds:xr.Dataset) -> xr.Dataset:
        rename = {
            "nCells" : "cell", 
            "Time" : "time", 
        }
        if "nVertLevels" in xds.dims:
            rename["nVertLevels"] = "level"
            
        return xds.rename(rename) 
    
    def __init__(
        self, 
        # Required inputs to create the sample dim iter 
        # used by the data mover. Goal is to create 
        # iterations. 
        # Monte: Since we set the static lead times 
        # sampled with "lead_times" below, we only 
        # need an init_time iter. 
        file_freqstr: Literal["05m", "15m"], 
        init_times : dict[Literal["start", "stop"], str],
        lead_times : dict[Literal["start", "stop"], str],
        variables : Optional[dict] = None, 
        static_variables : list = None, 
        levels : Optional[list | tuple] = None, 
        geographic_extent = dict[Literal["lat_min", "lat_max", "lon_min", "lon_max"], float],
        static_file_path : str = None, 
        destagger_kwargs = None, 
        temporal_aggregation_kwargs : dict = None,
        permutation_path_dir : str = None,
        subsample_by_month_kwargs : dict = {"frac": 1.0, "seed" :42},
    )-> None:
        """
        Args:
            init_times : dict 
                Dictionary with start and stop of a time range ( 
                e.g., "2012-01-01") to select cases in the 
                AWS GRAF S3 bucket 
            variables : dict 
                Variables to grab to grab from the 15min and 5min datasets
                keys are "15m" and "05m"
            static_variables: list of strs
                Static variables to grab from the static netcdf file. 
            levels : list/tuple
                Vertical levels to grab 
            lead_times : dict 
                Dictionary with start and top of a time range
                to select lead times to grab. Times must be
                expressed as timedeltas. 
                (e.g., lead_times = {"start" : "15min", "stop": "3h"}) 
                Can grab different lead time ranges based on initialization cycle. 
            geographic_extent : dict 
                Dictionary expressing lat/lon min and maxes
                for region to grab
            static_file_path : str
                Path to the static variables 
            destagger_kwargs : dict (default=None)
                If provided, used to destagger variables (horizontally or vertically) 
            temporal_aggregation_kwargs : dict (default=None)
                If provided, used to temporally aggregated data. 
                Primary used for aggregating the 5min data to the 15min data.
            monthly_frac : float = 0.5
                Randomly subsample data from a given month with this fraction.
        """  
        self.file_freqstr = file_freqstr
        self.permutation_path_dir = permutation_path_dir  
        
        self.expected_times_per_init = None 
        self.missing_times_per_init = None
        if "twc-graf-reforecast" in self.BUCKET:
            # These files contain the expected times 
            # and missing times in the time shuffled AWS bucket. 
            self.expected_times_per_init = load_times_dict_json(self.expected_times_path)
            self.missing_times_per_init = load_missing_times_parquet(self.missing_times_path)
            self.missing_indices_per_init = load_missing_indices_json(self.missing_indices_path)

        # This file contains the init time, forecast duration, 
        # and the case str. The init time is the index and 
        # can be indexed with a time range.
        graf_cases_df = pd.read_csv(self.GRAF_CASES_FILE,
                      index_col = "init_time",
                      parse_dates=["init_time"]
                     )
        graf_cases_df_sub = graf_cases_df.loc[init_times['start']:init_times['stop']]
        self.init_times_df = subsample_by_month(graf_cases_df_sub, **subsample_by_month_kwargs)

        # Returns the directory names in the AWS bucket
        # This is where the sample_dims are set for the 
        # eventual sample_indices used in DataMover. 
        self.init_time = self.init_times_df["case_str"].values
        
        # Returns the initial condition timestamps 
        # associated with the directory names above.
        # The init_time timestamps were set as the index.
        # Convert to strings. 
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
        self.static_file_path = static_file_path 
        self.destagger_kwargs = destagger_kwargs 
        self.temporal_aggregation_kwargs = temporal_aggregation_kwargs
        self._load_static_file(static_file_path)
        self._cache_geo_extent()
        
        self._ocean_mask = None
            
        # initial slicing based on geographic area, but 
        # below add based on lead_time and vertical levels.
        slices = { 
            "isel": {},  #{"cell" : self.geo_extent_indices},
            "sel": {},
        }
        
        if levels:
            slices["isel"]["level"] = levels
            # TODO: How to handle soil levels?!?!
            # Hardcoding this at the moment. 
            if "smois" in variables:
                slices["isel"]["nSoilLevels"] = 0 
        
        # To avoid the ufs2arco error, setting levels = None, as we'll apply the
        # level slicing later after computing composite reflectivity and geopotential height.
        super().__init__(variables, levels=None, use_nearest_levels=False, slices=slices)       
              
        # Precompute valid_time coordinate
        self._compute_valid_times()
        
        # The number of forecast steps based on the 15min timestep 
        self.forecast_step = np.arange(self.n_steps, dtype=int) + self.forecast_offset
        
        self._zarr_cache_per_init_time = {}
        self._perm_file_cache_per_init_time = {}
        self._base_xds_cache_per_init_time = {}
        self._perm_index_cache = {}

    def __len__(self):
        """Return the number of valid times"""
        return len(self.valid_times)
    
    def _cache_geo_extent(self): 
        """
        Pre-compute the geo extent mask and permanently slice static variables 
        to reduce memory usage.
        """ 
        b = self.geographic_extent 
        # lat/lon stored as (["cell"], data)
        lat, lon = self.lat_lon['latitude'][-1], self.lat_lon['longitude'][-1]
        mask = (lat >= b["lat_min"]) & (lat <= b["lat_max"]) & (lon >= b["lon_min"]) & (lon <= b["lon_max"]) 
        self.geo_extent_indices = np.nonzero(mask)[0] 
        
        # Memory Optimization: Slice the static variables in memory now
        # so we don't carry the global static fields
        for k, v in self.lat_lon.items():
            self.lat_lon[k] = (v[0], v[1][self.geo_extent_indices])
            
        if self.static_vars:
            for k, v in self.static_vars.items():
                self.static_vars[k] = (v[0], v[1][self.geo_extent_indices])
        
    def _compute_valid_times(self) -> pd.DatetimeIndex:
        """Determine the valid times for all samples in the final dataset 
        (based on the 15min timestep).
        """
        lt_start = pd.to_timedelta(self.lead_times["start"])
        lt_stop = pd.to_timedelta(self.lead_times["stop"])
        freq = pd.to_timedelta(self.FREQ)
                
        self.forecast_offset = int(lt_start / freq)
        self.n_steps = int(((lt_stop - lt_start) / freq) + 1)
    
        all_valid_times = []
        for init in self.init_times_df.index:
            start_time = init + lt_start
            times = pd.date_range(start=start_time, periods=self.n_steps, freq=freq)
            all_valid_times.append(times)
        
        # Determine the trajectory_ids, a unique integer label 
        # for each forecast ic. 
        n_ics = len(self.init_time)
        unique_ids = np.arange(n_ics)
        self.trajectory_ids = np.repeat(unique_ids, self.n_steps)
        self.trajectory_id_dict = {ic : label for label, ic in enumerate(self.init_time)}
        self.valid_times =  pd.DatetimeIndex(np.concatenate(all_valid_times))
        self.n_samples = len(self.valid_times) 
  
        assert len(self.trajectory_ids) == self.n_samples, 'Must have a trajectory id for each sample of the final dataset'

    def __str__(self) -> str:
        attrslist = ["init_time"] + [
            "lead_times",
            "variables",
            "static_variables",
            "levels",
            "geographic_extent",
            "static_file_path",
            "destagger_kwargs",
            "temporal_aggregation_kwargs",
        ] 
        
        title = f"Source: {self.name}"
        msg = f"\n{title}\n" + \
              "".join(["-" for _ in range(len(title))]) + "\n"

        for key in attrslist:
            msg += f"{key:<18s}: {getattr(self, key)}\n"

        return msg
               
    def _load_static_file(self, static_file_path):
        """Load the static netcdf file with latitude and longitude cell values"""
        with netCDF4.Dataset(static_file_path, 'r') as static_ds:
            cell_lat_rads = static_ds.variables['latCell'][:] 
            cell_lon_rads = static_ds.variables['lonCell'][:]
        
            # The MPAS lat/lon values are stored in radians. 
            # Convert back to degrees for ease of setting 
            # the geographic extent. 
            lat, lon = spherical_to_lat_lon(
                phi = cell_lon_rads,
                theta = cell_lat_rads, 
                invert_lat = False 
            )
        
            self.lat_lon = {
                # Keeping the cell latitude and longitude in degrees
                # to be consistent with anemoi framework.
                "latitude" : (["cell"], lat),
                "longitude" : (["cell"],lon) 
            }
        
            self.static_vars = {}
            if self.static_variables:
                for v in self.static_variables:
                    name = self.STATIC_VAR_RENAMER.get(v, v)
                    vals = static_ds.variables[v][:]
                    self.static_vars[name] = (["cell"], static_ds.variables[v][:])
                    self.variables.append(name) 
                
        
            # latitude/longitude added to both datasets
            # for the geographic extent. 
            self.variables += list(self.lat_lon)
                
    def _build_path(self, init_time : str) -> str:
        """Build the file path to a 15m or 05m GRAF file in AWS S3 bucket."""        
        zarr_path = os.path.join(self.BUCKET, init_time, f"mpasout_{self.file_freqstr}.zarr")
        
        perm_file = os.path.join(self.permutation_path_dir, f"{init_time}_{self.file_freqstr}.txt")
        
        return zarr_path, perm_file
    
    
    def get_valid_time(self, xds: xr.Dataset, 
                        init_time: str,
                        forecast_step: int
      ):
        """Returns the valid time w.r.t to a 15min time step"""
        init_dt = self._init_time_dt_map.get(init_time)
        if init_dt is None:
            init_dt = pd.to_datetime(init_time.split("_")[0], format="%Y%m%d%H")
        freq = pd.to_timedelta(self.FREQ)   # e.g. "15min"
        valid_time = init_dt + forecast_step * freq
        
        return valid_time 
    

    def add_valid_time_to_dataset(self, xds: xr.Dataset, valid_time) -> xr.Dataset:
        """Assign a single valid_time coordinate for a timestep."""              
        xds["valid_time"] = xr.DataArray(
            [valid_time],
            dims=("time",),
            attrs={"description": "Forecast valid time"},
        )
        return xds

    def add_level_coord(self, xds : xr.Dataset)->xr.Dataset: 
        if "level" in xds.dims:
            level_values = np.arange(xds.sizes["level"])
            xds.coords["level"] = level_values 
            
        return xds
    
    def maybe_rename_wind_vars(self, xds : xr.Dataset)->xr.Dataset:
        valid_map = {
            k: v for k, v in self.WIND_VAR_RENAMER.items() if k in xds.data_vars
        }
        if not valid_map:  # empty dict -> nothing to rename
            return xds

        return xds.rename(valid_map)
    
    def add_diagnostic_variables(self, xds, variables_to_add, vertical_dim='level'):
        """
        Add multiple diagnostic variables efficiently.
        Reuses loaded variables across computations.
        """
        if 'comp_refl' in variables_to_add and 'comp_refl' not in xds.data_vars:
            # ['pressure', 'temperature', 'qs', 'qr', 'qg', 'qv']
            xds = compute_composite_reflectivity(xds, vertical_dim=vertical_dim)
    
        if 'geopot' in variables_to_add and 'geopot' not in xds.data_vars:
            #['pressure', 'temperature', 'qv', 'surface_elevation', 'mslp', 't2m']
            target_dims = ('time', 'cell', 'level') 
            xds = compute_geopotential(xds)
            xds['geopot'] = xds['geopot'].transpose(*target_dims)
    
        return xds
    
    def get_5m_steps(self, step : int):
        """For a 15 min step, grab the 5-min steps for temporal aggregation"""
        # For the temporal aggregation of the precipitation fields 
        # we want the 15-min accumulation since the previous time step.
        # However, GRAF precip buckets are accumulated since the previous 
        # time step, so we do not include the first 5-min timestep. 
        final_step = step*self.TIMESTEP_RATIO
        return [final_step-2, final_step-1, final_step]
    
    def select_time(self, xds: xr.Dataset, forecast_step : int):
        if self.file_freqstr == "15m":
            xds = xds.isel(time=[forecast_step])
        else:
            # For the temporal aggregation of the precipitation fields 
            # we want the 15-min accumulation since the previous time step.
            # However, GRAF precip buckets are accumulated since the previous 
            # time step, so we do not include the first 5-min timestep. 
            time_idx_rng = self.get_5m_steps(forecast_step)
            xds = xds.isel(time = time_idx_rng) 
            
        return xds
    
    def get_nan_xds(self, xds : xr.Dataset)->xr.Dataset:
        # Performance update: Create new variables from coords/shapes only
        # to avoid triggering reads on the source xds and preserve dask chunks.
        new_vars = {}
        for var in xds.data_vars:
            dtype = xds[var].dtype
            if not np.issubdtype(dtype, np.floating):
                dtype = np.float32
            new_vars[var] = xr.full_like(xds[var], np.nan, dtype=dtype)
            
        return xr.Dataset(new_vars, coords=xds.coords)
    
    def select_time_with_perm(
        self,
        xds: xr.Dataset,
        perm_file: str,
        init_time: str,
        forecast_step: int,
    ) -> xr.Dataset:
        """
        Select a forecast timestep knowing the time order is shuffled.
        For the 5-min dataset, select a range of timesteps.
        """
        # Raise a deprecation warning about the time reordering.
        logger.warning(
            (
                "Using `parse_order_file` for time reordering is temporary."
                " Update workflow once new data is available"
            )
        )
        
        # time_indices are the permuted order as the data is stored.
        # time_to_perm maps logical timestamps to permuted indices.
        cache_key = (init_time, self.file_freqstr)
        cache_entry = self._perm_index_cache.get(cache_key)
        if cache_entry is None:
            ordered_times, time_indices, unordered_times = parse_order_file(perm_file)
            time_to_perm = {pd.Timestamp(t): i for i, t in enumerate(unordered_times)}
            cache_entry = {
                "time_indices": time_indices,
                "reference_idx": time_indices[0] if time_indices else 0,
                "time_to_perm": time_to_perm,
                "ordered_times": ordered_times,
            }
            self._perm_index_cache[cache_key] = cache_entry

        time_indices = cache_entry["time_indices"]
        reference_idx = cache_entry["reference_idx"]
        time_to_perm = cache_entry["time_to_perm"]

        valid_time = pd.Timestamp(self.get_valid_time(xds, init_time, forecast_step))
        
        print(forecast_step, reference_idx, time_to_perm, valid_time)
        
        if self.file_freqstr == "15m":
            # Is this forecast step missing? 
            if forecast_step in self.missing_indices_per_init[f"{init_time}_{self.file_freqstr}"]:
                xds = xds.isel(time=[reference_idx])
                xds = self.get_nan_xds(xds)
                return xds 
            else:
                perm_step = time_to_perm.get(valid_time)
                if perm_step is None:
                    xds = xds.isel(time=[reference_idx])
                    xds = self.get_nan_xds(xds)
                    return xds
                xds = xds.isel(time=[perm_step])
        else:
            # Map the 3 logical 5-min timestamps to permuted indices.
            # We accumulate over valid_time-10, valid_time-5, valid_time.
            freq_5m = pd.to_timedelta("5min")
            expected_times = [
                valid_time - 2 * freq_5m,
                valid_time - 1 * freq_5m,
                valid_time,
            ]
            
            print(expected_times) 
            
            perm_rng = []
            for t in expected_times:
                perm_idx = time_to_perm.get(pd.Timestamp(t))
                if perm_idx is None:
                    # No tolerance for missing timesteps, 
                    # if its missing return NaNs. 
                    xds = xds.isel(time=[reference_idx])
                    xds = self.get_nan_xds(xds)
                    return xds
                perm_rng.append(perm_idx)

            xds = xds.isel(time=perm_rng)
              
        return xds 

    def maybe_soil_impute_over_ocean(self, xds: xr.Dataset) -> xr.Dataset:
        """
        Imputes soil variables over the ocean with specific global mean values 
        instead of masking them with NaNs.
        """  
        if 'land_sea_mask' not in xds.data_vars:
            return xds
        
        if self._ocean_mask is None:
            # Create boolean mask: True where ocean (0), False where land (1)
            self._ocean_mask = xds["land_sea_mask"] == 0
    
        updates = {}
        for var, mean_val in self.SOIL_IMPUTED_MEAN_VALUES.items():
            if var not in xds:
                continue 

            da = xds[var]

            # Broadcast mask to match variable dimensions 
            # (Necessary because soil vars often have a depth dimension 'level' or 'k')
            ocean_mask_b = self._ocean_mask.broadcast_like(da)

            # condition (~ocean_mask_b): Where True (Land), keep original data.
            # other (mean_val): Where False (Ocean), replace with mean_val.
            updates[var] = da.where(~ocean_mask_b, other=mean_val)

        return xds.assign(**updates)
    
    def maybe_rescale_cloud_cover(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Searches for variables containing 'total_cloud_cover' and converts 
        them from percentage (0-100) to fraction (0-1).
        """
        updates = {}
        for var_name in ds.data_vars:
            if "total_cloud_cover" in var_name:
                da = (ds[var_name] / 100.0).copy()
                da.attrs = {**ds[var_name].attrs, "units": "fraction", "valid_range": "0-1"}
                updates[var_name] = da

        if not updates:
            return ds

        return ds.assign(**updates)
    
    def add_static_vars(self, xds: xr.Dataset)->xr.Dataset:
        # Add static variables (These are now already pre-sliced in __init__)
        for v in self.lat_lon:
            xds[v] = self.lat_lon[v]
        for v in self.static_vars:
            xds[v] = self.static_vars[v]
            
        return xds 
    
    def open_static_vars(self, path):
        pass
    
    def _open_zarr(self, dims: dict):
        zarr_path, perm_file = self._build_path(dims['init_time'])
        if "twc-graf-reforecast" in self.BUCKET:
            xds = xr.open_zarr(
                    zarr_path,
                    # Must be False to prevent .zmetadata issues!
                    consolidated=False,
                    storage_options=dict(anon=True),
                )
        else:
            perm_file = None
            xds = xr.open_zarr(
                    zarr_path,
                    # CAUTION: setting this explicitly as False
                    # was crucial to avoid some metadata loading issues.
                    # Unsure what the cause was! 
                    consolidated=False,
                )
            
        return xds, perm_file
    
    def _prepare_base_xds(self, xds: xr.Dataset) -> xr.Dataset:
        # Optimization: Subset variables immediately to reduce metadata overhead
        # Include diag vars components to ensure they are available for computation
        required_vars = set(self.variables)
        if 'comp_refl' in self.variables:
            required_vars.update(['pressure', 'temperature', 'qv', 'qs', 'qr', 'qg', 'mslp'])
        if 'geopot' in self.variables:
            required_vars.update(['pressure', 'temperature', 'qv', 'mslp', 't2m'])

        vars_to_keep = [v for v in required_vars if v in xds.data_vars]
        xds = xds[vars_to_keep]
        xds = self.rename_coords(xds)
        xds = self.add_level_coord(xds)
        return xds

    def open_sample_dataset(
        self, 
        dims : dict, 
        open_static_vars: bool, 
        cache_dir : Optional[str] = None,
    )-> xr.Dataset:
        """Lazily open a GRAF reforecast zarr file and process a single forecast time step""" 
        step = dims["forecast_step"]
        init_time = dims['init_time']
             
        if init_time in self._base_xds_cache_per_init_time:
            xds = self._base_xds_cache_per_init_time[init_time]
            perm_file = self._perm_file_cache_per_init_time[init_time]
        else:
            xds, perm_file = self._open_zarr(dims)
            xds = self._prepare_base_xds(xds)
            self._base_xds_cache_per_init_time[init_time] = xds
            self._perm_file_cache_per_init_time[init_time] = perm_file 
            
        valid_time = self.get_valid_time(xds,  **dims)   
        
        if "twc-graf-reforecast" in self.BUCKET:
            xds = self.select_time_with_perm(xds, perm_file, **dims)
        else:
            xds = self.select_time(xds, step)
        
        # -----------------------------------------------------------------
        # OPTIMIZATION: Horizontal Slice FIRST
        # Safe because all vars (even staggered) share the 'cell' dim.
        # -----------------------------------------------------------------
        if hasattr(self, 'geo_extent_indices'):
            xds = xds.isel(cell=self.geo_extent_indices)
        
        # Add static variables (These are now already pre-sliced in __init__)
        xds = self.add_static_vars(xds)
        
        if self.destagger_kwargs is not None:
            xds = destagger(xds, **self.destagger_kwargs)

        # its a big computation hit, but we need all vertical levels
        # to compute the comp refl and geopot correctly :( 
        if 'comp_refl' in self.variables:
            xds = self.add_diagnostic_variables(xds, self.variables)
            # Keep only the variables of interest 
            # after computing the diagnostic variables. 
            xds = xds[self.variables]     
            
        xds = self.apply_slices(xds)
        xds = self.maybe_soil_impute_over_ocean(xds)   
        xds = self.maybe_rename_wind_vars(xds)
        if self.temporal_aggregation_kwargs:
            xds = temporal_aggregation(xds, **self.temporal_aggregation_kwargs)  
        xds = self.maybe_rescale_cloud_cover(xds)
        xds = self.add_valid_time_to_dataset(xds, valid_time)
            
        # Adding this attribute to find unique trajectory id 
        # for the logical time indexing. 
        xds.attrs['init_time'] = dims['init_time'] 
        xds.attrs['forecast_step'] = dims['forecast_step']
 
        return xds
