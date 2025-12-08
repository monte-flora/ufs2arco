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
import s3fs
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
                    compute_geopotential_height
                   )

from ufs2arco.transforms.temporal_aggregation import temporal_aggregation

logger = logging.getLogger("ufs2arco")

    
class AWSGRAFArchive(Source):
    """
    Access for the TWCo GRAF Reforecast Zarr dataset on AWS. 
    
    Note: 
        Atmospheric data is stored with a 15-min frequency, but 
        preciptation data is stored separately with a 5-min frequency 
    """
    
    # Note: this list includes variables from both the 15m and 05m files. 
    available_variables = (
        # 15m variables
        "bli", "cape", "ceiling_agl", "cin", "cpoice", "cporain", "cposnow",
        "dewpoint_2m", "echotop", "fwi", "hpbl", "lcl", "lh", "lh01h", "mslp",
        "olrtoa", "pdi", "precipw", "pressure", "q2", "qc", "qg", "qi", "qr",
        "qs", "qv", "sh2o", "skintemp", "smois", "snow_ratio", "snowh", "swdnb",
        "swdnb01h", "swdnbdn", "swdnbdn01h", "t2m", "temperature", "theta",
        "tpi", "tslb", "u10", "uReconstructMeridional", "uReconstructZonal",
        "v10", "visibility", "w", "windgust10m", "composite_reflectivity",
        "geopotential_height",
        
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
    
    BUCKET = "s3://twc-graf-reforecast/"
    
    # When initializing this class, ensure that the variables
    # in sample_dims are class attributes (self.init_time=).
    # uf2arco.datamover.DataMover uses those attributes to determine sample indices,
    # batch size, etc. For the GRAF dataset, though init_time 
    # is not a dim of the final dataset, data is stored by 
    # init_time. 
    sample_dims = ("init_time",)
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
        subsample_by_month_kwargs : dict = {"frac": 1.0, "seed" :42}
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
        this_df = self.get_date_iter(
            self.BUCKET, 
            self.permutation_path_dir, 
            init_times["start"], 
            init_times["stop"])
        
        self.init_times_df = subsample_by_month(this_df, **subsample_by_month_kwargs)

        # Returns the directory names in the AWS bucket
        # This is where the sample_dims are set for the 
        # eventual sample_indices used in DataMover. 
        self.init_time = self.init_times_df["case"].values
        
        # Returns the initial condition timestamps 
        # associated with the directory names above.
        # The init_time timestamps were set as the index.
        # Convert to strings. 
        self.init_time_dts = pd.to_datetime(self.init_times_df.index)
       
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
            
        # initial slicing based on geographic area, but 
        # below add based on lead_time and vertical levels.
        slices = { 
            "isel": {"cell" : self.geo_extent_indices},
            "sel": {},
        }
        
        if levels:
            slices["isel"]["level"] = levels
            # TODO: How to handle soil levels?!?!
            # Hardcoding this at the moment. 
            if "smois" in variables:
                slices["isel"]["nSoilLevels"] = 0 
            
        super().__init__(variables, levels, use_nearest_levels=False, slices=slices)       
              
        # Precompute valid_time coordinate
        self._compute_valid_times()
    
    def __len__(self):
        """Return the number of valid times"""
        return len(self.valid_times)
    
    def _cache_geo_extent(self): 
        """Pre-compute the geo extent mask""" 
        b = self.geographic_extent 
        # lat/lon stored as (["cell"], data)
        lat, lon = self.lat_lon['latitude'][-1], self.lat_lon['longitude'][-1]
        mask = (lat >= b["lat_min"]) & (lat <= b["lat_max"]) & (lon >= b["lon_min"]) & (lon <= b["lon_max"]) 
        self.geo_extent_indices = np.nonzero(mask)[0]    
        
    def _compute_valid_times(self) -> pd.DatetimeIndex:
        """Determine the valid times for all samples in the final dataset."""
        lt_start = pd.to_timedelta(self.lead_times["start"])
        lt_stop = pd.to_timedelta(self.lead_times["stop"])
        freq = pd.to_timedelta(self.FREQ)
        n_steps = int(((lt_stop - lt_start) / freq) + 1)
    
        all_valid_times = []
        for init in self.init_times_df.index:
            start_time = init + lt_start
            times = pd.date_range(start=start_time, periods=n_steps, freq=freq)
            all_valid_times.append(times)
        
        # Determine the trajectory_ids, a unique integer label 
        # for each forecast ic. 
        n_ics = len(self.init_time)
        unique_ids = np.arange(n_ics)
        self.trajectory_ids = np.repeat(unique_ids, n_steps)
        self.trajectory_id_dict = {ic : label for label, ic in enumerate(self.init_time)}
        self.valid_times =  pd.DatetimeIndex(np.concatenate(all_valid_times))
        self.n_samples = len(self.valid_times) 
  
        assert len(self.trajectory_ids) == self.n_samples, 'Must have a trajectory id for each sample of the final dataset'

    def get_date_iter(self, 
                      bucket : str, 
                      permutation_dir: str, 
                      start : str, 
                      stop : str
    )->pd.DataFrame:
        """From AWS GRAF bucket, selects cases based on date ranges"""

        def parse_bucket_paths(casestr : str, perm_paths : list):
            """Convert AWS GRAF bucket paths to time ranges in a pd.DataFrame"""
            # Example: 2004010112_27 → "20040101" + "12"
            yyyymmdd = casestr [:8]
            hh = casestr [8:10]     # init hour
            fh = casestr .split("_")[-1]  # forecast length
            init_time_dt = pd.to_datetime(yyyymmdd + hh, format="%Y%m%d%H")
        
            return init_time_dt, int(fh), casestr, perm_paths[freqs[0]], perm_paths[freqs[1]]
    
        fs = s3fs.S3FileSystem(anon=True)
        paths = fs.ls(bucket)
    
        # With the temporary time reorderings, need to ensure 
        # that a given case has both the 15m and 05m permutation files 
        permutation_paths_dict = {}
        freqs = ["05m", "15m"]

        for path in paths:
            case = os.path.basename(path)
    
            both_exist = 0
            path_dict = {}
            for freq in freqs:
                this_path = os.path.join(permutation_dir, f"{case}_{freq}.txt")
                if os.path.exists(this_path):
                    path_dict[freq] = this_path
                    both_exist+=1
    
            if both_exist == 2:
                permutation_paths_dict[case] = path_dict

        # Parse and drop invalids
        records = [r for casestr, these_paths in permutation_paths_dict.items() 
               if (r := parse_bucket_paths(casestr, these_paths)) is not None]
        
        cols = ["init_time", "forecast_hour", "case"] + [f"{i}_perm_path" for i in freqs]  
        df = pd.DataFrame(records, columns=cols).set_index("init_time")
    
        return df.loc[start:stop] 
    
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

    def add_valid_time(self, 
                       xds: xr.Dataset,
                       freq : str, 
                       case_str : str, 
                       permutation_file_path : str,       
        )->dict:
        """
        To add a forecast valid time coordinate:
        1. ***temporary*** parse the permutation .txt file 
        2. Determine the expect times from time_resolution and fhrs 
        3. Assign the parsed time orderings
        4. Reindex with the expected times (missing times filled with NaNs)
        """
        # Raise a deprecation warning about the time reordering. 
        logger.warning(
            ("Using `parse_order_file` for time reordering is temporary."
            " Update workflow once new data is available",)
            )
        
        freq_int = int(freq.replace('min', '')) 
        
        # 2004010112_27 -> 27 
        fhrs = case_str.split('_')[-1]
        fhrs_in_mins = int(fhrs) * 60 
        # Adding +1 for the initialization time to total number of time steps. 
        n_expected_time_steps = int(fhrs_in_mins / freq_int) + 1
        expected_times = get_expected_times(xds, f"{freq_int}min", n_expected_time_steps)
        
        # times is the expected times while time indices
        # are the permuted order as the data is stored. 
        times, time_indices = parse_order_file(permutation_file_path)
        
        # Note: the time reordering is an expensive operation!
        # Add missing times for the 5-min dataset is costly. 
        xds = xds.isel(time=time_indices)
        xds = xds.assign_coords(time=times)
        xds = add_missing_times_with_nans(xds, expected_times)
        
        # Add as a variable, not a dimension coordinate
        xds["valid_time"] = xr.DataArray(
            expected_times,
            dims=("time",),
            attrs={"description": "Forecast valid time"},
        )
        
        start_td = pd.to_timedelta(self.lead_times["start"])
        stop_td = pd.to_timedelta(self.lead_times["stop"])
        
        # Compute lead times relative to first init time
        # (e.g., 0min, +15 min, +30min, and so on)
        # For simplicity, replacing the actual times 
        # with lead times for the selection process. 
        
        # Perform the slicing explicitly here rather 
        # apply_slices so its applied first. 
        init_time = xds.time.isel(time=0)
        lead_times = (xds["time"] - init_time).astype("timedelta64[s]")    
        xds = xds.assign_coords(time=lead_times)             
        xds = xds.sel(time = slice(start_td, stop_td))
        
        # Note: uncomment to log the missing data. 
        #missing_times = times_with_nans(times, expected_times)
        #if len(missing_times) > 0:
        #    logger.info(f"add_valid_times : {missing_times=}")
            
        return xds 
    
    def add_level_coord(self, xds : xr.Dataset)->xr.Dataset: 
        if "level" in xds.dims:
            level_values = np.arange(xds.sizes["level"])
            #xds = xds.assign_coords(level=("level", level_values))
            xds.coords["level"] = level_values 
            
        return xds
    
    def rename_wind_vars(self, xds : xr.Dataset)->xr.Dataset:
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
        if 'composite_reflectivity' in variables_to_add and 'composite_reflectivity' not in xds.data_vars:
            # ['pressure', 'temperature', 'qs', 'qr', 'qg', 'qv']
            xds = compute_composite_reflectivity(xds, vertical_dim=vertical_dim)
    
        if 'geopotential_height' in variables_to_add and 'geopotential_height' not in xds.data_vars:
            #['pressure', 'temperature', 'qv', 'surface_elevation']
            xds = compute_geopotential_height(xds)
    
        return xds

        
    def open_static_vars(self, path):
        pass
    
    def open_sample_dataset(
        self, 
        dims : dict, 
        open_static_vars: bool, 
        cache_dir : Optional[str] = None,
    )-> xr.Dataset:
        
        zarr_path, permutation_file_path = self._build_path(**dims) 
        
        # Note: the AWS files are chunked with Time : 1 
        xds = xr.open_zarr(
                zarr_path, 
                storage_options=dict(anon=True),
                consolidated=True, 
            )

        xds = self.rename_coords(xds)
        
        # TODO: Is this the optimal way to add the static variables? 
        # Add the latitude and longitudes
        # and optionally add static variables. 
        for v in self.lat_lon:
            xds[v] = self.lat_lon[v]
        for v in self.static_vars:
            xds[v] = self.static_vars[v]
 
        # Note: for composite reflectivity, geopotential_heights
        # added pressure, temperature, qs, qr, and qg
        # but remove them after the computation below.          
        diag_vars = ['composite_reflectivity', 'geopotential_height']
        temp_var_list = [v for v in self.variables if v not in diag_vars]
        
        if 'composite_reflectivity' in self.variables:
            # 15-min dataset
            xds = xds[temp_var_list + ['pressure', 'temperature', 'qv', 'qs', 'qr', 'qg']]
        else:
            xds = xds[self.variables] 
            
        xds = self.add_level_coord(xds) 
        
        xds = self.add_valid_time(xds, 
                                  self.FREQ_RENAMER[self.file_freqstr], 
                                  dims['init_time'],
                                  permutation_file_path
        )
            
        if self.destagger_kwargs is not None:
            ###logger.info(f"Performing destaggering")
            xds = destagger(xds, **self.destagger_kwargs) 
        
        xds = self.apply_slices(xds)
        # To save memory only compute diagnostics after the slicing
        if 'composite_reflectivity' in self.variables:
            xds = self.add_diagnostic_variables(xds, self.variables)
        
            # Keep only the variables of interest 
            # after computing the diagnostic variables. 
            xds = xds[self.variables+['valid_time']] 
            
        xds = self.rename_wind_vars(xds)
        
        if self.temporal_aggregation_kwargs:
            lat_lon_ds = xds[["latitude", "longitude"]] 
            xds = temporal_aggregation(xds, **self.temporal_aggregation_kwargs)
            # Add as a variable, not a dimension coordinate
            init_time = pd.to_datetime(dims["init_time"].split("_")[0], format="%Y%m%d%H")
            expected_times = init_time + xds.time
            
            xds["valid_time"] = xr.DataArray(
                expected_times,
                dims=("time",),
                attrs={"description": "Forecast valid time"},
            )
            
            xds["latitude"] = lat_lon_ds["latitude"]
            xds["longitude"] = lat_lon_ds["longitude"]
        
        # Drop the time coordinate values.
        # To concatenate non-unique forecast valid times,
        # using a DataArray ("valid_time"). 
        xds = xds.drop_vars(["time"])
               
        # Adding this attribute to find unique trajectory id 
        # for the logical time indexing. 
        xds.attrs['init_time'] = dims['init_time'] 

        return xds