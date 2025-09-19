# pylint: disable=line-too-long,trailing-whitespace
# Author: monte-flora 
# Email: monte.flora@weather.com 

# A source class for loading the 
# TWCo GRAF Reforecast dataset 
# compatiable with the UFS2ARCO repo

import logging 
from typing import Optional, Literal
import os 

import numpy as np
import pandas as pd 
import xarray as xr
import netCDF4
import s3fs

from ufs2arco.sources import Source 
from ufs2arco.transforms.destagger import destagger 
from .graf_utils import (parse_order_file, 
                    get_expected_times, 
                    add_missing_times_with_nans,
                    times_with_nans, 
                    spherical_to_lat_lon,
                    subsample_by_month,
                   )

logger = logging.getLogger("ufs2arco")

    
class AWSGRAFArchive(Source):
    """
    Access for the TWCo GRAF Reforecast Zarr dataset on AWS. 
    
    Note: 
        Atmospheric data is stored with a 15-min frequency, but 
        preciptation data is stored separately with a 5-min frequency 
    """
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
    # uf2arco.datamover.DataMover uses to determine sample indices,
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
        init_times : dict[Literal["start", "stop"], str],
        lead_times : dict[Literal["start", "stop"], str],
        variables : Optional[dict] = None, 
        static_variables : list = None, 
        levels : Optional[list | tuple] = None, 
        geographic_extent = dict[Literal["lat_min", "lat_max", "lon_min", "lon_max"], float],
        static_file_path : str = None, 
        destagger_kwargs = None, 
        temporal_aggregate_kwargs : dict = None,
        permutation_path_dir : str = None,
        monthly_frac : float = 0.5 
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
            temporal_aggregate_kwargs : dict (default=None)
                If provided, used to temporally aggregated data. 
                Primary used for aggregating the 5min data to the 15min data.
            monthly_frac : float = 0.5
                Randomly subsample data from a given month with this fraction.
        """  
        self.permutation_path_dir = permutation_path_dir                
        this_df = self.get_date_iter(
            self.BUCKET, 
            self.permutation_path_dir, 
            init_times["start"], 
            init_times["stop"])
        
        self.init_times_df = subsample_by_month(this_df, frac=monthly_frac)

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
        self.temporal_aggregate_kwargs = temporal_aggregate_kwargs
        self._load_static_file(static_file_path)
        self._cache_geo_extent()
        
        # Precompute valid_time coordinate
        self._compute_valid_times()
            
    def _cache_geo_extent(self):
        """Pre-compute the geo extent mask"""
        b = self.geographic_extent
        lat, lon = self.lat_lon['latitude'], self.lat_lon['longitude'] 
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
        attrslist = ["init_time", "valid_times"] + [
            #"init_time", this is the sample dim
            "lead_times",
            "variables",
            "levels",
            "geographic_extent",
            "static_file_path",
            "destagger_kwargs",
            "add_static_vars",
            "temporal_aggregate_kwargs",
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
                'latitude' : lat,
                'longitude' : lon }
        
            self.static_vars = {}
            for v in self.static_variables:
                name = self.STATIC_VAR_RENAMER.get(v, v)
                self.static_vars[name] = static_ds.variables[v][:]
        
            self.variables["15m"] += list(self.lat_lon)
            self.variables["15m"] += list(self.static_vars)

                
    def _build_path(self, init_time : str) -> str:
        """Build the file path to a 15min and 05min GRAF file on AWS."""
        # Monte: switched to anon=False on the new machine 
        # to avoid issues with using mpirun.
        #fs = s3fs.S3FileSystem(anon=False)
                               
        def make_entry(freq: str) -> tuple[str, str]:
            # Dataset path
            zarr_path = os.path.join(self.BUCKET, init_time, f"mpasout_{freq}.zarr")
            perm_file = os.path.join(self.permutation_path_dir, f"{init_time}_{freq}.txt")

            return zarr_path, perm_file

        paths = {}
        for freq in ("15m", "05m"):
            zarr_path, perm_file = make_entry(freq)
            paths[freq] = {
                "path": zarr_path,
                "permutation_path": perm_file,
            }
                            
        return paths     
        
    
    def open_static_vars(self, path):
        pass
        
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
        
        times, time_indices = parse_order_file(permutation_file_path)
        
        xds = xds.isel(time=time_indices)
        xds = xds.assign_coords(time=times)
        xds = add_missing_times_with_nans(xds, expected_times)
        
        # Add as a variable, not a dimension coordinate
        xds["valid_time"] = xr.DataArray(
            expected_times,
            dims=("time",),
            attrs={"description": "Forecast valid time"},
        )
                
        missing_times = times_with_nans(times, expected_times) 
        logger.info(f"add_valid_times : {missing_times=}")
            
        return xds 
        
    def add_level_coord(self, xds : xr.Dataset)->xr.Dataset: 
        if "level" in xds.dims:
            level_values = np.arange(xds.sizes["level"])
            xds = xds.assign_coords(level=("level", level_values))
        return xds
    
    def select_geographic_extent(self, xds:xr.Dataset)->xr.Dataset:
        """Select geographic region based on a bounding box"""
        return xds.isel(cell=self.geo_extent_indices)
    
    def select_levels(self, xds:xr.Dataset)->xr.Dataset:
        if "level" in xds.dims:
            xds = xds.isel(level=self.levels) 
        
        if "nSoilLevels" in xds.dims:
            xds = xds.isel(nSoilLevels=0)
        
        return xds

    def select_by_lead_times(self,
                             xds:xr.Dataset, 
                             start: str,
                             stop: str,                 
    )->xr.Dataset:
        """Using lead time, sub-select the forecast data"""
        # Convert inputs to timedelta
        start_td = pd.to_timedelta(start)
        stop_td = pd.to_timedelta(stop)
        
        # Compute lead times relative to first init time
        # (e.g., 0min, +15 min, +30min, and so on)
        #init_time = np.datetime64(xds.time[0], "ns")
        init_time = xds.time.isel(time=0)
        lead_times = (xds["time"] - init_time).astype("timedelta64[s]")
 
        mask = (lead_times >= start_td) & (lead_times <= stop_td)

        return xds.sel(time=xds.time[mask])

    def temporal_aggregate(self, 
                           xds: xr.Dataset, 
                           var_to_stat_mapping : dict, 
                           resample_kwargs : dict = {"time" :"15min"},
    ) -> xr.Dataset:
        data_vars = list(xds.data_vars) 
        #xds = xds.assign_coords(time=xds.valid_time)
    
        outs = []
        for stat, vars_ in var_to_stat_mapping.items():
            if not vars_:
                continue
            
            vs = [v for v in vars_ if v in data_vars]   
            if len(vs) == 0:
                continue                
            sub = xds[vs]
            
            if stat == "sum":
                outs.append(sub.resample(**resample_kwargs).sum(keep_attrs=True))
            elif stat == "max":
                outs.append(sub.resample(**resample_kwargs).max(keep_attrs=True))
            elif stat == "mean":
                outs.append(sub.resample(**resample_kwargs).mean(keep_attrs=True))
 
        xds_out = xr.merge(outs).transpose("time",...) 

        return xds_out
    
    def open_sample_dataset(
        self, 
        dims : dict, 
        open_static_vars: bool, 
        cache_dir : Optional[str] = None,
    )-> xr.Dataset:
        
        paths = self._build_path(**dims) 
        
        # Open and rename coordinates 
        xds_dict = {}
        for freq_key in paths:                        
            xds_dict[freq_key] = xr.open_zarr(
                paths[freq_key]["path"], 
                storage_options=dict(anon=True),
                consolidated=True, 
            )
       
        # Add the latitude and longitudes
        # and optionally add static variables. 
        # Only add to the 15min dataset to avoid redundancy.
        for var, values in self.lat_lon.items():
            xds_dict["15m"][var] = (['cell'], values)
        for var, values in self.static_vars.items():
            xds_dict["15m"][var] = (['cell'], values)
        
        # Add proper time and level dimensions. 
        # Perform first before adding the static variables 
        # so that aren't expanded with the new init_time dim.
        # Though Anemoi in ufs2arco will eventually 
        # expand the static variables.
        for freq in xds_dict: 
            xds_dict[freq] = self.rename_coords(xds_dict[freq])
            xds_dict[freq] = self.add_valid_time(xds_dict[freq], 
                                                 self.FREQ_RENAMER[freq], 
                                                 dims['init_time'],
                                                 paths[freq]["permutation_path"]
                                                 )
            xds_dict[freq] = self.add_level_coord(xds_dict[freq]) 
            xds_dict[freq] = xds_dict[freq][self.variables[freq] + ["valid_time"]]
        
        # Perform the temporal aggregation on the 5min dataset 
        # and concatenate with the 15min dataset 
        if self.temporal_aggregate_kwargs is not None:
            xds_dict["05m"] = self.temporal_aggregate(xds_dict["05m"], **self.temporal_aggregate_kwargs)
   
        xds = xr.merge([xds_dict["15m"], xds_dict["05m"]])

        # Select the geographic region, vertical levels, 
        # perform destaggering.
        xds = self.select_geographic_extent(xds)
        if self.destagger_kwargs is not None:
            xds = destagger(xds, **self.destagger_kwargs) 
        xds = self.select_levels(xds)  
                
        if self.lead_times is not None:
            xds = self.select_by_lead_times(xds, 
                                      self.lead_times["start"],
                                      self.lead_times["stop"],
                                      )
            
        # Drop the time coordinate values.
        # To concatenate non-unique forecast valid times,
        # using a DataArray ("valid_time")
        xds = xds.drop_vars("time")
        
        # Adding this attribute to find unique trajectory id 
        # for the logical time indexing. 
        xds.attrs['init_time'] = dims['init_time'] 

        return xds