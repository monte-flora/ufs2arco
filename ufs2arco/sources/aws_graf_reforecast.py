# Author: monte-flora 
# Email: monte.flora@weather.com 

# A source class for loading the 
# TWCo GRAF Reforecast dataset 
# compatiable with the UFS2ARCO repo
import logging 
from typing import Optional, Tuple, Literal
import warnings

import numpy as np
import pandas as pd 
import xarray as xr
import netCDF4
import os 
from datetime import datetime
import s3fs
import re

from ufs2arco.sources import Source 
from ufs2arco.transforms.destagger import destagger 

logger = logging.getLogger("ufs2arco")

# Monte: temporary function for fixing the time ordering. 
def parse_order_file(order_filename):
    with open(order_filename) as f:
        stamps = [datetime.strptime(line.strip(), "%Y-%m-%d_%H.%M.%S") for line in f]
    idx = np.argsort(stamps)
    # mapping if you need it: {stamps[i]: k for k, i in enumerate(idx)}
    return None, idx.tolist()  

def spherical_to_lat_lon(
    phi: np.ndarray,
    theta: np.ndarray,
    invert_lat: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert spherical coordinates to latitude and longitude in degrees.

    Args:
        phi : np.ndarray
            Azimuthal angle in radians (longitude-like)
        theta : np.ndarray
            Polar angle in radians (latitude-like if inverted)
        invert_lat : bool
            If True, latitude is computed as (90 - theta_deg) -> (0,180)
            If False, latitude is simply theta in degrees -> (-90, 90)

    Returns:
        lat, lon : np.ndarray, np.ndarray
            Latitude and longitude in degrees
    """
    lon = np.mod(np.rad2deg(phi), 360)
    lat_deg = np.rad2deg(theta)
    lat = 90.0 - lat_deg if invert_lat else lat_deg
    return lat, lon

def temporal_aggregate(xds: xr.Dataset, 
                       var_to_stat_mapping : dict, 
                       coarsen_kwargs : dict = dict(lead_time=3, boundary='trim', side='right'),
    ) -> xr.Dataset:
    data_vars = xds.data_vars
        
    outs = []
    for stat, vars_ in var_to_stat_mapping.items():
        if not vars_:
            continue
        vs = [v for v in vars_ if v in data_vars]   
        if len(vs) == 0:
            continue
                
        sub = xds[vs]
            
        if stat == "sum":
            outs.append(sub.coarsen(**coarsen_kwargs).sum(keep_attrs=True))
        elif stat == "max":
            outs.append(sub.coarsen(**coarsen_kwargs).max(keep_attrs=True))
        elif stat == "mean":
            outs.append(sub.coarsen(**coarsen_kwargs).mean(keep_attrs=True))
 
        xds_out = xr.merge(outs)

        xds_out = xds_out.transpose("init_time", "lead_time", ...)
        
       
        # TODO: Hardcoding these variables back in.
        xds_out['latitude'] = xds['latitude']
        xds_out['longitude'] = xds['longitude']
        
        
        return xds_out

def is_valid_basename(name: str) -> bool:
    pattern = re.compile(r"^\d{10}_\d+$")
    return bool(pattern.match(name))


def get_date_iter(bucket : str, start : str, stop : str)->pd.DataFrame:
    """From AWS GRAF bucket, selects cases based on date ranges"""

    def parse_bucket_paths(path: str):
        """Convert AWS GRAF bucket paths to time ranges in a pd.DataFrame"""
        casestr = os.path.basename(path)
        if not is_valid_basename(casestr):
            return 
        
        # Example: 2004010112_27 → "20040101" + "12"
        yyyymmdd = casestr [:8]
        hh = casestr [8:10]     # init hour
        fh = casestr .split("_")[-1]  # forecast length
        init_time = pd.to_datetime(yyyymmdd + hh, format="%Y%m%d%H")
        return init_time, int(fh), casestr 
    
    fs = s3fs.S3FileSystem()
    paths = fs.ls(bucket)
    # Parse and drop invalids
    records = [r for p in paths if (r := parse_bucket_paths(p)) is not None]
    df = pd.DataFrame(records, columns=["init_time", "forecast_hour", "case"]).set_index("init_time")
    
    return df.loc[start:stop] 
    
    

class AWSGRAFArchive(Source):
    """
    Access for the TWCo GRAF Reforecast Zarr dataset on AWS. 
    
    For more see: 
     #TODO: add details 
    
    Note: 
        Atmospheric data is stored with a 15-min frequency, but 
        preciptation data is stored separately with a 5-min frequency 
    """
    # Attribute used by the Anemoi class 
    # for creating a valid_time coordinates 
    lead_time = True 
    BUCKET = "s3://twc-graf-reforecast/"
    
    sample_dims = ("init_time",)
    horizontal_dims = ("cell",)
   # file_suffixes = ?
   # static_vars = ()
    
    MAPPER = {
        "15min" : "15m",
        "5min" : "05m"
             }
    
    @property
    def available_variables(self) -> tuple:
        return tuple(self._xds.data_vars)

    @property
    def available_levels(self) -> tuple:
        return tuple(self._xds["level"].values)
    
    def rename_coords(self, xds:xr.Dataset) -> xr.Dataset:
        rename = {"nCells" : "cell", 
                "Time" : "time", 
                }
        
        if "nVertLevels" in xds.dims:
            rename["nVertLevels"] = "level"
                
        return xds.rename(rename) 
    
    #def __str__(self):
    #    pass
        
    
    def __init__(
        self, 
        # Required inputs to create the sample dim iter 
        # used by the data mover. Goal is to create 
        # iterations. 
        # Monte: Since we set the static lead times 
        # sampled with "lead_times" below, we only 
        # need an init_time iter. 
        init_time : dict[Literal["start", "stop"], str],
        freq : Literal["15min", "5min"], 
        lead_times : dict[Literal["start", "stop"], str],
        variables : Optional[list | tuple] = None, 
        levels : Optional[list | tuple] = None, 
        geographic_extent =dict[Literal["lat_min", "lat_max", "lon_min", "lon_max"], float],
        static_file_path : str = None, 
        destagger_kwargs = None, 
        add_static_vars : bool = True,
        temporal_aggregate_kwargs : dict = None,
    )-> None:
        """
        Args:
            init_time : dict 
                Dictionary with start and stop of a time range ( 
                e.g., "2012-01-01") to select cases in the 
                AWS GRAF S3 bucket 
            freq : "15min" or "5min"
                Indicating whether to load the 15min or 5min zarr files
            variables : list of strs
                Variables to grab
            levels : list/tuple
                Vertical levels to grab 
            lead_times : dict 
                Dictionary with start and top of a time range
                to select lead times to grab. Times must be
                expressed as timedeltas 
                (e.g., lead_times = {"start" : "15min", "stop": "3h"}) 
            geographic_extent : dict 
                Dictionary expressing lat/lon min and maxes
                for region to grab
            static_file_path : str
                Path to the static variables 
            destagger_kwargs : dict (default=None)
                If provided, used to destagger variables (horizontally or vertically) 
            add_static_vars : bool (default=True) 
                Whether to add static variables to output dataset 
            temporal_aggregate_kwargs : dict (default=None)
                If provided, used to temporally aggregated data. 
                Primary used for aggregating the 5min data to the 15min data.
        """
        
        if lead_times is None:
            lead_times = {} 
            
        dts = get_date_iter(
            self.BUCKET, init_time["start"], 
            init_time["stop"])
        
        # Returns the directory names in the AWS bucket
        self.init_time = dts["case"].values
        # Returns the initial condition timestamps 
        # associated with the directory names above.
        # The init_time timestamps were set as the index.
        # Convert to strings. 
        self.init_time_dts = pd.to_datetime(dts.index)
        
        self.freq = freq 
        
        self.variables = variables
        self.levels = levels 
        self.lead_times = lead_times 
        self.geographic_extent = geographic_extent
        self.static_file_path = static_file_path 
        self.destagger_kwargs = destagger_kwargs 
        self.add_static_vars = add_static_vars
        self.temporal_aggregate_kwargs = temporal_aggregate_kwargs
        self._load_static_file(static_file_path)
               
    def _load_static_file(self, static_file_path):
        """Load the static netcdf file with latitude and longitude cell values"""
        static_nc = netCDF4.Dataset(static_file_path, 'r')
        cell_lat_rads = static_nc.variables['latCell'][:] 
        cell_lon_rads = static_nc.variables['lonCell'][:]
        
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
        
        self.static_vars = {
            'surface_elevation' : static_nc.variables['ter'][:],
            'land_sea_mask' : static_nc.variables['landmask'][:],
        }
        
        self.variables += list(self.lat_lon.keys())
        if self.add_static_vars:
            self.variables += list(self.static_vars.keys())

                
    def _build_path(self, init_time : str) -> str:
        """Build the file path to a GRAF file on AWS."""
        freq = self.MAPPER[self.freq]
        
        fullpath = os.path.join(self.BUCKET, init_time, f"mpasout_{freq}.zarr")
        logger.debug(f"{self.name}._build_path: reading {fullpath}")
        
        fs = s3fs.S3FileSystem(anon=True)
        permutation_file = f"{init_time}_{freq}.txt"
        fs.download(os.path.join(self.BUCKET, "permutations", permutation_file), '.')
        
        return fullpath, permutation_file 
    
    # Monte: temporary function for fixing the time ordering. 
    def parse_order_file(order_filename):
        with open(order_filename) as f:
            stamps = [datetime.strptime(line.strip(), "%Y-%m-%d_%H.%M.%S") for line in f]
        idx = np.argsort(stamps)
        # mapping if you need it: {stamps[i]: k for k, i in enumerate(idx)}
        return None, idx.tolist()  
    
    def open_static_vars(self, path):
        pass
        
    def add_init_and_lead_dims(self, xds: xr.Dataset, time_resolution:str) -> xr.Dataset:
        """Swap "time" dim for ("init_time", "lead_time") """

        # Build an absolute time coordinate from attrs (same as before)
        start_timestamp = xds.attrs['config_start_time']
        start_time_dt = datetime.strptime(start_timestamp, '%Y-%m-%d_%H:%M:%S')
        start_time = pd.Timestamp(start_time_dt)
        num_time_points = xds.sizes["time"]
        time_range = pd.date_range(start=start_time, periods=num_time_points, freq=time_resolution)

        xds = xds.assign_coords(time=("time", time_range))
    
        # Convert 'time' dimension to timedeltas from the first time point
        # So it is 'lead time' since the first time steps (e.g., 0min, +15 min, +30min, and so on)
        init_time = np.datetime64(time_range[0], 'ns')
        lead_time = (xds["time"] - init_time).astype("timedelta64[ns]")  

        xds = xds.assign_coords(lead_time=("time", lead_time.data))
        xds = xds.swap_dims({"time" : "lead_time"})
        
        # Add a length-1 init_time dimension/coord
        xds = xds.expand_dims("init_time", axis=0)
        xds = xds.assign_coords(init_time=("init_time", [np.datetime64(init_time, 'ns')]))
        
        xds = xds.drop_vars("time")
        
        return xds.transpose("init_time", "lead_time", ...)
    
    def add_level_coord(self, xds : xr.Dataset)->xr.Dataset: 
        if "level" in xds.dims:
            level_values = np.arange(xds.sizes["level"])
            xds = xds.assign_coords(level=("level", level_values))
        return xds
    
    def select_geographic_extent(self, xds:xr.Dataset)->xr.Dataset:
        # Monte: checked and this code works as expected. 
        """Select geographic region based on a bounding box"""
        b = self.geographic_extent
        lat, lon = xds.latitude.data, xds.longitude.data  # dask arrays are fine
        mask = (lat >= b["lat_min"]) & (lat <= b["lat_max"]) & (lon >= b["lon_min"]) & (lon <= b["lon_max"])
        idx = np.nonzero(mask)[0]  # integer positions (lazy-friendly)
        
        # TODO: cache the idx for speed-up over multiple calls.
        return xds.isel(cell=idx)
    
    def select_levels(self, xds:xr.Dataset)->xr.Dataset:
        if "level" in xds.dims:
            xds = xds.isel(level=self.levels) 
        
        return xds
    
    def open_sample_dataset(
        self, 
        dims : dict, 
        open_static_vars: bool, 
        cache_dir : Optional[str] = None,
    )-> xr.Dataset:
     
        self.stored_freq = "15min"
    
        path, permutation_path = self._build_path(**dims) 
        
        # Open and rename coordinates 
        xds = xr.open_zarr(
            path, 
            # Crucial to have {} rather than None
            # to lazily load data as dask arrays.
            #chunks={},
            storage_options=dict(anon=True),
            consolidated=True, 
            decode_timedelta=True
        )
        
        # Raise a deprecation warning about the time reordering. 
        warnings.warn(
            "Using `parse_order_file` for time reordering is deprecated "
            "Update workflow once new data is available",
            DeprecationWarning,
            stacklevel=2  # points warning at caller, not inside this function
            )
        
        _, time_indices = parse_order_file(permutation_path)
        xds = xds.isel(Time=time_indices)
        
        xds = self.rename_coords(xds)
        
        # Add init_time, lead_time, and level dimensions. 
        # Perform first before adding the static variables 
        # so that aren't expanded with the new init_time dim.
        # Though Anemoi in ufs2arco will eventually 
        # expand the static variables. 
        xds = self.add_init_and_lead_dims(xds, self.freq)
        xds = self.add_level_coord(xds) 
        
        # Add the latitude and longitudes
        # and optionally add static variables. 
        for var in self.lat_lon.keys():
            xds[var] = (['cell'], self.lat_lon[var])
        
        if self.add_static_vars:
            for var in self.static_vars.keys():
                xds[var] = (['cell'], self.static_vars[var])
                
        # Performing the sub-selecting: 
        # variables, geographic region, levels
        xds = xds[self.variables]
        xds = self.select_geographic_extent(xds)

        # Performing the sub-selecting: 
        # variables, geographic region, levels
        xds = xds[self.variables]
        xds = self.select_geographic_extent(xds)
              
        # Perform destaggering after adding the level dim.
        # and then sub-selecting levels
        if self.destagger_kwargs is not None:
            xds = destagger(xds, **self.destagger_kwargs) 
           
        xds = self.select_levels(xds)  
        
        if self.temporal_aggregate_kwargs is not None:
            xds = temporal_aggregate(xds, **self.temporal_aggregate_kwargs)
        
        if self.lead_times is not None:
            xds = xds.sel(
                    lead_time=slice(pd.to_timedelta(self.lead_times["start"]), 
                                    pd.to_timedelta(self.lead_times["stop"]))               
        )
           
        return xds 
        
        
        
        
        
        
        
              
    

