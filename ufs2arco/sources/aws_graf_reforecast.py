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
                       resample_kwargs : dict = dict(time="15min"),
    ) -> xr.Dataset:
    data_vars = list(xds.data_vars) 
    xds = xds.assign_coords(time=xds.valid_time.values)
    
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
 
        xds_out = xr.merge(outs)
        xds_out = xds_out.transpose("time", ...)
        
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
        init_time_dt = pd.to_datetime(yyyymmdd + hh, format="%Y%m%d%H")
        
        return init_time_dt, int(fh), casestr 
    
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
    FREQ = "15min"
    
    # pd.timedelta and anemoi does not like FREQ
    # so created this constant for the anemoi.py 
    # to use to store frequency. Unfortunate, 
    # frequency cannot be directly determined from 
    # the forecast datetimes :( 
    STORED_FREQ = "15m"
    
    BUCKET = "s3://twc-graf-reforecast/"
    
    # Monte: Though we are using a single time dimension, 
    # data loading and hence the sample dim is the model run init time.
    sample_dims = ("init_time",)
    horizontal_dims = ("cell",)
   # file_suffixes = ?
   # static_vars = ()
    
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
        rename = {"nCells" : "cell", 
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
        init_time : dict[Literal["start", "stop"], str],
        lead_times : dict[Literal["start", "stop"], str],
        variables : Optional[dict] = None, 
        static_variables : list = None, 
        levels : Optional[list | tuple] = None, 
        geographic_extent = dict[Literal["lat_min", "lat_max", "lon_min", "lon_max"], float],
        static_file_path : str = None, 
        destagger_kwargs = None, 
        temporal_aggregate_kwargs : dict = None,
    )-> None:
        """
        Args:
            init_time : dict 
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
        """  
        if lead_times is None:
            lead_times = {} 
                
        self.init_times_df = get_date_iter(
            self.BUCKET, init_time["start"], 
            init_time["stop"])
        
        self.init_time_dict = init_time 
        
        # Returns the directory names in the AWS bucket
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
        
        # Precompute valid_time coordinate
        self.valid_times = self._compute_valid_times()
    '''
    def _compute_valid_times(self) -> pd.DatetimeIndex:
        """Expand init_times × lead_times into a flat 1D valid_time index."""

        print(self.init_times_df)
        
        # Parse lead_time start/stop into Timedelta
        lt_start = pd.to_timedelta(self.lead_times["start"])
        lt_stop = pd.to_timedelta(self.lead_times["stop"])
        freq = pd.to_timedelta(self.FREQ)

        # Number of forecast steps per init_time
        n_steps = int(((lt_stop - lt_start) / freq) + 1)


        
        all_valid_times = []
        for init in self.init_times_df.index:
            # start forecast clock from init + lt_start
            start_time = init + lt_start
            times = pd.date_range(start=start_time, periods=n_steps, freq=freq)
            all_valid_times.append(times)

        # Flatten into one continuous index
        return pd.DatetimeIndex(np.concatenate(all_valid_times))
    '''
    def _compute_valid_times(self) -> pd.DatetimeIndex:
        """Expand init_times × cycle-based lead_times into a flat 1D valid_time index."""
        freq = pd.to_timedelta(self.FREQ)
        all_valid_times = []
        
        for init, row in self.init_times_df.iterrows():
            # extract cycle from "case" column (e.g., '2004010112_27')
            cycle = str(row["case"].split('_')[0][-2:])  # last 2 chars before underscore

            # Parse cycle-specific lead time range
            lt_cfg = self.lead_times.get(cycle, None)
            if lt_cfg is None:
                start = "6h"
                stop = "12h"
                lt_start = pd.to_timedelta("6h")
                lt_stop = pd.to_timedelta("12h")
            else:
                lt_start = pd.to_timedelta(lt_cfg["start"])
                lt_stop = pd.to_timedelta(lt_cfg["stop"])

            # number of forecast steps for this init_time
            n_steps = int(((lt_stop - lt_start) / freq) + 1)

            # valid times for this init_time
            start_time = init + lt_start
            times = pd.date_range(start=start_time, periods=n_steps, freq=freq)
            all_valid_times.append(times)

        # Flatten into one continuous index
        return pd.DatetimeIndex(np.concatenate(all_valid_times))
    
    
    def __str__(self) -> str:
        attrslist = ["init_time", "valid_times"] + [
            #"init_time", this is the sample dim
            "init_time_dict",
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
        
            self.variables["15m"] += list(self.lat_lon.keys())
            self.variables["15m"] += list(self.static_vars.keys())

                
    def _build_path(self, init_time : str) -> str:
        """Build the file path to a 15min and 05min GRAF file on AWS."""
        fs = s3fs.S3FileSystem(anon=True)
                               
        def make_entry(freq: str) -> tuple[str, str]:
            # Dataset path
            zarr_path = os.path.join(self.BUCKET, init_time, f"mpasout_{freq}.zarr")

            # Permutation file
            perm_file = f"{init_time}_{freq}.txt"
            fs.download(os.path.join(self.BUCKET, "permutations", perm_file), ".")

            return zarr_path, perm_file

        paths = {}
        for freq in ("15m", "05m"):
            zarr_path, perm_file = make_entry(freq)
            paths[freq] = {
                "path": zarr_path,
                "permutation_path": perm_file,
            }
                            
        return paths     
        
    # Monte: temporary function for fixing the time ordering. 
    def parse_order_file(order_filename):
        with open(order_filename) as f:
            stamps = [datetime.strptime(line.strip(), "%Y-%m-%d_%H.%M.%S") for line in f]
        idx = np.argsort(stamps)
        # mapping if you need it: {stamps[i]: k for k, i in enumerate(idx)}
        return None, idx.tolist()  
    
    def open_static_vars(self, path):
        pass
        
    def add_valid_time(self, xds: xr.Dataset, time_resolution:str) -> xr.Dataset:
        """Add a proper datetime coord for the time dimension """
        # The start time is the valid time of the initial conditions.
        # We can pull it from the dataset attributes. 
        start_timestamp = xds.attrs['config_start_time']
        start_time_dt = datetime.strptime(start_timestamp, '%Y-%m-%d_%H:%M:%S')
        start_time = pd.Timestamp(start_time_dt)
        num_time_points = xds.sizes["time"]
        valid_times = pd.date_range(start=start_time, periods=num_time_points, freq=time_resolution)
        
        #xds = xds.assign_coords(time=("time", time_range))
        
        # Add as a variable, not a dimension coordinate
        xds["valid_time"] = xr.DataArray(
            valid_times,
            dims=("time",),
            attrs={"description": "Forecast valid time"},
        )
                
        # Ensure that time is the first dimension. 
        return xds.transpose("time", ...)
    
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
        init_time = np.datetime64(xds.time.values[0], "ns")
        lead_times = (xds["time"].values - init_time).astype("timedelta64[s]").astype("timedelta64[ns]")
 
        mask = (lead_times >= start_td) & (lead_times <= stop_td)

        return xds.sel(time=xds.time[mask])
    
    def apply_temp_time_reordering(self, xds_dict : dict, paths : dict)->dict:
       # Raise a deprecation warning about the time reordering. 
        warnings.warn(
            "Using `parse_order_file` for time reordering is deprecated "
            "Update workflow once new data is available",
            DeprecationWarning,
            stacklevel=2  # points warning at caller, not inside this function
            )
        for freq_key in xds_dict.keys():
            permutation_path = paths[freq_key]["permutation_path"]
            _, time_indices = parse_order_file(permutation_path)
            xds_dict[freq_key] = xds_dict[freq_key].isel(Time=time_indices)
            # Delete the permutation path
            logger.info(f"Deleting {permutation_path}...")
            os.remove(permutation_path)
            
        return xds_dict
    
    def open_sample_dataset(
        self, 
        dims : dict, 
        open_static_vars: bool, 
        cache_dir : Optional[str] = None,
    )-> xr.Dataset:
         
        paths = self._build_path(**dims) 
        
        # Open and rename coordinates 
        xds_dict = {}
        for freq_key in paths.keys():                        
            xds_dict[freq_key] = xr.open_zarr(
                paths[freq_key]["path"], 
                storage_options=dict(anon=True),
                consolidated=True, 
                decode_timedelta=True
            )
        
        xds_dict = self.apply_temp_time_reordering(xds_dict, paths)
        
        # Add the latitude and longitudes
        # and optionally add static variables. 
        # Only add to the 15min dataset to avoid redundancy.
        for var in self.lat_lon.keys():
            xds_dict["15m"][var] = (['cell'], self.lat_lon[var])
        for var in self.static_vars.keys():
            xds_dict["15m"][var] = (['cell'], self.static_vars[var])
        
        # Add proper time and level dimensions. 
        # Perform first before adding the static variables 
        # so that aren't expanded with the new init_time dim.
        # Though Anemoi in ufs2arco will eventually 
        # expand the static variables.
        for freq in xds_dict.keys(): 
            xds_dict[freq] = self.rename_coords(xds_dict[freq])
            xds_dict[freq] = self.add_valid_time(xds_dict[freq], self.FREQ_RENAMER[freq])
            xds_dict[freq] = self.add_level_coord(xds_dict[freq]) 
            xds_dict[freq] = xds_dict[freq][self.variables[freq] + ["valid_time"]]
        
        # Perform the temporal aggregation on the 5min dataset 
        # and concatenate with the 15min dataset 
        if self.temporal_aggregate_kwargs is not None:
            xds_dict["05m"] = temporal_aggregate(xds_dict["05m"], **self.temporal_aggregate_kwargs)
   
        for var in self.variables["05m"]:
            xds_dict["15m"][var] = xds_dict["05m"][var]
        
        xds = xds_dict["15m"]

        # Select the geographic region, vertical levels, 
        # perform destaggering.
        xds = self.select_geographic_extent(xds)
        if self.destagger_kwargs is not None:
            xds = destagger(xds, **self.destagger_kwargs) 
        xds = self.select_levels(xds)  
                
        if self.lead_times is not None:
            # '2012010312_27' -> '12' 
            cycle = dims['init_time'].split('_')[0][-2:]
            xds = self.select_by_lead_times(xds, 
                                      self.lead_times[cycle]["start"],
                                      self.lead_times[cycle]["stop"],
                                      )
            
        # Drop the time coordinate values.
        # To concatenate non-unique forecast valid times,
        # using a DataArray ("valid_time")
        xds = xds.drop_vars("time")
        
        return xds 
        
        
        
        
        
        
        
              
    

