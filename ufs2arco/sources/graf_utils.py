from typing import Tuple

from datetime import datetime 
import pandas as pd 
import numpy as np
import xarray as xr

def _geopotential_height_block(p, Tv, surface_elevation, Rd=287.05, g=9.80665):
    """Pure NumPy routine for one block; Dask will call this per-chunk."""
    nz = p.shape[-1]
    z = np.zeros_like(p, dtype=np.float32)
    z[..., 0] = surface_elevation
    for k in range(nz - 1):
        Tv_bar = 0.5 * (Tv[..., k] + Tv[..., k + 1])
        dz = (Rd * Tv_bar / g) * np.log(p[..., k] / p[..., k + 1])
        z[..., k + 1] = z[..., k] + dz
    return z

def compute_geopotential_height(ds, vertical_dim="level"):
    """
    Dask-safe version of explicit-loop geopotential height computation.
    """
    Rd = 287.05
    g = 9.80665

    ds = ds.assign_coords({vertical_dim: np.arange(ds.sizes[vertical_dim])})
    Tv = ds["temperature"] * (1.0 + 0.61 * ds["qv"])
    p  = ds["pressure"]

    # ensure surface_elevation broadcasts to all dims
    se = ds["surface_elevation"]
    se_expanded = se.broadcast_like(p.isel({vertical_dim: 0}))

    z = xr.apply_ufunc(
        _geopotential_height_block,
        p,
        Tv,
        se_expanded,
        input_core_dims=[[vertical_dim], [vertical_dim], []],
        output_core_dims=[[vertical_dim]],
        dask="parallelized",
        output_dtypes=[np.float32],
        kwargs={"Rd": Rd, "g": g},
    )

    z.name = "geopotential_height"
    z.attrs["units"] = "m"
    
    return ds.assign(geopotential_height=z)

def compute_composite_reflectivity(ds, vertical_dim='nVertLevels'):
    """
    Simplified version using only rain, snow, and graupel.
    This is often sufficient and more robust.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing qr, qs, qg, pressure, temperature
    vertical_dim : str
        Name of the vertical dimension
    
    Returns:
    --------
    ds : xarray.Dataset
        Dataset with added 'composite_reflectivity' variable
    """
    # Gas constant for dry air
    R_d = 287.0
    
    # Calculate air density (kg/m³)
    rho = ds['pressure'] / (R_d * ds['temperature'])
    
    # Reflectivity coefficients (from WSR-88D relationships)
    a_rain = 3.63e9
    b_rain = 1.75
    a_snow = 2.02e10
    b_snow = 2.0
    a_graupel = 4.33e8
    b_graupel = 1.66
    
    # Calculate reflectivity factor Z (mm^6/m^3)
    # Only compute where mixing ratios are significant
    # Minimum mixing ratio threshold (kg/kg)
    # Below this, consider it no precipitation
    min_mixing_ratio = 1e-7  # 0.1 g/kg
    
    # Calculate reflectivity factor Z (mm^6/m^3) for each species
    Z_rain = xr.where(
        ds['qr'] > min_mixing_ratio,
        a_rain * (rho * ds['qr']) ** b_rain,
        0.0
    )
    
    Z_snow = xr.where(
        ds['qs'] > min_mixing_ratio,
        a_snow * (rho * ds['qs']) ** b_snow,
        0.0
    )
    
    Z_graupel = xr.where(
        ds['qg'] > min_mixing_ratio,
        a_graupel * (rho * ds['qg']) ** b_graupel,
        0.0
    )
    
    # Total reflectivity
    Z_total = Z_rain + Z_snow + Z_graupel
    
    # Convert to dBZ
    # HRRR uses -10 dBZ as minimum (no echo value)
    min_Z = 1e-1  # Z = 0.1 corresponds to -10 dBZ
    
    # Use where to avoid log10(0) or log10(negative)
    dbz_3d = xr.where(
        Z_total >= min_Z,
        10.0 * np.log10(Z_total.clip(min=min_Z)),  # Clip as safety
        -10.0  # HRRR's "no echo" value
    )
    
    # Composite (column maximum), ignoring NaN
    composite_dbz = dbz_3d.max(dim=vertical_dim)
    
    # Clip unrealistic values
    composite_dbz = composite_dbz.clip(min=-10, max=80)
    
    # Add to dataset
    ds['composite_reflectivity'] = composite_dbz
    ds['composite_reflectivity'].attrs = {
        'long_name': 'Composite Radar Reflectivity',
        'units': 'dBZ',
        'description': 'Column-maximum radar reflectivity from rain, snow, and graupel',
        'valid_range': '-10 to 80 dBZ',
        'note': 'Values below -10 dBZ indicate no significant precipitation',
    }
    
    return ds


''' Version used for v1.02/v1.03
def compute_composite_reflectivity(ds, vertical_dim='nVertLevels'):
    """
    Simplified radar reflectivity computation using rain, snow, and graupel.
    This is often sufficient and more robust than including all hydrometeor types.
    
    Method
    ------
    Calculates radar reflectivity using the Rayleigh scattering approximation
    with power-law relationships between hydrometeor mixing ratios and 
    reflectivity factor:
    
        Z = a × (ρ × q)^b
    
    where Z is reflectivity factor (mm⁶/m³), ρ is air density (kg/m³), 
    q is mixing ratio (kg/kg), and a, b are empirical coefficients for each
    hydrometeor type.
    
    References
    ----------
    - Smith, P. L., Myers, C. G., & Orville, H. D. (1975). Radar Reflectivity 
      Factor Calculations in Numerical Cloud Models Using Bulk Parameterization 
      of Precipitation. Journal of Applied Meteorology, 14(6), 1156-1165.
      DOI: 10.1175/1520-0450(1975)014<1156:RRFCIN>2.0.CO;2
    
    - Coefficients derived from WRF model post-processing implementations
      and mesoscale modeling diagnostic packages (e.g., Stoelinga 2005,
      WRF Post-Processing System).
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing qr, qs, qg (mixing ratios in kg/kg), 
        pressure (Pa), and temperature (K)
    vertical_dim : str, optional
        Name of the vertical dimension (default: 'nVertLevels')
    
    Returns
    -------
    ds : xarray.Dataset
        Dataset with added 'composite_reflectivity' variable in dBZ
    
    Notes
    -----
    - Uses simplified three-hydrometeor approach (rain, snow, graupel)
    - Applies column maximum to produce composite reflectivity
    - Minimum reflectivity threshold of 1e-3 mm⁶/m³ applied before dBZ conversion
    """
    # Gas constant for dry air
    R_d = 287.0
    
    # Calculate air density
    rho = ds['pressure'] / (R_d * ds['temperature'])
    
    # Reflectivity coefficients
    a_rain = 3.63e9
    b_rain = 1.75
    a_snow = 2.02e10
    b_snow = 2.0
    a_graupel = 4.33e8
    b_graupel = 1.66
    
    # Calculate contributions
    Z_total = (a_rain * (rho * ds['qr']) ** b_rain + 
               a_snow * (rho * ds['qs']) ** b_snow + 
               a_graupel * (rho * ds['qg']) ** b_graupel)
    
    # Avoid log(0)
    Z_total = Z_total.where(Z_total >= 1e-3, 1e-3)
    
    # Convert to dBZ
    dbz_3d = 10 * np.log10(Z_total)
    
    # Composite (column maximum)
    composite_dbz = dbz_3d.max(dim=vertical_dim)
    
    # Add to dataset
    ds['composite_reflectivity'] = composite_dbz
    ds['composite_reflectivity'].attrs = {
        'long_name': 'Composite Radar Reflectivity',
        'units': 'dBZ',
        'description': 'Column-maximum radar reflectivity from rain, snow, and graupel',
    }
    
    return ds

'''

def parse_order_file(order_filename : str):
    """
    ***Temporary***
    Load the time permutation .txt file for 
    correct time ordering of existing GRAF reforecast on S3
    """
    with open(order_filename) as f:
        stamps = [datetime.strptime(line.strip(), "%Y-%m-%d_%H.%M.%S") for line in f]
    stamps = pd.to_datetime(stamps)
    idx = np.argsort(stamps)
    stamps_ordered = stamps[idx]

    return stamps_ordered, idx.tolist()  


def get_expected_times(xds: xr.Dataset, time_resolution: str, n_steps: int) -> pd.DatetimeIndex:
    """Compute expected timeline from dataset config_start_time, resolution, and num of time steps"""
    start_timestamp = xds.attrs['config_start_time']
    start_time = pd.to_datetime(start_timestamp, format='%Y-%m-%d_%H:%M:%S')
    return pd.date_range(start=start_time, periods=n_steps, freq=time_resolution)

def add_missing_times_with_nans(
    xds: xr.Dataset, expected_times: pd.DatetimeIndex
    ) -> xr.Dataset:
    """
    Reindex dataset to expected times, inserting NaNs for missing slots.
    If the existing times already match, skip reindexing.
    """
    actual_times = xds.indexes["time"]

    if actual_times.equals(expected_times):
        #print("No time reindexing need!")
        return xds

    return xds.reindex(time=expected_times)

def times_with_nans(actual_times: pd.DatetimeIndex, expected_times: pd.DatetimeIndex)->pd.DatetimeIndex:
    missing = expected_times.difference(actual_times)
    return missing

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


def subsample_by_month(
    df: pd.DataFrame, 
    frac: float = 0.5, 
    seed: int = 42,
    months: list = None
) -> pd.DataFrame:
    """Randomly keep a fraction of samples per month, optionally filtering to specific months.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    frac : float
        Fraction of samples to keep per month (0 to 1). If 1.0, returns original.
    seed : int
        Random seed for reproducibility
    months : list, optional
        List of month numbers to keep (1-12). If None, keeps all months.
        Example: [4, 5, 6, 7] for April-July
    
    Returns
    -------
    pd.DataFrame
        Subsampled dataframe
        
    Examples
    --------
    >>> # Keep only April-July data
    >>> df_sub = subsample_by_month(df, frac=1.0, months=[4, 5, 6, 7])
    
    >>> # Keep 50% of April-July data
    >>> df_sub = subsample_by_month(df, frac=0.5, months=[4, 5, 6, 7])
    """
    df = df.copy()
    df["month"] = df.index.month
    
    # Filter by months if specified
    if months is not None:
        df = df[df["month"].isin(months)]
        if df.empty:
            raise ValueError(f"No data found for months {months}")
    
    # If frac=1 and no further subsampling needed, return early
    if frac >= 1.0:
        return df.drop(columns=["month"])
    
    # Subsample within each month
    groups = df.groupby("month", group_keys=False)
    df_sub = pd.concat(
        [g.sample(frac=frac, random_state=seed) for _, g in groups],
        axis=0
    ).sort_index()
    
    return df_sub.drop(columns=["month"])
    