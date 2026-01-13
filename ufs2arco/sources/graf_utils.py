from typing import Tuple

from datetime import datetime 
import importlib
import pandas as pd 
import numpy as np
import xarray as xr
import json 

def compute_virtual_pot_temp(ds, return_as="dataset"):
    theta = ds['theta']
    qv = ds['qv']
    
    theta_m = theta * (1.0 + 0.608 * qv)
    
    # 5. Metadata
    theta_m.name = "theta_m"
    theta_m.attrs["units"] = "K"
    theta_m.attrs["long_name"] = "Virtual Potential Temperature"
    
    if return_as == "data_array":
        return theta_m
    
    return ds.assign(theta_m=theta_m.astype(np.float32))


def compute_density(ds, return_as="dataset"):
    """
    Dask-compatible (lazy) moist air density computation.
    
    Physics:
    Uses the Ideal Gas Law adjusted for moisture (Virtual Temperature).
    rho = P / (Rd * Tv)
    """
    # 1. Constants
    Rd = 287.05
    
    # 2. Lazy Variable Extraction (Force float64 for calculation precision)
    # We assume standard variable names often found in WRF/MPAS outputs
    p = ds["pressure"].astype(np.float64)
    t = ds["temperature"].astype(np.float64)
    qv = ds["qv"].astype(np.float64)

    # 3. Virtual Temperature (Lazy Element-wise)
    # Moist air is lighter than dry air. Tv accounts for this buoyancy effect.
    # Tv = T * (1 + 0.608 * qv)
    tv = t * (1.0 + 0.608 * qv)

    # 4. Density Calculation (Equation of State)
    # rho = P / (Rd * Tv)
    rho_da = p / (Rd * tv)

    # 5. Metadata
    rho_da.name = "density"
    rho_da.attrs["units"] = "kg m-3"
    rho_da.attrs["long_name"] = "Moist Air Density"
    
    if return_as == "data_array":
        return rho_da
    
    return ds.assign(rho=rho_da.astype(np.float32))
    

def compute_geopotential(ds, z_dim="level", surface_elev_var="surface_elevation", return_as="dataset"):
    """
    Dask-compatible (lazy) geopotential computation.
    Uses vectorized shift/cumsum instead of loops to preserve the Dask graph.
    """
    # 1. Constants
    Rd = 287.05
    g = 9.80665
    
    # 2. Lazy Variable Extraction (Preserve float32 when possible)
    # dask='allowed' ensures we don't trigger compute on load
    base_dtype = np.float64 if ds["pressure"].dtype == np.float64 else np.float32
    p = ds["pressure"].astype(base_dtype)
    t = ds["temperature"].astype(base_dtype)
    qv = ds["qv"].astype(base_dtype)
    mslp = ds["mslp"].astype(base_dtype)
    t2m = ds["t2m"].astype(base_dtype)
    
    if surface_elev_var in ds:
        z_sfc = ds[surface_elev_var].astype(base_dtype)
    else:
        z_sfc = ds.coords[surface_elev_var].astype(base_dtype)

    # 3. Virtual Temperature (Lazy Element-wise)
    tv = t * (1.0 + 0.608 * qv)

    # 4. The "Elevator" Calculation (Lazy Element-wise)
    # Estimate Psfc to get the jump from terrain to the first model level
    t_mean_sfc = t2m + (0.0065 * z_sfc / 2.0) 
    psfc = mslp / np.exp((g * z_sfc) / (Rd * t_mean_sfc))

    # Calculate the thickness of the "ghost layer" between Surface and Level 0
    # We select Level 0 lazily
    p0 = p.isel({z_dim: 0})
    tv0 = tv.isel({z_dim: 0})
    d_z0 = (Rd * tv0 / g) * np.log(psfc / p0)
    
    # This is the height of the first model level
    z_base = z_sfc + d_z0

    # ---------------------------------------------------------
    # 5. Vectorized Integration (The "Loop" Replacement)
    # ---------------------------------------------------------
    
    # We calculate the thickness between level k and k+1 for ALL k at once.
    # We use .shift() to align level k with k+1.
    
    # Shift UP to get the "next" level (k+1) aligned with "current" level (k)
    # Note: The last element becomes NaN, which is fine (no layer above top)
    tv_next = tv.shift({z_dim: -1})
    p_next = p.shift({z_dim: -1})
    
    # Average Tv between k and k+1
    tv_bar = 0.5 * (tv + tv_next)
    
    # Log pressure thickness
    # Result is an array where index 'k' holds the thickness from k to k+1
    dlogp = np.log(p / p_next)
    
    # Calculate Hypsometric Thickness for every layer
    layer_thickness = (Rd * tv_bar / g) * dlogp

    # ---------------------------------------------------------
    # 6. Accumulation (cumsum)
    # ---------------------------------------------------------
    
    # We now have an array of thicknesses.
    # We need to sum them up. 
    # But layer_thickness[0] is the distance from Lev0 to Lev1.
    # That distance should be added to Lev1, not Lev0.
    
    # We shift DOWN by 1.
    # Index 0 becomes NaN (we fill with 0, because Lev0 has 0 accumulation from itself)
    # Index 1 receives the thickness from Lev0->Lev1.
    thickness_aligned = layer_thickness.shift({z_dim: 1}).fillna(0.0)
    
    # Cumulative Sum along the vertical dimension
    # This is efficient in Dask
    z_accumulation = thickness_aligned.cumsum(dim=z_dim)
    
    # 7. Add the Base Height
    # Broadcasting z_base (2D) across z_accumulation (3D) is handled by xarray
    z_final = z_base + z_accumulation

    z_final *= 9.8 #Convert from m to gpm
    
    # 8. Return
    z_final.name = "geopot"
    z_final.attrs["units"] = "m2 s-2"
    
    # Cast back to float32 only at the very end to save memory on write
    z_final= z_final.astype(np.float32)
    
    
    if return_as == "data_array":
        return z_final
    
    return ds.assign(geopot=z_final)

def compute_composite_reflectivity(ds, vertical_dim='level', return_as="dataset"):
    """
    Compute composite (column-maximum) radar reflectivity with convective hail enhancement.
    
    Calculates reflectivity from mixing ratios (rain, snow, graupel) using Z-M relationships
    tuned for severe convection. Includes temperature-dependent dielectric adjustments to 
    simulate radar bright band (melting snow) and wet hail enhancement effects.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing:
        - 'pressure' : Air pressure (Pa)
        - 'temperature' : Air temperature (K)
        - 'qr' : Rain mixing ratio (kg/kg)
        - 'qs' : Snow mixing ratio (kg/kg)
        - 'qg' : Graupel/hail mixing ratio (kg/kg)
    vertical_dim : str, default='level'
        Name of the vertical coordinate dimension to maximize over
    return_as : str, default='dataset'
        Return format (currently unused, returns DataArray)
    
    Returns
    -------
    xarray.DataArray
        Composite reflectivity in dBZ, clipped to [-10, 80] dBZ range.
        Shape matches input with `vertical_dim` removed.
    
    Notes
    -----
    Key enhancements over simple Z-M relationships:
    
    1. **Graupel Coefficient Boost** (a_graupel = 2.5e10):
       Increased from soft-graupel value to represent hard hail, adding ~8-10 dBZ
       to convective cores.
    
    2. **Wet Hail Enhancement**:
       Graupel above 0°C receives full dielectric factor (|K|² = 1.0) to simulate
       wet hail's increased radar cross-section, regardless of temperature.
    
    3. **Bright Band Simulation**:
       Snow between 0-5°C (273-278 K) receives enhanced dielectric factor to 
       represent the melting layer radar signature.
    
    Assumes ideal gas law for air density: ρ = p/(R_d·T)
    
    Examples
    --------
    >>> comp_refl = compute_composite_reflectivity(model_ds)
    >>> comp_refl.plot()
    """
    min_comp_refl_val = 0.0
    R_d = 287.05
    
    # Lazy Load
    p = ds['pressure']
    t = ds['temperature']
    qr = ds['qr']
    qs = ds['qs']
    qg = ds['qg']
    
    # Density
    rho = p / (R_d * t)

    # --- 1. Coefficients (Tuned for Convection) ---
    
    # Rain: Boosted slightly for heavy convection
    # Old: 3.63e9. New: 4.0e9 (Small bump)
    a_rain = 4.0e9 
    b_rain = 1.75
    
    # Snow: Standard Bright Band logic is fine here
    a_snow = 2.02e10
    b_snow = 2.0
    
    # Graupel/Hail: SIGNIFICANT BOOST
    # Old: 3.8e9 (Soft Graupel) -> New: 2.5e10 (Hard Hail)
    # This change alone adds ~8-10 dBZ to graupel cores
    a_graupel = 2.5e10 
    b_graupel = 1.75 
    
    # --- 2. Dynamic Dielectric Factor ---
    
    dielectric_dry = 0.19 # Frozen
    dielectric_wet = 1.0  # Liquid/Melting
    
    # A. Logic for SNOW (The Bright Band)
    # Snow melts quickly. We only want the boost in the transition zone (0C to 5C).
    # Above 5C, snow converts to rain (qr), so qs drops to near zero anyway.
    is_snow_melting = (t >= 273.15) & (t <= 278.15)
    
    dielectric_factor_snow = xr.where(
        is_snow_melting,
        dielectric_wet,
        dielectric_dry
    )

    # B. Logic for GRAUPEL/HAIL (The "Wet Hail" Effect)
    # Hail survives into warm air. If T > 0C, the surface is wet.
    # We DO NOT restrict this to < 5C. If it's 30C, hail is definitely wet!
    is_graupel_wet = (t >= 273.15)
    
    dielectric_factor_graupel = xr.where(
        is_graupel_wet,
        dielectric_wet, # 1.0 (Wet surface scatters like crazy)
        dielectric_dry  # 0.19 (Dry ice aloft)
    )

    # --- 3. Compute Z for each species ---
    
    Z_rain = a_rain * (rho * qr)**b_rain
    
    # Snow uses the band logic
    Z_snow = (a_snow * (rho * qs)**b_snow) * dielectric_factor_snow
    
    # Graupel uses the "Always Wet if Warm" logic
    Z_graupel = (a_graupel * (rho * qg)**b_graupel) * dielectric_factor_graupel

    # --- 4. Total and Convert ---
    
    Z_total = Z_rain + Z_snow + Z_graupel
    
    Z_min_threshold = 0.1 
    Z_safe = Z_total.clip(min=Z_min_threshold)
    
    dbz_3d = xr.where(
        Z_total > Z_min_threshold,
        10.0 * np.log10(Z_safe),
        min_comp_refl_val
    )
    
    composite_dbz = dbz_3d.max(dim=vertical_dim)
    
    # Clip max to reasonable hail limit (e.g., 75-80 dBZ)
    composite_dbz = composite_dbz.clip(min=-10, max=80)
    
    composite_dbz.name = 'comp_refl'
    composite_dbz.attrs = {
        'units': 'dBZ', 
        'description': 'Composite Reflectivity with Hail Enhancement'
    }

    composite_dbz = composite_dbz.astype(np.float32)
    
    if return_as == "data_array":
        return composite_dbz 
    
    return ds.assign(comp_refl=composite_dbz)
        

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

    return stamps_ordered, idx.tolist(), stamps


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

def save_times_dict_json(d: dict, path: str) -> None:
    serializable = {k: [t.isoformat() for t in v.to_pydatetime()] for k, v in d.items()}
    with open(path, "w") as f:
        json.dump(serializable, f)

def load_times_dict_json(path: str) -> dict[str, pd.DatetimeIndex]:
    with open(path, "r") as f:
        raw = json.load(f)
    return {k: pd.DatetimeIndex(pd.to_datetime(v)) for k, v in raw.items()}


def save_missing_times_parquet(d: dict, path: str) -> None:
    rows = []
    for key, times in d.items():
        for t in times:
            rows.append({"key": key, "time": pd.Timestamp(t)})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])

    df.to_parquet(path, index=False)

def load_missing_times_parquet(path: str) -> dict[str, set[pd.Timestamp]]:
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        return {}

    out = {}
    for key, g in df.groupby("key"):
        out[key] = set(pd.to_datetime(g["time"]))

    return out

def save_missing_indices_json(missing_indices_per_init, path):
    serializable = {
        k: sorted(map(int, v))
        for k, v in missing_indices_per_init.items()
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_missing_indices_json(path):
    with open(path) as f:
        data = json.load(f)
    return {
        k: set(v)
        for k, v in data.items()
    }


    
