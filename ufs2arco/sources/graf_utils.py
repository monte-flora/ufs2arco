from typing import Tuple

from datetime import datetime
import importlib
import pandas as pd
import numpy as np
import xarray as xr
import json
from scipy.linalg import solve_banded


def raymond_filter_1d(f, eps=0.5):
    """Apply one pass of the Raymond (1988) 2nd-order implicit tangent filter."""
    n = len(f)
    ab = np.zeros((3, n))
    ab[0, 1:] = -eps
    ab[1, :] = 1.0 + 2.0 * eps
    ab[2, :-1] = -eps
    return solve_banded((1, 1), ab, f)


def raymond_filter_2d(field_2d, eps=0.5, order=6):
    """Apply Raymond (1988) implicit tangent low-pass filter in both directions.

    Parameters
    ----------
    field_2d : np.ndarray, shape (ny, nx)
    eps : float
        Filter strength. 0.5 = moderate, 1.0 = aggressive.
    order : int
        Filter order (2, 4, or 6). Achieved by repeated application of 2nd order.
        Higher order = sharper spectral cutoff.
    """
    n_passes = order // 2
    result = field_2d.astype(np.float64)
    for _ in range(n_passes):
        for i in range(result.shape[0]):
            result[i, :] = raymond_filter_1d(result[i, :], eps)
        for j in range(result.shape[1]):
            result[:, j] = raymond_filter_1d(result[:, j], eps)
    return result.astype(np.float32)


def apply_raymond_filter_to_dataset(xds, eps=0.5, order=6, level_dim='level'):
    """Apply Raymond filter to all 3D variables in an xarray Dataset.

    Filters each 2D (y, x) slice at each vertical level independently.
    Skips 2D-only variables (no level dimension) and coordinate variables.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset with dims including 'y', 'x', and optionally level_dim.
    eps : float
        Raymond filter strength.
    order : int
        Filter order (2, 4, or 6).
    level_dim : str
        Name of the vertical level dimension.

    Returns
    -------
    xr.Dataset with filtered data variables.
    """
    # Variables to skip (static, coordinate-like, or 2D)
    skip_vars = {'latitude', 'longitude', 'surface_elevation', 'land_sea_mask'}

    updates = {}
    for var_name in xds.data_vars:
        if var_name in skip_vars:
            continue

        da = xds[var_name]
        dims = da.dims

        if 'y' not in dims or 'x' not in dims:
            continue

        vals = da.values.copy()

        # Squeeze any size-1 time dimension for filtering, then restore
        had_time = 'time' in dims
        if had_time:
            time_axis = dims.index('time')
            vals = vals.squeeze(axis=time_axis)
            squeezed_dims = tuple(d for d in dims if d != 'time')
        else:
            squeezed_dims = dims

        if level_dim in squeezed_dims:
            level_axis = squeezed_dims.index(level_dim)
            n_levels = vals.shape[level_axis]
            for k in range(n_levels):
                if level_axis == 0:
                    slc = vals[k, :, :]
                elif level_axis == 1:
                    slc = vals[:, k, :]
                else:
                    slc = vals[:, :, k]

                if not np.isnan(slc).any():
                    filtered = raymond_filter_2d(slc, eps=eps, order=order)
                    if level_axis == 0:
                        vals[k, :, :] = filtered
                    elif level_axis == 1:
                        vals[:, k, :] = filtered
                    else:
                        vals[:, :, k] = filtered
        else:
            if not np.isnan(vals).any():
                vals = raymond_filter_2d(vals, eps=eps, order=order)

        # Restore time dimension if it was squeezed
        if had_time:
            vals = np.expand_dims(vals, axis=time_axis)

        updates[var_name] = (da.dims, vals)

    return xds.assign({k: v for k, v in updates.items()})


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

    # --- 1. Coefficients ---

    # Rain: higher exponent (b=2.0) makes the Z-M response more nonlinear.
    # Light qr (moderate precip) contributes far less → cuts false alarms and
    # areal over-coverage; heavy qr (intense cores) changes by only ~2-3 dBZ.
    # This better approximates how Morrison 2-moment varies N0r with intensity.
    a_rain = 3.0e9
    b_rain = 2.0

    # Snow: reduced from original 2.02e10; diagnostics showed snow was the
    # dominant bias driver (column-max mean ~8 dBZ, p90 ~33 dBZ).
    # Morrison 2-moment variable N0s gives less Z per unit qs than fixed-N0.
    a_snow = 1.0e10
    b_snow = 2.0

    # Graupel (base): moderate coefficient, steep exponent.
    # b=2.0 concentrates signal in heavy-graupel cores vs. light rimed particles.
    a_graupel = 8.0e9
    b_graupel = 2.0

    # --- 2. Dynamic Dielectric Factor ---

    dielectric_dry = 0.19           # dry ice / frozen particles
    dielectric_wet_snow = 1.0       # melting snow bright band (full liquid coat)
    dielectric_wet_graupel = 0.7    # thin water film on graupel (partial wet coat)

    # A. Logic for SNOW (The Bright Band)
    # Boost limited to the 0-5 °C melting layer; above 5 °C, qs ≈ 0 anyway.
    is_snow_melting = (t >= 273.15) & (t <= 278.15)

    dielectric_factor_snow = xr.where(
        is_snow_melting,
        dielectric_wet_snow,
        dielectric_dry
    )

    # B. Logic for GRAUPEL (wet-coat effect above 0 °C)
    is_graupel_wet = (t >= 273.15)

    dielectric_factor_graupel = xr.where(
        is_graupel_wet,
        dielectric_wet_graupel,
        dielectric_dry
    )

    # --- 3. Compute Z for each species ---

    Z_rain    = a_rain    * (rho * qr)**b_rain
    Z_snow    = (a_snow   * (rho * qs)**b_snow)    * dielectric_factor_snow
    Z_graupel = (a_graupel * (rho * qg)**b_graupel) * dielectric_factor_graupel

    # --- 3b. Hail Proxy (qg-only substitute for missing QHAIL) ---
    # Diagnostics show that large QGRAUPEL (> ~0.5 g/kg) occurs at COLD levels
    # in active updraft cores — NOT in warm air — so the proxy must NOT be
    # restricted to T > 0 °C.  Large rimed particles at any level scatter
    # intensely; this term specifically targets the high-reflectivity cores
    # that would otherwise be captured by a dedicated QHAIL field.
    # The same temperature-dependent dielectric already modulates the wet/dry
    # factor appropriately (dry aloft, partial wet coat near melting layer).
    qg_hail_threshold = 0.5e-3      # 0.5 g/kg  (p99.9 of 3D qg field ≈ 0.2 g/kg;
                                    # 0.5 g/kg selects the most intense cores)
    a_hail_proxy      = 2.0e10      # hard-rimed / proto-hail coefficient
    b_hail_proxy      = 1.75        # standard exponent for dense particles

    Z_hail_proxy = xr.where(
        qg >= qg_hail_threshold,
        # Do NOT multiply by dielectric_factor_graupel here: a_hail_proxy is a
        # hard-rimed-particle coefficient that already embeds the appropriate
        # scattering cross-section for dense ice.  Applying 0.19 (dry-ice
        # dielectric) on top would cut the signal by 5× at cold levels, which
        # is physically incorrect for large, dense graupel.
        a_hail_proxy * (rho * qg)**b_hail_proxy,
        0.0
    )

    # --- 4. Total and Convert ---

    Z_total = Z_rain + Z_snow + Z_graupel + Z_hail_proxy
    
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


def compute_reflectivity_3d(ds, vertical_dim='level', return_as="dataset"):
    """Compute 3D radar reflectivity (dBZ) at each model level.

    Same physics as compute_composite_reflectivity but retains the vertical
    dimension instead of taking the column max. Provides the model with
    vertical structure information: brightband, updraft cores, echo tops.

    Returns a 3D field with the same dimensions as the input 3D variables.
    """
    R_d = 287.05
    min_refl_val = 0.0

    p = ds['pressure']
    t = ds['temperature']
    qr = ds['qr']
    qs = ds['qs']
    qg = ds['qg']

    rho = p / (R_d * t)

    # Coefficients (identical to compute_composite_reflectivity)
    a_rain, b_rain = 3.0e9, 2.0
    a_snow, b_snow = 1.0e10, 2.0
    a_graupel, b_graupel = 8.0e9, 2.0

    dielectric_dry = 0.19
    dielectric_wet_snow = 1.0
    dielectric_wet_graupel = 0.7

    dielectric_factor_snow = xr.where(
        (t >= 273.15) & (t <= 278.15), dielectric_wet_snow, dielectric_dry
    )
    dielectric_factor_graupel = xr.where(
        t >= 273.15, dielectric_wet_graupel, dielectric_dry
    )

    Z_rain = a_rain * (rho * qr)**b_rain
    Z_snow = (a_snow * (rho * qs)**b_snow) * dielectric_factor_snow
    Z_graupel = (a_graupel * (rho * qg)**b_graupel) * dielectric_factor_graupel

    # Hail proxy
    qg_hail_threshold = 0.5e-3
    a_hail_proxy, b_hail_proxy = 2.0e10, 1.75
    Z_hail_proxy = xr.where(
        qg >= qg_hail_threshold,
        a_hail_proxy * (rho * qg)**b_hail_proxy,
        0.0
    )

    Z_total = Z_rain + Z_snow + Z_graupel + Z_hail_proxy
    Z_min_threshold = 0.1
    Z_safe = Z_total.clip(min=Z_min_threshold)

    dbz_3d = xr.where(
        Z_total > Z_min_threshold,
        10.0 * np.log10(Z_safe),
        min_refl_val
    )

    dbz_3d = dbz_3d.clip(min=-10, max=80).astype(np.float32)
    dbz_3d.name = 'refl_3d'
    dbz_3d.attrs = {'units': 'dBZ', 'description': '3D Radar Reflectivity'}

    if return_as == "data_array":
        return dbz_3d

    return ds.assign(refl_3d=dbz_3d)


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


    
