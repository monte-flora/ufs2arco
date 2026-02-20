"""
Regrid from an unstructured MPAS grid (1D ``cell`` dimension) to a regular 2D grid,
then flatten back to ``cell`` for Anemoi compatibility.

Transform flow:
    1. Open target grid NetCDF (subselected HRRR 4km with lat/lon on (y, x))
    2. Separate latitude/longitude from data variables (don't regrid coordinates)
    3. Build source grid descriptor from dataset's lat/lon (1D on cell → points)
    4. Rename cell → points for xESMF locstream compatibility
    5. Create xesmf.Regridder with locstream_in=True, reuse_weights=True
    6. Apply regridding: (time, points, ...) → (time, y, x, ...)
    7. Flatten (y, x) → cell via stack + integer reindex
    8. Replace latitude/longitude with target grid values (flattened)
    9. Store hrrr_grid_shape: [ny, nx] in attrs for 2D reconstruction
"""

import os
import logging

import numpy as np
import xarray as xr
import xesmf

logger = logging.getLogger("ufs2arco")


def unstructured_regrid(
    xds: xr.Dataset,
    weights_path: str,
    target_grid_path: str,
    method: str = "nearest_s2d",
) -> xr.Dataset:
    """
    Regrid an unstructured-grid dataset to a regular 2D grid and flatten back to ``cell``.

    Parameters
    ----------
    xds : xr.Dataset
        Input dataset with dimensions including ``cell`` and variables
        ``latitude`` / ``longitude`` as 1D arrays on ``cell``.
    weights_path : str
        Path to pre-computed xESMF weight file.
    target_grid_path : str
        Path to NetCDF with target grid ``lat`` and ``lon`` on ``(y, x)``.
    method : str
        Regridding method (must match the weights file).

    Returns
    -------
    xr.Dataset
        Regridded dataset with ``cell`` = ny * nx (flattened 2D grid).
    """

    weights_path = os.path.expandvars(weights_path)
    target_grid_path = os.path.expandvars(target_grid_path)

    # ── 1. Open target grid ───────────────────────────────────────────
    grid_tgt = xr.open_dataset(target_grid_path)
    ny = grid_tgt.sizes["y"]
    nx = grid_tgt.sizes["x"]
    logger.info(f"unstructured_regrid: target grid shape ({ny}, {nx})")

    # ── 2. Separate lat/lon and non-spatial variables ──────────────────
    coord_vars = {}
    for name in ("latitude", "longitude"):
        if name in xds.data_vars:
            coord_vars[name] = xds[name]
            xds = xds.drop_vars(name)

    # Separate variables that don't live on cell (e.g. valid_time on time)
    # so they aren't lost during regridding/stacking
    non_spatial_vars = {}
    for name in list(xds.data_vars):
        if "cell" not in xds[name].dims:
            non_spatial_vars[name] = xds[name]
            xds = xds.drop_vars(name)

    # ── 3–4. Build source grid descriptor (cell → points) ────────────
    src_lat = coord_vars["latitude"].values if "latitude" in coord_vars else xds.coords["latitude"].values
    src_lon = coord_vars["longitude"].values if "longitude" in coord_vars else xds.coords["longitude"].values

    grid_src = xr.Dataset({
        "lon": (["points"], src_lon),
        "lat": (["points"], src_lat),
    })

    xds = xds.rename({"cell": "points"})

    # ── 5. Create regridder with pre-computed weights ─────────────────
    regridder = xesmf.Regridder(
        grid_src,
        grid_tgt,
        method=method,
        locstream_in=True,
        reuse_weights=True,
        filename=weights_path,
    )

    # ── 6. Apply regridding: (time, points, ...) → (time, y, x, ...) ─
    xds_regridded = regridder(xds, keep_attrs=True)

    # ── 7. Flatten (y, x) → cell ─────────────────────────────────────
    xds_flat = xds_regridded.stack(cell=("y", "x"))
    # Replace the MultiIndex with a simple integer index
    n_cells = ny * nx
    xds_flat = xds_flat.drop_vars(["y", "x"]).assign_coords(cell=np.arange(n_cells))

    # Restore non-spatial variables
    for name, da in non_spatial_vars.items():
        xds_flat[name] = da

    # ── 8. Replace lat/lon with target grid values (flattened) ────────
    tgt_lat = grid_tgt["lat"].values.ravel()
    tgt_lon = grid_tgt["lon"].values.ravel()
    xds_flat["latitude"] = ("cell", tgt_lat)
    xds_flat["longitude"] = ("cell", tgt_lon)

    # ── 9. Store grid shape in attrs ──────────────────────────────────
    xds_flat.attrs["hrrr_grid_shape"] = [ny, nx]

    logger.info(
        f"unstructured_regrid: regridded {len(src_lat)} source cells → "
        f"{n_cells} target cells ({ny} x {nx})"
    )

    grid_tgt.close()

    return xds_flat
