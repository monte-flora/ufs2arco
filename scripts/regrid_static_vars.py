#!/usr/bin/env python
"""
One-time script to regrid MPAS static variables to the structured 2D grid
used by the regridded GRAF reforecast data.

Uses the same nearest-neighbor (NN) regridding weights (ESMF sparse matrix
format: col/row/S with dimension n_s) that were used to regrid the
atmospheric fields.

Usage:
    python regrid_static_vars.py \
        --static-file /home/mflora/graf-ai/grafai/data/rpm4km.static.nc \
        --weight-file /home/mflora/graf-reforecast-conus-interp/data/graf_to_grafconus_4km_weights.nc \
        --reference-zarr /grafrr/2004010112_27/mpasout_15m.zarr \
        --output /home/mflora/graf-ai/grafai/data/graf_regridded_static.nc

Author: monte-flora
"""
import argparse
import logging

import numpy as np
import xarray as xr
import netCDF4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Variables to regrid from the MPAS static file
STATIC_VARS = ["ter", "landmask", "soiltemp", "var2d", "con", "oa1", "oa2", "oa3", "oa4"]


def load_nn_weights(weight_file: str, dst_shape: tuple[int, int]):
    """Load nearest-neighbor regridding weights from ESMF sparse matrix format.

    The weight file has dimension n_s = ny*nx (one entry per destination point)
    with variables:
      - col: 1-based source cell index
      - row: 1-based destination flat index
      - S: weight (1.0 for NN)

    Args:
        weight_file: path to the ESMF weight netCDF
        dst_shape: (ny, nx) from the reference zarr

    Returns:
        src_indices: 1D array of length ny*nx, where src_indices[dst_flat] = source cell index
    """
    with netCDF4.Dataset(weight_file, "r") as wds:
        col = np.array(wds.variables["col"][:]) - 1  # 1-based → 0-based source index
        row = np.array(wds.variables["row"][:]) - 1  # 1-based → 0-based destination flat index

    n_dst = dst_shape[0] * dst_shape[1]
    assert len(col) == n_dst, (
        f"Weight file n_s={len(col)} != destination grid size {n_dst} "
        f"(expected {dst_shape[0]}×{dst_shape[1]})"
    )

    # Build the mapping: for each destination flat index, store the source cell index
    # row should already be 0..n_dst-1 for a complete NN mapping
    src_indices = np.empty(n_dst, dtype=np.int64)
    src_indices[row] = col

    return src_indices


def regrid_field(field_1d: np.ndarray, src_indices: np.ndarray, dst_shape: tuple) -> np.ndarray:
    """Apply NN regridding: index source field by src_indices and reshape to 2D."""
    return field_1d[src_indices].reshape(dst_shape)


def main():
    parser = argparse.ArgumentParser(description="Regrid MPAS static vars to structured 2D grid")
    parser.add_argument("--static-file", required=True, help="Path to MPAS static file (rpm4km.static.nc)")
    parser.add_argument("--weight-file", required=True, help="Path to ESMF NN regridding weight file")
    parser.add_argument("--reference-zarr", required=True, help="Path to a reference regridded zarr (for grid shape)")
    parser.add_argument("--output", required=True, help="Output netCDF path")
    args = parser.parse_args()

    # Get destination grid shape from reference zarr
    logger.info("Loading reference zarr for grid shape: %s", args.reference_zarr)
    ref_ds = xr.open_zarr(args.reference_zarr, consolidated=True)
    ny, nx = ref_ds.sizes["y"], ref_ds.sizes["x"]
    ref_ds.close()
    dst_shape = (ny, nx)
    logger.info("Destination grid shape: (y=%d, x=%d), total=%d", ny, nx, ny * nx)

    logger.info("Loading NN weight file: %s", args.weight_file)
    src_indices = load_nn_weights(args.weight_file, dst_shape)

    logger.info("Loading static file: %s", args.static_file)
    regridded = {}
    with netCDF4.Dataset(args.static_file, "r") as sds:
        for var in STATIC_VARS:
            if var not in sds.variables:
                logger.warning("Variable '%s' not found in static file, skipping", var)
                continue
            raw = np.array(sds.variables[var][:]).squeeze()
            if raw.ndim != 1:
                logger.warning("Variable '%s' has shape %s (expected 1D), skipping", var, raw.shape)
                continue
            regridded[var] = regrid_field(raw, src_indices, dst_shape)
            logger.info("  %s: min=%.4f, max=%.4f, shape=%s",
                        var, regridded[var].min(), regridded[var].max(), regridded[var].shape)

    # Build output dataset
    ds_out = xr.Dataset(
        {var: (["y", "x"], data) for var, data in regridded.items()},
        coords={
            "y": np.arange(dst_shape[0]),
            "x": np.arange(dst_shape[1]),
        },
    )

    logger.info("Writing output to: %s", args.output)
    ds_out.to_netcdf(args.output)
    logger.info("Done. Output shape: (y=%d, x=%d)", *dst_shape)

    # Spot-check: landmask should be binary
    if "landmask" in regridded:
        unique_vals = np.unique(regridded["landmask"])
        logger.info("Landmask unique values: %s", unique_vals)
        if not np.all(np.isin(unique_vals, [0, 1])):
            logger.warning("Landmask contains non-binary values! NN regridding should preserve binary values.")

    # Spot-check: terrain should be non-negative
    if "ter" in regridded:
        neg_count = np.sum(regridded["ter"] < 0)
        if neg_count > 0:
            logger.warning("Terrain has %d negative values (min=%.2f)", neg_count, regridded["ter"].min())
        else:
            logger.info("Terrain all non-negative (min=%.2f, max=%.2f)",
                        regridded["ter"].min(), regridded["ter"].max())


if __name__ == "__main__":
    main()
