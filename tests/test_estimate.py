"""Unit tests for the size-estimator helpers (ufs2arco.estimate).

Standalone — no source instantiation, no MPI. The end-to-end estimator
output is validated separately by running it against the known-completed
conus-fullcase-5cases.zarr build (predicted size within ±30% of actual).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ufs2arco import estimate as est


# ----------------------------------------------------------------------
# _count_frames_per_event
# ----------------------------------------------------------------------
def test_frames_5_at_15m():
    """lead 6h -> 7h at 15-min cadence = 5 frames inclusive."""
    sc = {"lead_times": {"start": "6h", "stop": "7h"}, "file_freqstr": "15m"}
    assert est._count_frames_per_event(sc) == 5


def test_frames_73_at_15m():
    """lead 6h -> 24h at 15-min = (18*4)+1 = 73 frames."""
    sc = {"lead_times": {"start": "6h", "stop": "24h"}, "file_freqstr": "15m"}
    assert est._count_frames_per_event(sc) == 73


def test_frames_at_5m_cadence():
    """lead 6h -> 6h5m at 5-min cadence = 2 frames (6:00 and 6:05)."""
    sc = {"lead_times": {"start": "6h", "stop": "6h5m"}, "file_freqstr": "5m"}
    assert est._count_frames_per_event(sc) == 2


# ----------------------------------------------------------------------
# _count_channels
# ----------------------------------------------------------------------
def _multisource_recipe(vars_3d, vars_2d, levels, static_vars,
                       secondary_vars=(), forcings=()):
    """Build a minimal multisource recipe dict for channel-count testing."""
    src1 = {"source": {
        "name": "aws_graf_regridded_archive",
        "variables": list(vars_3d) + list(vars_2d),
        "levels": list(levels),
        "static_variables": list(static_vars),
    }}
    sources = [src1]
    if secondary_vars:
        sources.append({"source": {
            "name": "aws_graf_regridded_archive",
            "variables": list(secondary_vars),
        }})
    return {
        "multisource": sources,
        "target": {"forcings": list(forcings)},
    }


def test_channels_basic():
    """6 3D vars × 19 levels + 10 2D + 2 statics + 9 forcings = 138."""
    cfg = _multisource_recipe(
        vars_3d=["uReconstructMeridional", "uReconstructZonal", "w",
                 "theta", "pressure", "qv"],
        vars_2d=["t2m", "comp_refl", "u10", "v10", "windgust10m",
                 "dewpoint_2m", "precipw", "mslp", "snowh", "skintemp"],
        levels=list(range(19)),
        static_vars=["ter", "landmask"],
        secondary_vars=["apcp_bucket", "snow_bucket", "total_cloud_cover"],
        forcings=["cos_latitude"] * 9,
    )
    ch = est._count_channels(cfg)
    assert ch["per_source"][0]["subtotal"] == 6 * 19 + 10 + 2  # = 126
    assert ch["per_source"][1]["subtotal"] == 3  # 3 secondary 2D vars
    assert ch["forcings"] == 9
    assert ch["total"] == 138


def test_channels_no_levels_treats_all_as_2d():
    cfg = _multisource_recipe(
        vars_3d=[],
        vars_2d=["apcp_bucket", "snow_bucket"],
        levels=[],
        static_vars=[],
        forcings=[],
    )
    ch = est._count_channels(cfg)
    # No levels listed → all vars counted as 2D, no double-count
    assert ch["per_source"][0]["subtotal"] == 2


def test_channels_single_source_recipe():
    """Single-source recipe (no `multisource` key) — uses `source:` directly."""
    cfg = {
        "source": {
            "name": "aws_graf_regridded_archive",
            "variables": ["theta", "qv", "t2m"],
            "levels": [0, 1, 2],
            "static_variables": ["ter"],
        },
        "target": {"forcings": []},
    }
    ch = est._count_channels(cfg)
    # 2 3D × 3 levels + 1 2D + 1 static = 8
    assert ch["per_source"][0]["subtotal"] == 8


# ----------------------------------------------------------------------
# _fmt_bytes — formatting helper
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n_bytes, contains", [
    (512, "512.00 B"),
    (2048, "2.00 KB"),
    (5 * 1024**3, "5.00 GB"),
    (3 * 1024**4, "3.00 TB"),
])
def test_fmt_bytes(n_bytes, contains):
    assert contains in est._fmt_bytes(n_bytes)


# ----------------------------------------------------------------------
# _count_cells — longitude normalization across both conventions
# ----------------------------------------------------------------------
def _make_static_nc(path, lat_range=(7.2, 63.3), lon_range=(-141.5, -39.9),
                    shape=(50, 60)):
    """Write a tiny synthetic static netCDF with 2D lat/lon coords."""
    y_lin = np.linspace(lat_range[0], lat_range[1], shape[0])
    x_lin = np.linspace(lon_range[0], lon_range[1], shape[1])
    lat = np.broadcast_to(y_lin[:, None], shape).copy()
    lon = np.broadcast_to(x_lin[None, :], shape).copy()
    ds = xr.Dataset(
        {"ter": (("y", "x"), np.zeros(shape, dtype=np.float32))},
        coords={"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
    )
    ds.to_netcdf(path)


def test_cells_lon_convention_negative_file_positive_bbox(tmp_path):
    """Static file in -180..180 convention, bbox in 0..360 convention.
    Estimator must normalize and find cells correctly."""
    static_path = tmp_path / "static.nc"
    _make_static_nc(str(static_path))
    cfg = {
        "multisource": [{"source": {
            "static_regridded_file_path": str(static_path),
            "geographic_extent": {
                "lat_min": 21.0, "lat_max": 53.0,
                "lon_min": 226.0, "lon_max": 300.0,  # 0..360 convention
            },
        }}],
    }
    cells = est._count_cells(cfg)
    # Should find cells in the conus-equivalent region. With lon -180..180,
    # 226..300 is normalized to -134..-60. The synthetic grid spans
    # -141..-40, so most cells fall inside.
    assert cells["n_cells"] > 100
    assert cells["n_strict_in_bbox"] > 0
    assert cells["n_cells"] >= cells["n_strict_in_bbox"]  # subset ≥ strict


def test_is_patches_source_detection():
    """`_is_patches_source` must recognize the patches source name."""
    patches = {"multisource": [{"source": {"name": "aws_graf_regridded_patches"}}]}
    archive = {"multisource": [{"source": {"name": "aws_graf_regridded_archive"}}]}
    single_patches = {"source": {"name": "aws_graf_regridded_patches"}}
    assert est._is_patches_source(patches)
    assert not est._is_patches_source(archive)
    assert est._is_patches_source(single_patches)


def test_patches_cells_from_patch_size_km():
    """1500 km @ _PIXEL_KM=4.0 -> 375x375 = 140,625 cells; 1000 km -> 62,500."""
    cfg_1500 = {"multisource": [{"source": {
        "name": "aws_graf_regridded_patches", "patch_size_km": 1500,
    }}]}
    assert est._patches_cell_count(cfg_1500)["n_cells"] == 375 * 375
    cfg_1000 = {"multisource": [{"source": {
        "name": "aws_graf_regridded_patches", "patch_size_km": 1000,
    }}]}
    assert est._patches_cell_count(cfg_1000)["n_cells"] == 250 * 250


def test_patches_event_count_reads_manifest(tmp_path):
    """`_patches_event_count` reads the manifest JSON and reports entry count."""
    import json
    manifest_path = tmp_path / "manifest.json"
    entries = [{"init_time": f"2004{m:02d}0112_27", "intensity": 5.0,
                "source": "storm", "start_offset_min": 360, "patch_idx": 0,
                "lat_c": 40.0, "lon_c": -100.0, "trajectory_id": f"tj{i}"}
               for i, m in enumerate([1,2,3])]
    with open(manifest_path, "w") as f:
        json.dump({"metadata": {"foo": "bar"}, "entries": entries}, f)

    cfg = {"multisource": [{"source": {
        "name": "aws_graf_regridded_patches",
        "manifest_path": str(manifest_path),
        "n_frames_per_sample": 6,
    }}]}
    info = est._patches_event_count(cfg)
    assert info["n_entries"] == 3
    assert info["n_frames_per_sample"] == 6
    assert info["metadata"]["foo"] == "bar"


def test_cells_dateline_wrap_handled(tmp_path):
    """bbox wraps the dateline (lon_min > lon_max). Mask should OR not AND."""
    static_path = tmp_path / "static.nc"
    # 360-convention file, spanning Pacific (e.g., 150..250)
    _make_static_nc(str(static_path), lon_range=(150.0, 250.0))
    cfg = {
        "multisource": [{"source": {
            "static_regridded_file_path": str(static_path),
            "geographic_extent": {
                "lat_min": 7.0, "lat_max": 64.0,
                "lon_min": 240.0, "lon_max": 180.0,  # wraps via 0/360
            },
        }}],
    }
    cells = est._count_cells(cfg)
    assert cells["n_strict_in_bbox"] > 0
