"""Estimate the on-disk (compressed) size of an ufs2arco-built zarr from
its recipe YAML, before running the build.

Handles two source families:
- `aws_graf_regridded_archive`: full-CONUS / regional builds. Samples are
  enumerated via `_instantiate_first_source` (post-subsample event count
  × frames-per-event from `lead_times` / `file_freqstr`). Cell count comes
  from slicing the static file by `geographic_extent`.
- `aws_graf_regridded_patches`: patch-wise builds. Samples come from the
  manifest JSON (`len(manifest["entries"]) × n_frames_per_sample`). Cells
  come from `(patch_size_km / _PIXEL_KM)²` (where `_PIXEL_KM = 4.0` per
  the source class).

Empirically measured compression ratios from prior GRAF regridded builds:
  1.6x   patches train     (138 ch, 62500 cells, 1M frames)
  1.87x  oklahoma spring   (147 ch, 60516 cells, 45k frames)
  2.1x   conus-fullcase 1c (138 ch, 1.51M cells, 73 frames)
  3.6x   conus-fullcase 5c (138 ch, 1.51M cells, 365 frames)

The default 2.5x is a conservative compromise — override via
``--compression-ratio`` if you have a closer empirical anchor.

Usage:
    python -m ufs2arco.estimate <recipe.yaml> [--compression-ratio N]

Standalone (no dependency on ufs2arco.tranches).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter

import pandas as pd
import yaml
import xarray as xr


# Default compression ratio. See module docstring for empirical anchors.
DEFAULT_COMPRESSION_RATIO = 2.5

# Float32 = 4 bytes per cell.
BYTES_PER_ELEMENT = 4

# Patches source pixel size (km) -- mirrors _PIXEL_KM in
# ufs2arco/sources/aws_graf_regridded_patches.py.
_PATCHES_PIXEL_KM = 4.0


def _is_patches_source(config: dict) -> bool:
    """Return True if the primary source is `aws_graf_regridded_patches`."""
    if "multisource" in config:
        first_name = config["multisource"][0]["source"].get("name", "").lower()
    else:
        first_name = config.get("source", {}).get("name", "").lower()
    return first_name == "aws_graf_regridded_patches"


# --------------------------------------------------------------------- #
# Source enumeration (sample_indices count, post-subsample)
# --------------------------------------------------------------------- #
def _instantiate_first_source(config: dict):
    """Return the first source object — used to enumerate sample_indices.

    Mirrors what the Driver does internally; duplicated here to keep this
    module standalone.
    """
    import ufs2arco.sources

    if "multisource" in config:
        src_cfg = config["multisource"][0]["source"]
    else:
        src_cfg = config["source"]
    name = src_cfg["name"].lower()
    SourceCls = getattr(ufs2arco.sources, ufs2arco.sources._recognized[name])
    kw = {k: v for k, v in src_cfg.items() if k != "name"}
    return SourceCls(**kw)


def _count_frames_per_event(src_cfg: dict) -> int:
    """How many time frames each event contributes, derived from lead_times
    and file_freqstr.

    For lead_times.start='6h', lead_times.stop='7h', file_freqstr='15m' ->
    (60min / 15min) + 1 = 5 frames inclusive.
    """
    lead = src_cfg["lead_times"]
    start_td = pd.Timedelta(lead["start"])
    stop_td = pd.Timedelta(lead["stop"])
    freq_str = src_cfg.get("file_freqstr", "15m")
    freq_td = pd.Timedelta(freq_str.replace("m", "min") if freq_str.endswith("m") and not freq_str.endswith("min") else freq_str)
    n = int((stop_td - start_td) / freq_td) + 1
    return n


# --------------------------------------------------------------------- #
# Channel counting (from YAML — no source instantiation needed)
# --------------------------------------------------------------------- #
def _count_channels(config: dict) -> dict:
    """Sum channels across all sources in the recipe.

    Per-source: (n_3D_vars × n_levels) + n_2D_vars + n_static_vars + n_temporal_aggregated_vars.
    Plus the target's forcings counted once (forcings are written once at
    the top-level target, not per source).

    Returns a dict with breakdown for diagnostic printing.
    """
    sources_list = (
        config["multisource"]
        if "multisource" in config
        else [{"source": config["source"]}]
    )

    breakdown = {"per_source": [], "forcings": 0, "total": 0}

    # Heuristic: variables with no level dimension are 2D. We don't know
    # which variables are 3D vs 2D from the YAML alone — the source class
    # decides. As an approximation we assume the union of `variables`
    # field × `levels` field for 3D vars, plus any without level info as
    # 2D. Easiest approximation: all variables in primary source × levels
    # if levels are listed, otherwise treat as 2D.
    for s in sources_list:
        sc = s["source"]
        vars_ = sc.get("variables") or []
        levels = sc.get("levels") or []
        static_vars = sc.get("static_variables") or []
        # Distinguishing 3D vs 2D in a recipe-only context: 3D vars in
        # the GRAF source set are the prognostic mass-balance fields
        # (uReconstructMeridional, uReconstructZonal, w, theta, pressure,
        # qv, refl_3d, ...). For the estimator, when `levels` is listed,
        # use it for all named variables EXCEPT a hard-coded set of known
        # 2D vars below. Otherwise (no levels) all vars are 2D.
        KNOWN_2D = {
            "t2m", "comp_refl", "u10", "v10", "windgust10m",
            "dewpoint_2m", "precipw", "mslp", "snowh", "skintemp",
            "apcp_bucket", "snow_bucket", "total_cloud_cover",
        }
        vars_3d = [v for v in vars_ if v not in KNOWN_2D]
        vars_2d = [v for v in vars_ if v in KNOWN_2D]
        if levels:
            n_3d = len(vars_3d) * len(levels)
            n_2d = len(vars_2d)
        else:
            # No levels listed -> treat all listed variables as 2D.
            n_3d = 0
            n_2d = len(vars_)
        n_static = len(static_vars)
        sub = n_3d + n_2d + n_static
        breakdown["per_source"].append({
            "file_freqstr": sc.get("file_freqstr", "?"),
            "n_3d_vars": len(vars_3d), "n_levels": len(levels),
            "n_3d_channels": n_3d, "n_2d_vars": n_2d,
            "n_static_vars": n_static, "subtotal": sub,
        })
        breakdown["total"] += sub

    # Forcings count once, written by the first source's target only.
    forcings = config.get("target", {}).get("forcings", []) or []
    breakdown["forcings"] = len(forcings)
    breakdown["total"] += breakdown["forcings"]

    return breakdown


# --------------------------------------------------------------------- #
# Cell counting (slice the static file by geographic_extent)
# --------------------------------------------------------------------- #
def _count_cells(config: dict) -> dict:
    """Count cells inside geographic_extent.

    The regridded LC static file has 2D latitude(y,x) / longitude(y,x)
    coordinate arrays on a structured (y,x) grid (y=1308, x=1524). We
    can't use xarray.sel() with slices on 2D coords; instead, build a
    mask from the bbox and sum.
    """
    import numpy as np
    sources_list = (
        config["multisource"]
        if "multisource" in config
        else [{"source": config["source"]}]
    )
    sc = sources_list[0]["source"]
    static_path = sc["static_regridded_file_path"]
    ext = sc["geographic_extent"]

    ds = xr.open_dataset(static_path)
    # Find latitude/longitude variables (case-insensitive)
    lat_name = next((v for v in ds.variables if v.lower() in ("latitude", "lat")), None)
    lon_name = next((v for v in ds.variables if v.lower() in ("longitude", "lon")), None)
    if lat_name is None or lon_name is None:
        ds.close()
        raise RuntimeError(
            f"Could not find lat/lon variables in {static_path}. "
            f"Variables present: {list(ds.variables)}"
        )
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    full_shape = lat.shape  # (y, x)

    # Normalize bbox longitude to match the static file's convention.
    lon_min_bbox = ext["lon_min"]
    lon_max_bbox = ext["lon_max"]
    if lon.min() < 0 and (lon_min_bbox > 180 or lon_max_bbox > 180):
        # File uses -180..180; bbox uses 0..360. Convert bbox.
        lon_min_bbox = ((lon_min_bbox + 180) % 360) - 180
        lon_max_bbox = ((lon_max_bbox + 180) % 360) - 180
    elif lon.min() >= 0 and (lon_min_bbox < 0 or lon_max_bbox < 0):
        # File uses 0..360; bbox uses -180..180. Convert bbox.
        lon_min_bbox = lon_min_bbox % 360
        lon_max_bbox = lon_max_bbox % 360
    # After conversion, handle dateline wrap (min > max).
    if lon_min_bbox <= lon_max_bbox:
        lon_in_bbox = (lon >= lon_min_bbox) & (lon <= lon_max_bbox)
    else:
        lon_in_bbox = (lon >= lon_min_bbox) | (lon <= lon_max_bbox)
    mask = (lat >= ext["lat_min"]) & (lat <= ext["lat_max"]) & lon_in_bbox
    # The source class drops y/x rows that have NO cells inside the bbox
    # (xarray .where(..., drop=True) on a (y, x) mask). The resulting
    # subset is rectangular — cells = remaining_y × remaining_x — and is
    # ~20-25% larger than mask.sum() because rows that intersect the bbox
    # diagonally keep all their cells. Replicate that here.
    any_in_row = mask.any(axis=1)  # (y,)
    any_in_col = mask.any(axis=0)  # (x,)
    n_cells = int(any_in_row.sum()) * int(any_in_col.sum())
    n_strict_in_bbox = int(mask.sum())
    ds.close()
    return {
        "n_cells": n_cells,
        "n_strict_in_bbox": n_strict_in_bbox,
        "full_grid_shape": full_shape,
        "subset_shape": (int(any_in_row.sum()), int(any_in_col.sum())),
        "static_path": static_path,
    }


# --------------------------------------------------------------------- #
# Patches-source helpers (manifest length, patch_size_km cells)
# --------------------------------------------------------------------- #
def _patches_event_count(config: dict) -> dict:
    """Read the manifest JSON; return entry count + per-month/year breakdown."""
    if "multisource" in config:
        sc = config["multisource"][0]["source"]
    else:
        sc = config["source"]
    manifest_path = sc["manifest_path"]
    import json
    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = manifest.get("entries", [])
    return {
        "manifest_path": manifest_path,
        "n_entries": len(entries),
        "n_frames_per_sample": int(sc.get("n_frames_per_sample", 3)),
        "metadata": manifest.get("metadata", {}),
        "entries": entries,
    }


def _patches_cell_count(config: dict) -> dict:
    """Cells = (patch_size_km / _PIXEL_KM)² for the primary patches source."""
    if "multisource" in config:
        sc = config["multisource"][0]["source"]
    else:
        sc = config["source"]
    patch_km = float(sc.get("patch_size_km", 1000.0))
    pix = int(round(patch_km / _PATCHES_PIXEL_KM))
    return {
        "n_cells": pix * pix,
        "patch_size_km": patch_km,
        "patch_size_pix": pix,
    }


# --------------------------------------------------------------------- #
# Disk free space on the zarr's parent filesystem
# --------------------------------------------------------------------- #
def _free_space(config: dict) -> dict:
    zarr_path = config["directories"]["zarr"]
    parent = os.path.dirname(zarr_path) or "/"
    if not os.path.exists(parent):
        parent = "/"
    usage = shutil.disk_usage(parent)
    return {"path": parent, "total": usage.total, "used": usage.used, "free": usage.free}


# --------------------------------------------------------------------- #
# Main estimator
# --------------------------------------------------------------------- #
def estimate(recipe_path: str, compression_ratio: float = DEFAULT_COMPRESSION_RATIO) -> dict:
    """Estimate dataset size from a recipe yaml. Returns a dict of fields."""
    with open(recipe_path) as f:
        config = yaml.safe_load(f)

    # Count channels BEFORE instantiating the source. The source
    # class's __init__ mutates the config dict's `variables` list (adds
    # diagnostic-computed vars like geopot/rho/theta_m/refl_3d), which
    # would corrupt our user-facing channel count.
    ch = _count_channels(config)

    is_patches = _is_patches_source(config)
    if is_patches:
        # Patches branch: samples from manifest length × n_frames_per_sample,
        # cells from patch_size_km. No source instantiation needed (source
        # class would attempt to load the manifest + apply transforms — wasteful).
        pe = _patches_event_count(config)
        n_events = pe["n_entries"]
        n_frames = pe["n_frames_per_sample"]
        n_samples = n_events * n_frames
        entries = pe["entries"]
        months = [int(e["init_time"][4:6]) for e in entries]
        month_counts = Counter(months)
        cells = _patches_cell_count(config)
        source_info = {
            "kind": "patches",
            "manifest_path": pe["manifest_path"],
            "metadata": pe["metadata"],
        }
    else:
        # Archive (full-CONUS / regional) branch.
        cells = _count_cells(config)
        source = _instantiate_first_source(config)
        n_events = len(source.init_times_df)
        months = source.init_times_df.index.month.tolist()
        month_counts = Counter(months)
        sources_list = (
            config["multisource"]
            if "multisource" in config
            else [{"source": config["source"]}]
        )
        n_frames = _count_frames_per_event(sources_list[0]["source"])
        n_samples = n_events * n_frames
        source_info = {"kind": "archive"}
    free = _free_space(config)

    raw_bytes = n_samples * ch["total"] * cells["n_cells"] * BYTES_PER_ELEMENT
    stored_bytes = raw_bytes / compression_ratio
    margin_bytes = free["free"] - stored_bytes
    verdict = "OK to build" if margin_bytes > 0 else "INSUFFICIENT FREE SPACE"

    return {
        "recipe": recipe_path,
        "source_info": source_info,
        "n_events": n_events,
        "n_frames_per_event": n_frames,
        "n_samples": n_samples,
        "month_counts": dict(sorted(month_counts.items())),
        "channels": ch,
        "cells": cells,
        "raw_bytes": raw_bytes,
        "stored_bytes": stored_bytes,
        "compression_ratio": compression_ratio,
        "free": free,
        "margin_bytes": margin_bytes,
        "verdict": verdict,
    }


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024 or unit == "PB":
            return f"{n:7.2f} {unit}"
        n /= 1024


def _main():
    p = argparse.ArgumentParser(
        prog="python -m ufs2arco.estimate",
        description=__doc__.splitlines()[0],
    )
    p.add_argument("recipe", type=str, help="Path to the ufs2arco recipe YAML.")
    p.add_argument(
        "--compression-ratio",
        type=float,
        default=DEFAULT_COMPRESSION_RATIO,
        help=(
            f"raw_bytes / stored_bytes. Default {DEFAULT_COMPRESSION_RATIO}. "
            f"Empirical anchors: 1.6x (patches train), 1.87x (oklahoma spring), "
            f"2.1x (conus-fullcase 1c), 3.6x (conus-fullcase 5c)."
        ),
    )
    args = p.parse_args()

    r = estimate(args.recipe, compression_ratio=args.compression_ratio)

    print(f"Recipe:                  {r['recipe']}")
    if r["source_info"]["kind"] == "patches":
        print(f"Source kind:             patches  (manifest: {r['source_info']['manifest_path']})")
        print(f"Manifest entries:        {r['n_events']:>10,d}")
        print(f"Frames per entry:        {r['n_frames_per_event']:>10,d}")
        print(f"Total samples (time):    {r['n_samples']:>10,d}")
    else:
        print(f"Source kind:             archive")
        print(f"Events (post-subsample): {r['n_events']:>10,d}")
        print(f"Frames per event:        {r['n_frames_per_event']:>10,d}  "
              f"(lead window @ file_freqstr cadence)")
        print(f"Total samples (time):    {r['n_samples']:>10,d}")
    print(f"Channels:                {r['channels']['total']:>10,d}  "
          f"(per-source: {[s['subtotal'] for s in r['channels']['per_source']]}, "
          f"forcings: {r['channels']['forcings']})")
    if r["source_info"]["kind"] == "patches":
        c = r["cells"]
        print(f"Cells:                   {c['n_cells']:>10,d}  "
              f"({c['patch_size_pix']} x {c['patch_size_pix']} px @ "
              f"{c['patch_size_km']:.0f} km / {_PATCHES_PIXEL_KM} km/px)")
    else:
        c = r["cells"]
        print(f"Cells:                   {c['n_cells']:>10,d}  "
              f"(subset {c['subset_shape']} of full {c['full_grid_shape']}; "
              f"{c['n_strict_in_bbox']:,} strictly inside bbox)")
    print()
    print("Per-month event distribution:")
    for m in sorted(r["month_counts"]):
        bar = "#" * min(r["month_counts"][m], 60)
        print(f"  M{m:02d}: {r['month_counts'][m]:>4d}  {bar}")
    print()
    print(f"Raw size (float32):       {_fmt_bytes(r['raw_bytes'])}")
    print(f"Stored size @ {r['compression_ratio']:>4.1f}x:        {_fmt_bytes(r['stored_bytes'])}")
    print(f"Free on {r['free']['path']:18s}: {_fmt_bytes(r['free']['free'])}")
    print(f"Margin:                   {_fmt_bytes(r['margin_bytes'])}")
    print()
    print(f"VERDICT: {r['verdict']}")
    if r["margin_bytes"] < 0:
        sys.exit(1)


if __name__ == "__main__":
    _main()
