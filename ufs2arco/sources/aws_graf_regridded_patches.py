"""Patch-based GRAF regridded archive source.

Reads the same S3 zarrs as :class:`AWSGRAFRegriddedArchive` but produces
many short (3-frame) trajectories per case instead of one long per-init
trajectory. Each trajectory is a 1000×1000 km spatial patch sampled from
a manifest that was built ahead of time by
``grafai/datasets/patch_sampler/build_patch_manifest.py``.

Key differences from the parent class
-------------------------------------
* ``sample_dims = ("init_time", "forecast_step")`` unchanged — the DataMover
  iteration stays identical — but ``forecast_step`` now indexes the *flat
  enumeration* of (patch_idx, frame_offset) tuples within a case rather
  than lead-time offsets. A fixed number ``n_patches_per_init ×
  n_frames_per_sample`` steps per init.

* Each forecast_step maps to:
    - a specific manifest entry (one patch center)
    - a specific 15-min frame offset within that patch's 3-frame window
  The zarr time index decodes as
    ``t_idx = (entry.start_offset_min // 15) + frame_offset``.

* Each sample is spatially sliced to a 1000×1000 km window centered on
  the patch's lat/lon. Per-patch lat/lon replace the parent's static
  ``self.lat_lon`` — passed into the sample xds so downstream forcing
  computation (cos_lat, sin_lat, insolation, …) produces patch-local
  values.

* Trajectory IDs are per-(case, start, patch) triples. The anemoi
  zarr's ``trajectory_ids`` ends up as an integer array of length
  ``n_inits × n_patches_per_init × n_frames_per_sample`` with the same
  trajectory id repeated ``n_frames_per_sample`` times per patch.

Requires the manifest JSON produced by ``build_patch_manifest.py`` —
which pre-computes each (case, start, patch) tuple and their lat/lon
centers ahead of the main dataset build.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Literal, Optional

import netCDF4  # noqa: F401 — parent class imports
import numpy as np
import pandas as pd
import xarray as xr

# Manifests may carry replica suffixes (``<case_str>@r<n>``) so that the same
# underlying case can contribute several instances with distinct random
# start-offsets + patches (precip-rank upsampling). Strip before URL lookup.
_REPLICA_SUFFIX = re.compile(r"@r\d+$")

from ufs2arco.sources import Source
from ufs2arco.sources.aws_graf_reforecast_regridded import AWSGRAFRegriddedArchive
from ufs2arco.sources.graf_utils import (
    apply_raymond_filter_to_dataset,
    compute_reflectivity_3d,
)
from ufs2arco.transforms.destagger import destagger
from ufs2arco.transforms.temporal_aggregation import temporal_aggregation

logger = logging.getLogger("ufs2arco")

# GRAF regridded 4 km Lambert Conformal grid pixel size (km).
# Not HRRR's LC — GRAF uses its own projection; the 4 km pixel size happens
# to match HRRR's, so the pixel-km math is the same.
_PIXEL_KM = 4.0


# Trajectory IDs are assigned as sequential small ints (matches the Oklahoma
# dataset convention via np.repeat(np.arange(n_init), n_steps)). The
# human-readable "<case>:<start_offset>:<patch_idx>" string is preserved
# in self.trajectory_id_dict for inspection.


class AWSGRAFRegriddedPatchesArchive(AWSGRAFRegriddedArchive):
    """Patch-based variant of AWSGRAFRegriddedArchive.

    Consumes a pre-built manifest JSON listing (init_time, start_offset,
    patch_center) triples and produces one short 3-frame trajectory per
    triple.
    """

    # Same sample-dim convention as the parent — DataMover iteration unchanged.
    sample_dims = ("init_time", "forecast_step")

    # Each sample is a different spatial slice → static vars (ter, landmask,
    # per-patch lat/lon) differ across samples even though their tendency
    # within a trajectory is 0. See Source.statics_vary_per_sample docstring.
    statics_vary_per_sample = True

    def __init__(
        self,
        manifest_path: str,
        file_freqstr: Literal["05m", "15m"],
        n_frames_per_sample: int = 3,
        patch_size_km: float = 1000.0,
        variables: Optional[list] = None,
        static_variables: list = None,
        levels: Optional[list | tuple] = None,
        static_regridded_file_path: str = None,
        destagger_kwargs=None,
        temporal_aggregation_kwargs: Optional[dict] = None,
        raymond_filter_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize the patch-based source from a manifest.

        Parameters
        ----------
        manifest_path : str
            Path to the manifest JSON produced by build_patch_manifest.py.
            The manifest defines WHICH patches to sample from each case.
        file_freqstr : "05m" or "15m"
            5-min or 15-min zarr source. For apcp-style sums use "05m"
            with ``temporal_aggregation_kwargs`` doing a 3-frame sum
            to yield 15-min output. Prognostic vars use "15m" directly.
        n_frames_per_sample : int, default 3
            Frames per trajectory (2 input + 1 target).
        patch_size_km : float, default 1000.0
            Side length of the square patch in km. Patch bbox is
            resolved in HRRR grid pixels (4 km), with patch cropped to
            ``patch_size_km / 4`` pixels per side.
        variables, static_variables, levels, ... : same as parent.
        """
        # ---- Load manifest + group by init_time ----
        self.manifest_path = manifest_path
        with open(manifest_path) as f:
            manifest = json.load(f)
        all_entries = manifest["entries"]
        self.manifest_meta = manifest.get("metadata", {})

        self.manifest_by_init: dict[str, list[dict]] = {}
        for e in all_entries:
            self.manifest_by_init.setdefault(e["init_time"], []).append(e)

        # Enforce uniform patch count per case so forecast_step range is
        # identical across init_times (required by DataMover's Cartesian
        # iteration). build_patch_manifest.py is expected to enforce this
        # invariant — we just validate.
        patch_counts = {k: len(v) for k, v in self.manifest_by_init.items()}
        unique_counts = set(patch_counts.values())
        if len(unique_counts) != 1:
            mismatched = {k: c for k, c in patch_counts.items() if c != max(unique_counts)}
            raise ValueError(
                f"Manifest {manifest_path!r} has varying patch count per case "
                f"(unique counts: {unique_counts}). This variant requires each "
                f"case to have the same number of patches. Cases with fewer: "
                f"{list(mismatched.items())[:5]}..."
            )
        self.n_patches_per_init = unique_counts.pop()

        # Sort each case's patches by (start_offset, patch_idx) for stable order
        for init in self.manifest_by_init:
            self.manifest_by_init[init].sort(
                key=lambda e: (e["start_offset_min"], e["patch_idx"]),
            )

        self.n_frames_per_sample = int(n_frames_per_sample)
        self.patch_size_km = float(patch_size_km)
        self.patch_size_pix = int(round(self.patch_size_km / _PIXEL_KM))

        # ---- Build our own init_time list from manifest (ignore init_times arg) ----
        self.init_time = sorted(self.manifest_by_init.keys())
        self._init_time_dt_map = {
            ic: pd.to_datetime(ic.split("_")[0], format="%Y%m%d%H")
            for ic in self.init_time
        }
        self.init_time_dts = pd.to_datetime(
            [self._init_time_dt_map[ic] for ic in self.init_time]
        )
        # init_times_df mimics parent's attr; used by __str__ etc.
        self.init_times_df = pd.DataFrame(
            {"case_str": self.init_time},
            index=self.init_time_dts,
        )
        self.init_times_df.index.name = "init_time"

        self.file_freqstr = file_freqstr
        self._storage_options = {"anon": False} if self.BUCKET.startswith("s3://") else {}
        self.variables = variables
        self.static_variables = static_variables
        self.levels = levels
        self.static_regridded_file_path = static_regridded_file_path
        self.static_file_path = static_regridded_file_path
        self.destagger_kwargs = destagger_kwargs
        self.temporal_aggregation_kwargs = temporal_aggregation_kwargs
        self.raymond_filter_kwargs = raymond_filter_kwargs

        # Placeholder: needed by parent methods. We DON'T use a single
        # geographic_extent because each patch has its own bbox.
        self.geographic_extent = None
        # lead_times not used for iteration but kept for __str__ compatibility.
        self.lead_times = {"start": "0h", "stop": "24h"}

        # ---- Load static vars + full-grid lat/lon (no geo-extent slicing) ----
        self._load_static_file_regridded(static_regridded_file_path)
        # NOTE: skip _cache_geo_extent_2d — we slice per-patch at sample time.

        self._ocean_mask = None

        # Build slices (level-only; no cell-based geo slicing)
        slices: dict = {"isel": {}, "sel": {}}
        if levels:
            slices["isel"]["level"] = levels
            if variables and "smois" in variables:
                slices["isel"]["nSoilLevels"] = 0

        # Initialize via Source (grandparent), skipping the parent __init__.
        Source.__init__(self, variables, levels=None, use_nearest_levels=False, slices=slices)

        # ---- Compute valid_times + trajectory_ids per (init, patch, frame) ----
        self._compute_valid_times()

        self._zarr_cache_per_init_time: dict = {}
        self._base_xds_cache_per_init_time: dict = {}

        logger.info(
            "%s: %d cases × %d patches/case × %d frames = %d total samples "
            "(patch_size=%d px, freq=%s, manifest=%s)",
            self.name, len(self.init_time), self.n_patches_per_init,
            self.n_frames_per_sample, self.n_samples, self.patch_size_pix,
            file_freqstr, manifest_path,
        )

    @property
    def name(self) -> str:
        return "AWSGRAFRegriddedPatchesArchive"

    def _build_path(self, init_time: str) -> str:
        """Build the zarr URL, stripping any ``@r<n>`` replica suffix.

        Case replicas share the same underlying data — only the random
        start-offsets and patch centers differ across replicas.
        """
        case_str = _REPLICA_SUFFIX.sub("", init_time)
        return f"{self.BUCKET}{case_str}/mpasout_{self.file_freqstr}.zarr"

    # ------------------------------------------------------------------
    # Iteration: forecast_step enumerates (patch_idx × n_frames) per init
    # ------------------------------------------------------------------
    def _compute_valid_times(self) -> None:
        """Build per-(init, patch, frame) valid_times + trajectory_ids arrays.

        forecast_step s ∈ [0, n_steps):
            patch_idx    = s // n_frames_per_sample
            frame_offset = s %  n_frames_per_sample

        ``self.trajectory_ids`` is length (n_init × n_steps,) with the
        same int id repeated n_frames times for each patch.
        """
        self.n_steps = self.n_patches_per_init * self.n_frames_per_sample
        self.forecast_offset = 0  # unused, kept for parent compatibility
        self.forecast_step = np.arange(self.n_steps, dtype=int)

        all_valid_times: list[pd.Timestamp] = []
        all_traj_ids: list[int] = []
        # Sequential small-int IDs (matches the Oklahoma dataset convention).
        # Keep the human-readable string mapping for inspection / tracing back
        # to the manifest entry.
        self.trajectory_id_dict: dict[str, int] = {}
        next_int_id = 0

        for init_time in self.init_time:
            init_dt = self._init_time_dt_map[init_time]
            entries = self.manifest_by_init[init_time]
            for patch_idx_slot in range(self.n_patches_per_init):
                entry = entries[patch_idx_slot]
                traj_id_str: str = entry["trajectory_id"]
                traj_id_int = next_int_id
                next_int_id += 1
                self.trajectory_id_dict[traj_id_str] = traj_id_int

                start_off_td = pd.Timedelta(minutes=int(entry["start_offset_min"]))
                for f in range(self.n_frames_per_sample):
                    vt = init_dt + start_off_td + pd.Timedelta(minutes=15 * f)
                    all_valid_times.append(vt)
                    all_traj_ids.append(traj_id_int)

        self.valid_times = pd.DatetimeIndex(all_valid_times)
        self.trajectory_ids = np.array(all_traj_ids, dtype=np.int64)
        self.n_samples = len(self.valid_times)
        assert self.n_samples == len(self.trajectory_ids), \
            "trajectory_ids and valid_times length mismatch"

    # ------------------------------------------------------------------
    # Sample dispatch: decode forecast_step → (patch, frame)
    # ------------------------------------------------------------------
    def _decode_forecast_step(self, forecast_step: int) -> tuple[dict, int]:
        """Return (manifest_entry, frame_offset) for a given forecast_step."""
        patch_idx = forecast_step // self.n_frames_per_sample
        frame_offset = forecast_step % self.n_frames_per_sample
        return patch_idx, frame_offset

    def _patch_yx_slice(self, lat_c: float, lon_c: float) -> tuple[slice, slice]:
        """Pixel-index slice into the full HRRR grid for a patch centered
        at (lat_c, lon_c) in the HRRR grid's lat/lon convention.

        Uses nearest-pixel matching + ``patch_size_pix / 2`` half-width.
        If the nominal slice would run off the grid edge, the slice is
        SHIFTED INWARD so the full ``(patch_size_pix, patch_size_pix)``
        window always fits. This keeps sample shape uniform (required
        by the downstream zarr writer) at the cost of shifting edge
        patches by up to ``half`` pixels; the stored per-patch lat/lon
        reflect the actual (shifted) center, so downstream consumers
        see the real location of the data, not the manifest's intent.
        """
        lat_2d = self.lat_lon["latitude"][-1]    # (y, x)
        lon_2d = self.lat_lon["longitude"][-1]   # in 360° convention
        H, W = lat_2d.shape
        # Convert input lon to 360° for matching
        lon_match = lon_c if lon_c >= 0 else lon_c + 360.0
        d2 = (lat_2d - lat_c) ** 2 + (lon_2d - lon_match) ** 2
        y_c, x_c = np.unravel_index(int(np.argmin(d2)), d2.shape)
        half = self.patch_size_pix // 2
        # Clamp the low-corner so [y_lo, y_lo + patch_size_pix) is fully
        # inside [0, H). Same for x.
        y_lo = max(0, min(H - self.patch_size_pix, y_c - half))
        y_hi = y_lo + self.patch_size_pix
        x_lo = max(0, min(W - self.patch_size_pix, x_c - half))
        x_hi = x_lo + self.patch_size_pix
        return slice(int(y_lo), int(y_hi)), slice(int(x_lo), int(x_hi))

    def get_valid_time(self, xds: xr.Dataset, **dims) -> pd.Timestamp:
        """Look up valid_time by flat sample index = init_idx × n_steps + forecast_step."""
        init_idx = self.init_time.index(dims["init_time"])
        sample_idx = init_idx * self.n_steps + dims["forecast_step"]
        return self.valid_times[sample_idx]

    # ------------------------------------------------------------------
    # Override add_static_vars so per-patch lat/lon get added (not global)
    # ------------------------------------------------------------------
    def _add_patch_static_vars(
        self, xds: xr.Dataset, y_slice: slice, x_slice: slice,
    ) -> xr.Dataset:
        """Add static vars + lat/lon SLICED TO THIS PATCH."""
        for v, (dims, arr) in self.lat_lon.items():
            xds[v] = xr.DataArray(arr[y_slice, x_slice], dims=dims)
        for v, (dims, arr) in self.static_vars.items():
            xds[v] = xr.DataArray(arr[y_slice, x_slice], dims=dims)
        return xds

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------
    def open_sample_dataset(
        self,
        dims: dict,
        open_static_vars: bool,
        cache_dir: Optional[str] = None,
    ) -> xr.Dataset:
        """Open one (init_time, forecast_step) sample → a single patch frame.

        Steps:
          1. Resolve (patch_idx, frame_offset) from forecast_step.
          2. Look up the manifest entry for this (init, patch).
          3. Decide the 15-min time index in the case.
          4. Open the case's zarr (cached per init_time).
          5. Select the right time frame.
          6. Compute patch (y, x) slice from the patch center.
          7. Apply the spatial slice + patch-local static vars.
          8. Run the usual post-processing (diagnostics, filter, slicing…).
        """
        init_time = dims["init_time"]
        forecast_step = int(dims["forecast_step"])

        patch_idx, frame_offset = self._decode_forecast_step(forecast_step)
        entries = self.manifest_by_init[init_time]
        entry = entries[patch_idx]

        # Open + cache the full case zarr
        if init_time in self._base_xds_cache_per_init_time:
            xds = self._base_xds_cache_per_init_time[init_time]
        else:
            xds = self._open_zarr({"init_time": init_time})
            xds = self._prepare_base_xds(xds)
            self._base_xds_cache_per_init_time[init_time] = xds

        valid_time = self.get_valid_time(xds, **dims)

        # Time index: 15-min cadence for forecast_step enumeration. For
        # 5m files, the parent's select_time is able to pick the right
        # 5-min frames that correspond to this 15-min output window
        # (via TIMESTEP_RATIO in the grandparent). Here we convert our
        # flat forecast_step into the case's lead-time offset in 15-min
        # multiples and delegate to select_time.
        start_off_min = int(entry["start_offset_min"])
        time_offset_15m = (start_off_min // 15) + frame_offset
        xds = self.select_time(xds, time_offset_15m)

        # Patch slice
        y_slice, x_slice = self._patch_yx_slice(entry["lat_c"], entry["lon_c"])
        xds = xds.isel(y=y_slice, x=x_slice)

        # Per-patch static vars + lat/lon
        xds = self._add_patch_static_vars(xds, y_slice, x_slice)

        if self.destagger_kwargs is not None:
            xds = destagger(xds, **self.destagger_kwargs)

        # Raymond filter: same as parent — remove grid-scale artifacts
        if self.raymond_filter_kwargs is not None:
            xds = apply_raymond_filter_to_dataset(xds, **self.raymond_filter_kwargs)

        # Diagnostic variables (comp_refl, refl_3d, geopot, etc.)
        diag_vars = {"comp_refl", "refl_3d", "geopot", "rho", "theta_m"}
        if diag_vars & set(self.variables):
            xds = self.add_diagnostic_variables(xds, self.variables)
            if "refl_3d" in self.variables and "refl_3d" not in xds.data_vars:
                xds = compute_reflectivity_3d(xds)
            xds = xds[self.variables]

        xds = self.apply_slices(xds)
        xds = self.maybe_soil_impute_over_ocean(xds)
        xds = self.maybe_rename_wind_vars(xds)

        if self.temporal_aggregation_kwargs:
            xds = temporal_aggregation(xds, **self.temporal_aggregation_kwargs)

        xds = self.maybe_rescale_cloud_cover(xds)
        xds = self.add_valid_time_to_dataset(xds, valid_time)

        # Metadata for downstream steps
        xds.attrs["stack_order"] = ["y", "x"]
        xds.attrs["init_time"] = init_time
        xds.attrs["forecast_step"] = forecast_step
        xds.attrs["patch_idx"] = patch_idx
        xds.attrs["frame_offset"] = frame_offset
        xds.attrs["patch_center_lat"] = float(entry["lat_c"])
        xds.attrs["patch_center_lon"] = float(entry["lon_c"])
        xds.attrs["patch_source"] = entry["source"]
        xds.attrs["trajectory_id_str"] = entry["trajectory_id"]

        # CRITICAL: Many patches within a single (case, start) share the SAME
        # valid_time (they're time-aligned, differ only spatially), which would
        # collide onto one output time slot via the target's default
        # valid_time-based indexing. Set _sample_index so the Anemoi target
        # uses our flat sample position as the time-axis index — same hook
        # WoFSCast uses for ensemble members sharing valid_times.
        init_idx = self.init_time.index(init_time)
        xds.attrs["_sample_index"] = init_idx * self.n_steps + forecast_step

        return xds
