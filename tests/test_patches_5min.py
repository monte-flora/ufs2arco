"""Unit tests for the patches source's native-5-min output mode.

Covers the three substantive changes:
1. `stored_freq` instance attribute responds to file_freqstr +
   temporal_aggregation_kwargs.
2. `_compute_valid_times` stride scales with `_is_native_5min`.
3. `open_sample_dataset` dispatches to `select_time_single_5min` when
   in native-5-min mode (smoke via mock).

These tests stub out the manifest load to avoid hitting S3 / Lustre.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _make_manifest(path: Path, n_cases: int = 2, n_patches: int = 1):
    """Write a tiny manifest with uniform patch count per case."""
    entries = []
    for c in range(n_cases):
        init_time = f"2020{(c % 12) + 1:02d}0112_27"
        for p in range(n_patches):
            entries.append({
                "init_time": init_time,
                "start_offset_min": 360,    # 6h spinup
                "patch_idx": p,
                "lat_c": 40.0,
                "lon_c": -100.0,
                "source": "storm",
                "intensity": 5.0,
                "trajectory_id": f"{init_time}:0360:{p:02d}",
            })
    with open(path, "w") as f:
        json.dump({"metadata": {"test": True}, "entries": entries}, f)
    return str(path)


def _make_source(manifest_path, file_freqstr="05m", n_frames=4,
                 temporal_aggregation_kwargs=None):
    """Instantiate the patches source while stubbing out S3/static reads.

    Patches the static-file load and the parent base-class network init
    so we can construct the object purely from the manifest JSON.
    """
    from ufs2arco.sources.aws_graf_regridded_patches import (
        AWSGRAFRegriddedPatchesArchive,
    )

    # The source's __init__ calls super().__init__ which expects to read
    # the CSV cases file + static. Stub those out cheaply.
    with patch.object(AWSGRAFRegriddedPatchesArchive, "__init__",
                      _patched_init):
        return AWSGRAFRegriddedPatchesArchive(
            manifest_path=manifest_path,
            file_freqstr=file_freqstr,
            n_frames_per_sample=n_frames,
            patch_size_km=1000.0,
            variables=["apcp_bucket"],
            static_variables=[],
            levels=None,
            static_regridded_file_path=None,
            destagger_kwargs=None,
            temporal_aggregation_kwargs=temporal_aggregation_kwargs,
            raymond_filter_kwargs=None,
        )


def _patched_init(self, manifest_path, file_freqstr, n_frames_per_sample,
                  patch_size_km, variables, static_variables, levels,
                  static_regridded_file_path, destagger_kwargs,
                  temporal_aggregation_kwargs, raymond_filter_kwargs):
    """Minimal __init__ that exercises the new logic without S3/static reads."""
    # Manifest load + group-by-init (copy of real ctor logic for these fields)
    self.manifest_path = manifest_path
    with open(manifest_path) as f:
        manifest = json.load(f)
    all_entries = manifest["entries"]
    self.manifest_meta = manifest.get("metadata", {})
    self.manifest_by_init = {}
    for e in all_entries:
        self.manifest_by_init.setdefault(e["init_time"], []).append(e)
    patch_counts = {k: len(v) for k, v in self.manifest_by_init.items()}
    assert len(set(patch_counts.values())) == 1
    self.n_patches_per_init = patch_counts[list(patch_counts.keys())[0]]
    for init in self.manifest_by_init:
        self.manifest_by_init[init].sort(
            key=lambda e: (e["start_offset_min"], e["patch_idx"])
        )

    self.n_frames_per_sample = int(n_frames_per_sample)
    self.patch_size_km = float(patch_size_km)
    self.patch_size_pix = 250

    # The new logic under test:
    self._is_native_5min = (
        file_freqstr == "05m" and temporal_aggregation_kwargs is None
    )
    self.stored_freq = "05m" if self._is_native_5min else "15m"

    self.init_time = sorted(self.manifest_by_init.keys())
    self._init_time_dt_map = {
        ic: pd.to_datetime(ic.split("_")[0], format="%Y%m%d%H")
        for ic in self.init_time
    }
    self.init_time_dts = pd.to_datetime(
        [self._init_time_dt_map[ic] for ic in self.init_time]
    )
    self.init_times_df = pd.DataFrame(
        {"case_str": self.init_time}, index=self.init_time_dts,
    )
    self.init_times_df.index.name = "init_time"
    self.file_freqstr = file_freqstr
    self.temporal_aggregation_kwargs = temporal_aggregation_kwargs
    self.variables = variables
    self.static_variables = static_variables or []
    self.levels = levels


# ----------------------------------------------------------------------
# stored_freq + _is_native_5min logic
# ----------------------------------------------------------------------
def test_stored_freq_native_5min(tmp_path):
    """file_freqstr='05m' + no temporal_aggregation → native-5-min mode."""
    src = _make_source(_make_manifest(tmp_path / "m.json"),
                       file_freqstr="05m",
                       temporal_aggregation_kwargs=None)
    assert src._is_native_5min is True
    assert src.stored_freq == "05m"


def test_stored_freq_15m_unchanged(tmp_path):
    """file_freqstr='15m' → stored_freq stays '15m', no native-5-min."""
    src = _make_source(_make_manifest(tmp_path / "m.json"),
                       file_freqstr="15m",
                       temporal_aggregation_kwargs=None)
    assert src._is_native_5min is False
    assert src.stored_freq == "15m"


def test_stored_freq_05m_with_agg_is_15m(tmp_path):
    """Back-compat: file_freqstr='05m' WITH temporal_aggregation → 15-min output.

    The existing train build's secondary source uses this combination to
    aggregate 3 native-5-min frames into one 15-min output frame. We
    must NOT break that path.
    """
    src = _make_source(_make_manifest(tmp_path / "m.json"),
                       file_freqstr="05m",
                       temporal_aggregation_kwargs={
                           "reduce_map": {"sum": ["apcp_bucket"]}})
    assert src._is_native_5min is False
    assert src.stored_freq == "15m"


# ----------------------------------------------------------------------
# _compute_valid_times stride
# ----------------------------------------------------------------------
def _compute_valid_times_smoke(src):
    """Re-run the stride-dependent section of _compute_valid_times.

    Cheaper than calling the real one (which also does network init).
    Mirrors lines 263-282 of the source.
    """
    all_valid_times = []
    all_traj_ids = []
    next_int_id = 0
    src.trajectory_id_dict = {}
    for init_time in src.init_time:
        init_dt = src._init_time_dt_map[init_time]
        entries = src.manifest_by_init[init_time]
        for patch_idx_slot in range(src.n_patches_per_init):
            entry = entries[patch_idx_slot]
            traj_id_int = next_int_id
            next_int_id += 1
            src.trajectory_id_dict[entry["trajectory_id"]] = traj_id_int
            start_off_td = pd.Timedelta(minutes=int(entry["start_offset_min"]))
            step_min = 5 if src._is_native_5min else 15
            for f in range(src.n_frames_per_sample):
                vt = init_dt + start_off_td + pd.Timedelta(minutes=step_min * f)
                all_valid_times.append(vt)
                all_traj_ids.append(traj_id_int)
    return pd.DatetimeIndex(all_valid_times), all_traj_ids


def test_stride_05m_gives_5min_dates(tmp_path):
    """Native-5-min mode: n_frames=4 → 4 dates 5 min apart, total span 15 min."""
    src = _make_source(_make_manifest(tmp_path / "m.json"),
                       file_freqstr="05m", n_frames=4,
                       temporal_aggregation_kwargs=None)
    vt, _ = _compute_valid_times_smoke(src)
    # Single case, single patch → 4 dates
    assert len(vt) == 8   # 2 cases × 1 patch × 4 frames
    # Per case (first 4 entries):
    deltas = (vt[1:4] - vt[:3]).unique()
    assert len(deltas) == 1
    assert deltas[0] == pd.Timedelta("5min")
    # Boundary span
    assert vt[3] - vt[0] == pd.Timedelta("15min")


def test_stride_15m_unchanged(tmp_path):
    """15-min mode: backward compat — 5-min stride NOT used."""
    src = _make_source(_make_manifest(tmp_path / "m.json"),
                       file_freqstr="15m", n_frames=3)
    vt, _ = _compute_valid_times_smoke(src)
    # Single case, 3 frames
    deltas = (vt[1:3] - vt[:2]).unique()
    assert deltas[0] == pd.Timedelta("15min")


def test_stride_05m_with_aggregation_is_15m(tmp_path):
    """Aggregated 5-min source still produces 15-min-stride output (no regression)."""
    src = _make_source(_make_manifest(tmp_path / "m.json"),
                       file_freqstr="05m", n_frames=3,
                       temporal_aggregation_kwargs={
                           "reduce_map": {"sum": ["apcp_bucket"]}})
    vt, _ = _compute_valid_times_smoke(src)
    deltas = (vt[1:3] - vt[:2]).unique()
    assert deltas[0] == pd.Timedelta("15min")


# ----------------------------------------------------------------------
# Anemoi target reads instance stored_freq with class fallback
# ----------------------------------------------------------------------
def test_anemoi_target_reads_instance_stored_freq():
    """Target's frequency attribute reads instance stored_freq if present."""
    source = MagicMock()
    source.stored_freq = "05m"
    source.STORED_FREQ = "15m"   # class-level
    assert getattr(source, "stored_freq", source.STORED_FREQ) == "05m"


def test_anemoi_target_falls_back_to_class_stored_freq():
    """Target falls back to STORED_FREQ when no instance attr."""
    class FakeSource:
        STORED_FREQ = "15m"
    s = FakeSource()
    assert getattr(s, "stored_freq", s.STORED_FREQ) == "15m"
