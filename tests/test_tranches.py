"""Unit tests for the tranche manifest helpers (ufs2arco.tranches).

These cover the pure-Python helpers (no source instantiation, no MPI).
End-to-end multi-session builds are validated via the smoke recipe at
graf-ai/grafai/datasets/graf-regridded-oklahoma/oklahoma-tranche-smoke.yaml.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from ufs2arco import tranches


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _write_manifest(tmp_path, total_batches=20, num_tranches=4, batch_size=8):
    """Write a synthetic manifest by hand (skipping init_template, which would
    require instantiating a real source)."""
    base = total_batches // num_tranches
    rem = total_batches % num_tranches
    cursor = 0
    tr_list = []
    for i in range(num_tranches):
        size = base + (1 if i < rem else 0)
        tr_list.append({
            "id": i, "start": cursor, "stop": cursor + size,
            "description": f"tranche {i}", "state": "pending",
            "samples_processed": None, "started_utc": None, "finished_utc": None,
        })
        cursor += size
    manifest = {
        "zarr": str(tmp_path / "fake.zarr"),
        "config_hash": "abc123",
        "total_samples": total_batches * batch_size,
        "total_batches": total_batches,
        "batch_size": batch_size,
        "created_utc": "2026-01-01T00:00:00Z",
        "tranches": tr_list,
    }
    path = tmp_path / "fake.tranches.yaml"
    tranches.save_manifest(str(path), manifest)
    return str(path), manifest


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_default_tranche_path_appends_suffix():
    assert tranches.default_tranche_path("/x/y/build.yaml") == "/x/y/build.tranches.yaml"
    assert tranches.default_tranche_path("recipe.yml") == "recipe.tranches.yaml"


def test_hash_recipe_stable_across_reads(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("foo: bar\n")
    h1 = tranches.hash_recipe(str(p))
    h2 = tranches.hash_recipe(str(p))
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_hash_changes_when_yaml_changes(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("foo: bar\n")
    h1 = tranches.hash_recipe(str(p))
    p.write_text("foo: bar\n# trivial change\n")
    h2 = tranches.hash_recipe(str(p))
    assert h1 != h2


def test_save_then_load_roundtrip(tmp_path):
    path, m_in = _write_manifest(tmp_path)
    m_out = tranches.load_manifest(path)
    assert m_out["total_batches"] == m_in["total_batches"]
    assert m_out["batch_size"] == m_in["batch_size"]
    assert len(m_out["tranches"]) == 4


def test_load_with_recipe_hash_drift_detected(tmp_path):
    """If the recipe yaml's hash differs from the manifest's stored hash,
    load_manifest must error with both hashes in the message."""
    recipe = tmp_path / "r.yaml"
    recipe.write_text("foo: bar\n")
    path, _ = _write_manifest(tmp_path)
    # Manifest's config_hash is the synthetic 'abc123', recipe's is real sha256.
    with pytest.raises(RuntimeError, match="Recipe hash mismatch"):
        tranches.load_manifest(path, recipe_path=str(recipe))


def test_load_with_recipe_hash_match_succeeds(tmp_path):
    recipe = tmp_path / "r.yaml"
    recipe.write_text("foo: bar\n")
    actual_hash = tranches.hash_recipe(str(recipe))
    path, m = _write_manifest(tmp_path)
    m["config_hash"] = actual_hash
    tranches.save_manifest(path, m)
    # Should not raise.
    tranches.load_manifest(path, recipe_path=str(recipe))


def test_get_tranche_returns_dict(tmp_path):
    path, _ = _write_manifest(tmp_path)
    m = tranches.load_manifest(path)
    tr = tranches.get_tranche(m, 2)
    assert tr["id"] == 2
    assert tr["start"] == 10  # 20 batches / 4 tranches; even split = [0,5,10,15]


def test_get_tranche_missing_id_raises(tmp_path):
    path, _ = _write_manifest(tmp_path)
    m = tranches.load_manifest(path)
    with pytest.raises(KeyError):
        tranches.get_tranche(m, 99)


def test_update_state_persists_changes(tmp_path):
    path, _ = _write_manifest(tmp_path)
    tranches.update_state(path, 1, state="done", finished_utc="2026-01-02T00:00:00Z")
    m = tranches.load_manifest(path)
    tr = tranches.get_tranche(m, 1)
    assert tr["state"] == "done"
    assert tr["finished_utc"] == "2026-01-02T00:00:00Z"
    # Other tranches untouched
    assert tranches.get_tranche(m, 0)["state"] == "pending"


def test_verify_complete_reports_pending(tmp_path):
    path, _ = _write_manifest(tmp_path)
    m = tranches.load_manifest(path)
    ok, pending = tranches.verify_complete(m)
    assert not ok
    assert pending == [0, 1, 2, 3]

    for i in range(4):
        tranches.update_state(path, i, state="done")
    m = tranches.load_manifest(path)
    ok, pending = tranches.verify_complete(m)
    assert ok
    assert pending == []


def test_atomic_save_uses_rename(tmp_path):
    """save_manifest writes to a .tmp file then renames — readers never see a
    partial file. Verify the .tmp path doesn't linger after success."""
    path, _ = _write_manifest(tmp_path)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


def test_init_template_overwrite_protection(tmp_path):
    """Calling init_template twice without overwrite=True raises FileExistsError."""
    path, _ = _write_manifest(tmp_path)
    # init_template is the public entry; mock the source-loading dependency.
    # We can't easily run it without a real source, but we can verify the
    # FileExistsError is raised when out_path exists.
    recipe = tmp_path / "r.yaml"
    recipe.write_text("directories:\n  zarr: /nope\n")
    with pytest.raises(FileExistsError):
        tranches.init_template(
            recipe_path=str(recipe), num_tranches=2, mpi_batch_size=4,
            out_path=path, overwrite=False,
        )
