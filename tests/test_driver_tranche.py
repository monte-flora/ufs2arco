"""Unit tests for the tranche helpers on ufs2arco.driver.Driver.

We don't instantiate a real Driver (it parses a full recipe and builds
sources/targets/movers). Instead we construct a lightweight stand-in
with the attributes ``_apply_tranche`` and ``_run_finalize_only`` touch,
then invoke those methods as unbound functions on the stand-in.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from ufs2arco import tranches
from ufs2arco.driver import Driver


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------
def _write_recipe_and_manifest(
    tmp_path,
    num_tranches=4,
    total_batches=20,
    batch_size=8,
):
    """Write a minimal recipe yaml plus a hand-built tranche manifest with
    the correct config_hash. Returns (recipe_path, manifest_path)."""
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("directories: {zarr: /tmp/fake.zarr}\nsource: {name: dummy}\n")

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
        "zarr": "/tmp/fake.zarr",
        "config_hash": tranches.hash_recipe(str(recipe)),
        "total_samples": total_batches * batch_size,
        "total_batches": total_batches,
        "batch_size": batch_size,
        "created_utc": "2026-01-01T00:00:00Z",
        "tranches": tr_list,
    }
    manifest_path = tranches.default_tranche_path(str(recipe))
    tranches.save_manifest(manifest_path, manifest)
    return str(recipe), manifest_path


def _make_mock_driver(recipe_path, batch_size=8):
    """Stand-in with just the attributes _apply_tranche / _run_finalize_only need."""
    mover = SimpleNamespace(
        start=0,
        stop=None,
        counter=0,
        data_counter=0,
        batch_size=batch_size,
        restart=mock.MagicMock(),
    )
    topo = SimpleNamespace(is_root=True)
    return SimpleNamespace(
        config_filename=recipe_path,
        mover=mover,
        topo=topo,
        tranche_id=None,
        # _run_finalize_only also touches these:
        setup=mock.MagicMock(),
        target=SimpleNamespace(finalize=mock.MagicMock()),
        finalize_attributes=mock.MagicMock(),
        store_path="/tmp/fake.zarr",
    )


# ----------------------------------------------------------------------
# _apply_tranche
# ----------------------------------------------------------------------
def test_apply_tranche_overrides_mover_state(tmp_path):
    recipe, _ = _write_recipe_and_manifest(tmp_path)
    drv = _make_mock_driver(recipe)

    Driver._apply_tranche(drv, tranche_id=2, overwrite=False)

    # tranche 2 of {5,5,5,5}: [10, 15)
    assert drv.mover.start == 10
    assert drv.mover.stop == 15
    assert drv.mover.counter == 10
    assert drv.mover.data_counter == 10
    drv.mover.restart.assert_called_once_with(idx=10)
    assert drv.tranche_id == 2


def test_apply_tranche_rejects_batch_size_mismatch(tmp_path):
    recipe, _ = _write_recipe_and_manifest(tmp_path, batch_size=8)
    drv = _make_mock_driver(recipe, batch_size=16)  # mismatch

    with pytest.raises(RuntimeError, match="batch_size"):
        Driver._apply_tranche(drv, tranche_id=0, overwrite=False)


def test_apply_tranche_rejects_overwrite_on_nonzero_start(tmp_path):
    recipe, _ = _write_recipe_and_manifest(tmp_path)
    drv = _make_mock_driver(recipe)

    with pytest.raises(ValueError, match="overwrite"):
        Driver._apply_tranche(drv, tranche_id=1, overwrite=True)


def test_apply_tranche_allows_overwrite_for_tranche_zero(tmp_path):
    recipe, _ = _write_recipe_and_manifest(tmp_path)
    drv = _make_mock_driver(recipe)

    Driver._apply_tranche(drv, tranche_id=0, overwrite=True)
    assert drv.mover.start == 0


def test_apply_tranche_missing_manifest_raises(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("directories: {zarr: /tmp/fake.zarr}\n")
    drv = _make_mock_driver(str(recipe))

    with pytest.raises(FileNotFoundError, match="tranche manifest"):
        Driver._apply_tranche(drv, tranche_id=0, overwrite=False)


# ----------------------------------------------------------------------
# _run_finalize_only
# ----------------------------------------------------------------------
def test_run_finalize_only_raises_on_pending_tranches(tmp_path):
    """All tranches still 'pending' → must raise without --force."""
    recipe, _ = _write_recipe_and_manifest(tmp_path)
    drv = _make_mock_driver(recipe)

    with pytest.raises(RuntimeError, match="not done"):
        Driver._run_finalize_only(drv, force=False)

    drv.target.finalize.assert_not_called()
    drv.finalize_attributes.assert_not_called()


def test_run_finalize_only_force_proceeds_with_pending(tmp_path):
    """--force overrides the pending-tranches gate."""
    recipe, _ = _write_recipe_and_manifest(tmp_path)
    drv = _make_mock_driver(recipe)

    Driver._run_finalize_only(drv, force=True)

    drv.setup.assert_called_once_with(runtype="finalize")
    drv.target.finalize.assert_called_once_with(drv.topo)
    drv.finalize_attributes.assert_called_once_with()


def test_run_finalize_only_proceeds_when_all_done(tmp_path):
    """All tranches marked done → finalize runs without --force."""
    recipe, manifest_path = _write_recipe_and_manifest(tmp_path)
    m = tranches.load_manifest(manifest_path)
    for tr in m["tranches"]:
        tr["state"] = "done"
    tranches.save_manifest(manifest_path, m)

    drv = _make_mock_driver(recipe)
    Driver._run_finalize_only(drv, force=False)

    drv.target.finalize.assert_called_once()
    drv.finalize_attributes.assert_called_once()


def test_run_finalize_only_without_manifest_proceeds(tmp_path):
    """No manifest at all → finalize runs (single-shot recipe)."""
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("directories: {zarr: /tmp/fake.zarr}\n")
    drv = _make_mock_driver(str(recipe))

    Driver._run_finalize_only(drv, force=False)

    drv.target.finalize.assert_called_once()
    drv.finalize_attributes.assert_called_once()
