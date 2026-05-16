"""Unit tests for the ufs2arco CLI argparse + driver-dispatch logic.

Driver/MultiDriver are mocked — these tests only verify that flags are
parsed correctly and forwarded to driver.run() with the right kwargs.
"""
from __future__ import annotations

from unittest import mock

import pytest

from ufs2arco import cli


def _write_recipe(tmp_path, multisource=False):
    """Write a minimal recipe yaml. Content doesn't matter — Driver is mocked."""
    path = tmp_path / "recipe.yaml"
    if multisource:
        path.write_text("multisource:\n  - source: {name: dummy}\n")
    else:
        path.write_text("source: {name: dummy}\n")
    return str(path)


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["ufs2arco"] + argv)
    cli.main()


def test_tranche_and_finalize_are_mutually_exclusive(monkeypatch, tmp_path):
    recipe = _write_recipe(tmp_path)
    monkeypatch.setattr("sys.argv", ["ufs2arco", recipe, "--tranche", "0", "--finalize"])
    with mock.patch.object(cli, "Driver") as MockDriver:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    # argparse parser.error exits with code 2
    assert exc.value.code == 2
    MockDriver.assert_not_called()


def test_plain_run_forwards_default_kwargs_to_driver(monkeypatch, tmp_path):
    recipe = _write_recipe(tmp_path)
    with mock.patch.object(cli, "Driver") as MockDriver:
        _run_cli(monkeypatch, [recipe])
    MockDriver.assert_called_once_with(recipe)
    MockDriver.return_value.run.assert_called_once_with(
        overwrite=False,
        tranche_id=None,
        finalize_only=False,
        force=False,
    )


def test_tranche_flag_forwarded(monkeypatch, tmp_path):
    recipe = _write_recipe(tmp_path)
    with mock.patch.object(cli, "Driver") as MockDriver:
        _run_cli(monkeypatch, [recipe, "--tranche", "3"])
    kwargs = MockDriver.return_value.run.call_args.kwargs
    assert kwargs["tranche_id"] == 3
    assert kwargs["finalize_only"] is False
    assert kwargs["force"] is False


def test_finalize_and_force_forwarded(monkeypatch, tmp_path):
    recipe = _write_recipe(tmp_path)
    with mock.patch.object(cli, "Driver") as MockDriver:
        _run_cli(monkeypatch, [recipe, "--finalize", "--force"])
    kwargs = MockDriver.return_value.run.call_args.kwargs
    assert kwargs["tranche_id"] is None
    assert kwargs["finalize_only"] is True
    assert kwargs["force"] is True


def test_overwrite_forwarded(monkeypatch, tmp_path):
    recipe = _write_recipe(tmp_path)
    with mock.patch.object(cli, "Driver") as MockDriver:
        _run_cli(monkeypatch, [recipe, "--overwrite"])
    assert MockDriver.return_value.run.call_args.kwargs["overwrite"] is True


def test_multisource_dispatches_to_multidriver(monkeypatch, tmp_path):
    recipe = _write_recipe(tmp_path, multisource=True)
    with mock.patch.object(cli, "MultiDriver") as MockMulti, \
         mock.patch.object(cli, "Driver") as MockDriver:
        _run_cli(monkeypatch, [recipe, "--tranche", "2"])
    MockMulti.assert_called_once_with(recipe)
    MockDriver.assert_not_called()
    # MultiDriver.run also receives validate=
    kwargs = MockMulti.return_value.run.call_args.kwargs
    assert kwargs["tranche_id"] == 2
    assert kwargs["validate"] is False
