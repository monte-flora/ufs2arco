"""Tranche manifest helpers for multi-session ufs2arco builds.

A tranche manifest enumerates fixed `[start, stop)` batch ranges of an
otherwise single-shot ufs2arco build, so the user can ingest data in
sessions (e.g., when source data is staged 100-200 TB at a time).

Each session runs `ufs2arco recipe.yaml --tranche=N`; the driver looks
up tranche N's batch range and ingests only that subset, skipping
finalize. After all tranches are `done`, the user runs
`ufs2arco recipe.yaml --finalize` to compute statistics over the full
zarr. Re-running the same tranche is idempotent — `to_zarr(region=...)`
silently overwrites chunks.

Manifest path convention: alongside the recipe yaml, with `.tranches.yaml`
appended (e.g., `build.yaml` -> `build.tranches.yaml`).
"""
import argparse
import hashlib
import itertools
import os
from datetime import datetime
from math import ceil

import yaml


VALID_STATES = ("pending", "running", "done", "failed")


def default_tranche_path(recipe_path: str) -> str:
    """Tranche manifest path next to the recipe yaml."""
    base, _ = os.path.splitext(recipe_path)
    return f"{base}.tranches.yaml"


def hash_recipe(recipe_path: str) -> str:
    """SHA-256 of the raw recipe-yaml bytes — drift detection between sessions."""
    with open(recipe_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _instantiate_first_source(config: dict):
    """Return the first source object — used to enumerate sample_indices."""
    import ufs2arco.sources

    if "multisource" in config:
        src_cfg = config["multisource"][0]["source"]
    else:
        src_cfg = config["source"]
    name = src_cfg["name"].lower()
    SourceCls = getattr(ufs2arco.sources, ufs2arco.sources._recognized[name])
    kw = {k: v for k, v in src_cfg.items() if k != "name"}
    return SourceCls(**kw)


def compute_total_samples(recipe_path: str) -> int:
    """Build the source's sample-index cartesian product and return its length."""
    with open(recipe_path) as f:
        config = yaml.safe_load(f)
    source = _instantiate_first_source(config)
    iterations = [getattr(source, key) for key in source.sample_dims]
    total = 1
    for v in iterations:
        total *= len(v)
    return total


def init_template(
    recipe_path: str,
    num_tranches: int,
    mpi_batch_size: int,
    out_path: str = None,
    overwrite: bool = False,
):
    """Generate an initial tranche manifest with N evenly-split batch ranges.

    Args:
        recipe_path: path to the ufs2arco recipe yaml.
        num_tranches: number of tranches to split the build into.
        mpi_batch_size: MPI rank count that will be used during the actual
            build (e.g., 576 for 6 nodes × 96 ranks). Tranche batch indices
            are tied to this size — re-running with a different MPI size
            will be rejected at runtime.
        out_path: optional override for the output manifest path.
        overwrite: if True, replace an existing manifest at out_path.

    Returns:
        (out_path, manifest_dict)
    """
    if out_path is None:
        out_path = default_tranche_path(recipe_path)
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(
            f"Tranche manifest already exists at {out_path}. "
            f"Pass --overwrite to replace it."
        )
    if num_tranches < 1:
        raise ValueError(f"num_tranches must be >= 1, got {num_tranches}")
    if mpi_batch_size < 1:
        raise ValueError(f"mpi_batch_size must be >= 1, got {mpi_batch_size}")

    total_samples = compute_total_samples(recipe_path)
    total_batches = ceil(total_samples / mpi_batch_size)

    if num_tranches > total_batches:
        raise ValueError(
            f"Asked for {num_tranches} tranches but the build has only "
            f"{total_batches} batches at batch_size={mpi_batch_size}."
        )

    base = total_batches // num_tranches
    rem = total_batches % num_tranches
    cursor = 0
    tranches = []
    for i in range(num_tranches):
        size = base + (1 if i < rem else 0)
        tranches.append({
            "id": i,
            "start": cursor,
            "stop": cursor + size,
            "description": f"tranche {i}",
            "state": "pending",
            "samples_processed": None,
            "started_utc": None,
            "finished_utc": None,
        })
        cursor += size

    with open(recipe_path) as f:
        recipe = yaml.safe_load(f)
    zarr_path = recipe["directories"]["zarr"]

    manifest = {
        "zarr": zarr_path,
        "config_hash": hash_recipe(recipe_path),
        "total_samples": total_samples,
        "total_batches": total_batches,
        "batch_size": mpi_batch_size,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "tranches": tranches,
    }
    save_manifest(out_path, manifest)
    return out_path, manifest


def load_manifest(path: str, recipe_path: str = None) -> dict:
    """Load and (optionally) verify the manifest matches the recipe hash."""
    with open(path) as f:
        manifest = yaml.safe_load(f)
    if recipe_path is not None:
        actual = hash_recipe(recipe_path)
        if manifest.get("config_hash") != actual:
            raise RuntimeError(
                f"Recipe hash mismatch for {recipe_path}.\n"
                f"  Manifest hash: {manifest.get('config_hash')}\n"
                f"  Current hash:  {actual}\n"
                f"YAML changes between tranches change batch-index semantics. "
                f"If this change is intentional, regenerate the manifest with "
                f"`python -m ufs2arco.tranches init`."
            )
    return manifest


def save_manifest(path: str, manifest: dict) -> None:
    """Atomic write via tmp file + rename."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    os.replace(tmp, path)


def update_state(path: str, tranche_id: int, **fields) -> None:
    """Update one tranche's fields. Atomic; intended to be called from rank 0 only."""
    manifest = load_manifest(path)
    for tr in manifest["tranches"]:
        if tr["id"] == tranche_id:
            for k, v in fields.items():
                tr[k] = v
            save_manifest(path, manifest)
            return
    raise KeyError(f"Tranche id {tranche_id} not found in {path}")


def get_tranche(manifest: dict, tranche_id: int) -> dict:
    for tr in manifest["tranches"]:
        if tr["id"] == tranche_id:
            return tr
    raise KeyError(f"Tranche id {tranche_id} not found")


def verify_complete(manifest: dict):
    """Return (ok, pending_ids). ok=True iff every tranche is 'done'."""
    pending = [tr["id"] for tr in manifest["tranches"] if tr["state"] != "done"]
    return (len(pending) == 0, pending)


def _main():
    parser = argparse.ArgumentParser(
        prog="python -m ufs2arco.tranches",
        description="Tranche manifest tools for multi-session ufs2arco builds.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Generate a tranche manifest template.")
    p_init.add_argument("recipe", type=str, help="Path to the ufs2arco recipe yaml.")
    p_init.add_argument("--num-tranches", type=int, required=True,
                        help="Number of evenly-sized tranches to split the build into.")
    p_init.add_argument("--mpi-batch-size", type=int, required=True,
                        help="MPI rank count expected during builds (e.g., 576 for 6 nodes x 96).")
    p_init.add_argument("--out", type=str, default=None,
                        help="Output manifest path (default: <recipe-stem>.tranches.yaml).")
    p_init.add_argument("--overwrite", action="store_true",
                        help="Replace an existing manifest at the output path.")

    p_show = sub.add_parser("show", help="Print summary of an existing manifest.")
    p_show.add_argument("manifest", type=str)

    args = parser.parse_args()
    if args.cmd == "init":
        out, manifest = init_template(
            args.recipe, args.num_tranches, args.mpi_batch_size,
            out_path=args.out, overwrite=args.overwrite,
        )
        print(f"Wrote {out}")
        print(f"  total_samples = {manifest['total_samples']}")
        print(f"  total_batches = {manifest['total_batches']} (batch_size = {manifest['batch_size']})")
        for tr in manifest["tranches"]:
            print(f"  tranche {tr['id']}: batches [{tr['start']}, {tr['stop']})  state={tr['state']}")
    elif args.cmd == "show":
        m = load_manifest(args.manifest)
        ok, pending = verify_complete(m)
        print(f"zarr:          {m['zarr']}")
        print(f"config_hash:   {m['config_hash'][:16]}...")
        print(f"total_samples: {m['total_samples']}")
        print(f"total_batches: {m['total_batches']} (batch_size = {m['batch_size']})")
        print(f"created_utc:   {m['created_utc']}")
        print(f"tranches: ({len(m['tranches'])} total, {'all done' if ok else f'pending: {pending}'})")
        for tr in m["tranches"]:
            print(f"  {tr['id']:3d}: [{tr['start']:>7d}, {tr['stop']:>7d})  state={tr['state']:<8s}  {tr['description']}")


if __name__ == "__main__":
    _main()
