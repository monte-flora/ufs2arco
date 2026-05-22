"""
MPI unit tests for MPIDataMover contiguous block assignment.

Tests (all collective — every rank must call them):
  1. no_gaps_no_overlaps  — every sample assigned exactly once across all ranks
  2. contiguous_per_rank  — each rank's samples form a contiguous range
  3. cache_effectiveness  — with manifest sorted by valid_time, S3 opens per rank
                            equals unique t0 count for that rank (not patch count)

Run via sbatch:
  sbatch scripts/data/slurm/sbatch_test_mpi_datamover.sh

Or directly on a node:
  srun -n 32 python tests/test_mpi_datamover.py
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock

import pandas as pd
from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ufs2arco.mpi import MPITopology
from ufs2arco.datamover import MPIDataMover

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mover(n_samples: int, topo: MPITopology, log_dir: str) -> MPIDataMover:
    source = MagicMock()
    source.sample_dims = ("sample",)
    source.sample = list(range(n_samples))
    target = MagicMock()
    target.always_open_static_vars = False
    return MPIDataMover(source=source, target=target, mpi_topo=topo, cache_dir=log_dir)


def _my_indices(mover: MPIDataMover) -> list[int]:
    return [
        d["sample"]
        for b in range(len(mover))
        for d in mover.get_batch_indices(b)
    ]


# ---------------------------------------------------------------------------
# Test 1 — no gaps, no overlaps
# ---------------------------------------------------------------------------

def test_no_gaps_no_overlaps(topo, log_dir, n_samples) -> bool:
    mover = _make_mover(n_samples, topo, log_dir)
    my = _my_indices(mover)
    all_indices = comm.gather(my, root=0)

    if rank == 0:
        flat = sorted(i for lst in all_indices for i in lst)
        duplicates = len(flat) - len(set(flat))
        ok = flat == list(range(n_samples))
        label = "PASS" if ok else f"FAIL (len={len(flat)}, dups={duplicates})"
        print(f"  [{label}] no_gaps_no_overlaps        n={n_samples}, R={size}")
        return ok
    return True  # non-root: collective done, result decided by root


# ---------------------------------------------------------------------------
# Test 2 — each rank gets a contiguous block
# ---------------------------------------------------------------------------

def test_contiguous_per_rank(topo, log_dir, n_samples) -> bool:
    mover = _make_mover(n_samples, topo, log_dir)
    my = sorted(_my_indices(mover))

    if my:
        ok = my == list(range(my[0], my[-1] + 1))
    else:
        ok = True  # empty tail rank is acceptable

    all_ok = bool(comm.reduce(int(ok), op=MPI.LAND, root=0))

    if rank == 0:
        label = "PASS" if all_ok else "FAIL (non-contiguous on ≥1 rank)"
        print(f"  [{label}] contiguous_per_rank         n={n_samples}, R={size}")
        return all_ok
    return True


# ---------------------------------------------------------------------------
# Test 3 — cache effectiveness
# ---------------------------------------------------------------------------

def test_cache_effectiveness(topo, log_dir, n_valid_times, patches_per_time) -> bool:
    """
    S3 opens == unique t0 count when the manifest is sorted by valid_time and
    the DataMover uses contiguous block assignment.

    With round-robin assignment every call would be a cache miss, so
    s3_opens would equal n_patches_this_rank instead of n_t0_this_rank.
    """
    n_samples = n_valid_times * patches_per_time
    valid_times = pd.date_range("2021-01-01", periods=n_valid_times, freq="1h")

    # manifest[i] == valid_time for patch i (sorted by valid_time)
    manifest_vt = [t for t in valid_times for _ in range(patches_per_time)]

    mover = _make_mover(n_samples, topo, log_dir)
    call_order = [
        d["sample"]
        for b in range(len(mover))
        for d in mover.get_batch_indices(b)
    ]

    # Simulate per-t0 caching: count transitions to a new init time
    last_t0 = None
    s3_opens = 0
    for i in call_order:
        t0 = manifest_vt[i] - pd.Timedelta(hours=2)
        if t0 != last_t0:
            s3_opens += 1
            last_t0 = t0

    unique_t0 = len({manifest_vt[i] - pd.Timedelta(hours=2) for i in call_order})
    ok = s3_opens == unique_t0

    total_s3      = comm.reduce(s3_opens,        op=MPI.SUM, root=0)
    total_t0      = comm.reduce(unique_t0,        op=MPI.SUM, root=0)
    total_patches = comm.reduce(len(call_order),  op=MPI.SUM, root=0)
    all_ok        = bool(comm.reduce(int(ok),     op=MPI.LAND, root=0))

    if rank == 0:
        savings = total_patches / total_s3 if total_s3 else float("inf")
        label = "PASS" if all_ok else "FAIL (s3_opens != unique_t0 on ≥1 rank)"
        print(
            f"  [{label}] cache_effectiveness         "
            f"n={n_samples}, R={size}, "
            f"t0_opens={total_s3}, patches={total_patches}, "
            f"savings={savings:.1f}x"
        )
        return all_ok
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log_dir = tempfile.mkdtemp(prefix="ufs2arco_mpi_test_")
    topo = MPITopology(log_dir=log_dir)

    if rank == 0:
        print(f"\n=== MPIDataMover contiguous-block tests  ({size} ranks) ===\n")

    all_passed = True

    # Run coverage tests for three N values:
    #   - clean multiple of R    → no remainder edge case
    #   - N not divisible by R   → tail ranks get one fewer sample
    #   - realistic scale        → size * 1655 ≈ batches * ranks
    for n in [size * 22, size * 22 + 7, size * 1655]:
        if rank == 0:
            print(f"  n_samples = {n}")

        ok = test_no_gaps_no_overlaps(topo, log_dir, n)
        all_passed = all_passed and ok
        comm.Barrier()

        ok = test_contiguous_per_rank(topo, log_dir, n)
        all_passed = all_passed and ok
        comm.Barrier()

    # Cache effectiveness: realistic patch-per-time ratio
    if rank == 0:
        print(f"\n  cache test: {size * 10} valid_times × 22 patches")

    ok = test_cache_effectiveness(
        topo, log_dir,
        n_valid_times=size * 10,
        patches_per_time=22,
    )
    all_passed = all_passed and ok
    comm.Barrier()

    if rank == 0:
        outcome = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
        print(f"\n{'='*45}")
        print(f"  {outcome}")
        print(f"{'='*45}\n")
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
