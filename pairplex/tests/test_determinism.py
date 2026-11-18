"""Same input + same options -> identical outputs, whatever the worker count.

Runs PairPlex (default options) on the two golden wells plus a third, multi-chunk well
(> 1000 reads, so barcode parsing is split over several chunks), once with 1 worker and once
with 2, then compares every output file byte for byte, and the ``pair-pooled`` result of both.
"""

import gzip
import os
import shutil
from pathlib import Path

import pytest
from polars.testing import assert_frame_equal

import pairplex.pairplex as ppx
from pairplex.pooled_pairing import run_pooled_pairing
from pairplex.tests.golden import INPUT

# outputs that must not change between runs (logs carry timestamps and the worker count)
COMPARED = ["metadata", "consensus", "annotated", "pairs", "qc"]


def _multi_chunk_well(path: Path) -> None:
    """well002 = well000 + well001 + well000 again (read ids suffixed): 1499 reads, 2 chunks."""
    with gzip.open(path, "wt") as out:
        for src, suffix in (("well000", ""), ("well001", ""), ("well000", "-b")):
            with gzip.open(INPUT / f"{src}.fastq.gz", "rt") as f:
                for i, line in enumerate(f):
                    if i % 4 == 0 and suffix:
                        line = line.split()[0] + suffix + "\n"
                    out.write(line)


def _run(work: Path, inp: Path, workers: int) -> Path:
    old = os.environ.get("PAIRPLEX_MAX_WORKERS")
    os.environ["PAIRPLEX_MAX_WORKERS"] = str(workers)
    try:
        ppx.run(str(inp), str(work), quiet=True)
    finally:
        if old is None:
            os.environ.pop("PAIRPLEX_MAX_WORKERS", None)
        else:
            os.environ["PAIRPLEX_MAX_WORKERS"] = old
    return work


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("determinism")
    inp = base / "input"
    shutil.copytree(INPUT, inp)
    _multi_chunk_well(inp / "well002.fastq.gz")
    return _run(base / "w1", inp, 1), _run(base / "w2", inp, 2)


def _files(out: Path) -> dict[str, Path]:
    return {
        str(p.relative_to(out)): p
        for d in COMPARED
        if (out / d).is_dir()
        for p in sorted((out / d).rglob("*"))
        if p.is_file()
    }


def test_outputs_byte_identical_across_worker_counts(two_runs):
    a, b = (_files(o) for o in two_runs)
    assert a.keys() == b.keys()
    assert any(k.startswith("metadata/well002") for k in a), "multi-chunk well missing"
    differ = [k for k in a if a[k].read_bytes() != b[k].read_bytes()]
    assert not differ, f"outputs differ between 1 and 2 workers: {differ}"


def test_metadata_in_fixed_row_order(two_runs):
    import polars as pl
    from natsort import natsorted

    for f in sorted((two_runs[0] / "metadata").glob("*.csv")):
        names = pl.read_csv(f)["name"].to_list()
        assert names == natsorted(names), f.name


def test_pooled_pairing_identical(two_runs):
    pa, ia = run_pooled_pairing(two_runs[0], output_file=two_runs[0] / "pooled" / "p.parquet")
    pb, ib = run_pooled_pairing(two_runs[1], output_file=two_runs[1] / "pooled" / "p.parquet")
    assert ia["leakage"] == ib["leakage"]
    assert_frame_equal(pa, pb)
    # the same input twice also gives the same pooled result
    pc, _ = run_pooled_pairing(two_runs[0], output_file=two_runs[0] / "pooled" / "p2.parquet")
    assert_frame_equal(pa, pc)
