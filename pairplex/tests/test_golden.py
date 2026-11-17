"""End-to-end golden-output regression tests (real PairPlex run + abstar on a tiny fixed input).

1. The reference behaviour must not drift: with ``legacy=True`` (or plain ``run()`` before a legacy
   flag exists) the outputs must equal the stored golden tables.
   * PairPlex's own stages (cluster metadata, consensus sequences) and the pairing decisions
     (which contigs pair, their loci) are compared exactly.
   * Annotation calls (V/J genes, junction, productive) are compared exactly only when the
     installed abstar matches the version that made the golden tables; otherwise that part is
     skipped, because germline database updates can legitimately change calls.
2. The default (new) behaviour must run end to end and produce well-formed outputs.

Regenerate the golden tables only for an intended, reviewed change:
``python pairplex/tests/data/golden/make_golden.py``.
"""

import inspect

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pairplex.pairplex as ppx
from pairplex.tests.golden import ANNOTATION_CALLS, abstar_version, extract, load_expected, run_pairplex

HAS_LEGACY = "legacy" in inspect.signature(ppx.run).parameters
PAIR_CONTRACT = ["barcode", "well", "name", "confidence"] + [
    f"{c}:{i}" for i in (0, 1) for c in ("locus", "v_gene", "cdr3_aa", "sequence_id", "sequence")]


@pytest.fixture(scope="module")
def reference_tables(tmp_path_factory):
    kwargs = {"legacy": True} if HAS_LEGACY else {}
    return extract(run_pairplex(tmp_path_factory.mktemp("golden_ref"), **kwargs))


@pytest.fixture(scope="module")
def expected():
    return load_expected()


@pytest.mark.parametrize("table", ["metadata", "consensus", "paired"])
def test_reference_matches_golden(reference_tables, expected, table):
    exp, _ = expected
    assert_frame_equal(reference_tables[table], exp[table], check_dtypes=False)


def test_reference_unpaired_structure_matches_golden(reference_tables, expected):
    exp, _ = expected
    keep = ["well", "sequence_id", "sequence", "locus"]
    assert_frame_equal(reference_tables["unpaired"].select(keep), exp["unpaired"].select(keep), check_dtypes=False)


def test_reference_annotation_calls_match_golden(reference_tables, expected):
    exp, versions = expected
    if abstar_version() != versions["abstar"]:
        pytest.skip(f"abstar {abstar_version()} != golden {versions['abstar']}: annotation calls not compared")
    cols = ["well", "sequence_id"] + [c for c in ANNOTATION_CALLS if c in exp["unpaired"].columns]
    assert_frame_equal(reference_tables["unpaired"].select(cols), exp["unpaired"].select(cols), check_dtypes=False)


def test_golden_fixture_is_meaningful(expected):
    exp, _ = expected
    assert exp["paired"].height >= 10, "fixture should produce pairs"
    assert exp["unpaired"].height > exp["paired"].height * 2
    assert exp["metadata"].filter(~pl.col("pass_filters")).height > 0, "fixture should exercise the filters"


def test_default_mode_runs_and_is_well_formed(tmp_path):
    out = run_pairplex(tmp_path)
    t = extract(out)
    assert t["metadata"].height > 0 and t["consensus"].height > 0
    assert t["unpaired"].height > 0
    assert t["paired"].height > 0, "default mode produced no pairs on the golden input"
    pairs_file = out / "pairs" / "pairs.parquet"
    if pairs_file.exists():  # optional final-pairs contract (diagnosis/bench/README.md)
        cols = pl.read_parquet(pairs_file).columns
        missing = [c for c in PAIR_CONTRACT if c not in cols]
        assert not missing, f"pairs/pairs.parquet misses contract columns {missing}"
