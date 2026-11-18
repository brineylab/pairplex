"""Tests for pairplex.qc: stage counters, failure table, support columns and QC report metrics."""

import json
import warnings

import polars as pl
import pytest
from abutils import Sequence

from pairplex import qc


def _seq(sid, **ann):
    rec = {"sequence_id": sid, "sequence": "ACGT", "umi": None, **ann}
    return Sequence(rec, id_key="sequence_id", seq_key="sequence")


class _Pair:
    def __init__(self, nh, nl):
        self.heavies = [object()] * nh
        self.lights = [object()] * nl


# ---------------------------------------------------------------- stage counters


def test_read_stage_counts():
    parsed = pl.DataFrame({"barcode": ["A", "B"], "umi": ["x", "y"], "sequence": ["AC", "GT"]})
    assert qc.read_stage_counts(5, parsed) == {
        "input_reads": 5, "valid_barcode_reads": 2, "dropped_orientation_or_barcode": 3,
    }


def test_barcode_stage_counts_groups_and_empty():
    parsed = pl.DataFrame({
        "barcode": ["A"] * 4 + ["B"] * 2 + ["C"],
        "umi": ["u"] * 7,
        "sequence": ["ACG", "", "ACG", "", "", "ACG", "A"],
    })
    out = qc.barcode_stage_counts(parsed, min_cluster_reads=3)
    # B (2 reads) and C (1 read) dropped; empties counted only in kept group A
    assert out == {"barcode_groups": 3, "groups_dropped_min_cluster_reads": 2,
                   "reads_in_dropped_groups": 3, "empty_sequences_skipped": 2}


def test_barcode_stage_counts_empty_frame():
    parsed = pl.DataFrame(schema={"barcode": pl.String, "umi": pl.String, "sequence": pl.String})
    assert qc.barcode_stage_counts(parsed, 3)["barcode_groups"] == 0


def test_cluster_stage_counts_each_filter():
    meta = pl.DataFrame({
        "reads": [10, 2, 5, 1],
        "umis": [3, 1, 1, 1],
        "cluster_fraction": [0.6, 0.1, 0.25, 0.05],
        "pass_filters": [True, False, False, False],
    })
    out = qc.cluster_stage_counts(meta, min_cluster_reads=3, min_cluster_umis=2, min_cluster_fraction=0.2)
    assert out == {"clusters": 4, "clusters_fail_reads": 2, "clusters_fail_umis": 3,
                   "clusters_fail_fraction": 2, "clusters_pass": 1}


def _ann_seqs():
    return [
        _seq("A_contig-0", annotation_status="annotated", locus="IGH", productive=True,
             junction="TGT", complete_vdj=True, v_germline_start=1),
        _seq("A_contig-1", annotation_status="annotated", locus="IGK", productive=False,
             productivity_issues="junction does not end with conserved F|V/J junction is out of frame",
             junction="TGC", complete_vdj=False, v_germline_start=2),
        _seq("B_contig-0", annotation_status="annotated", locus="IGH", productive=False,
             productivity_issues="", junction=None, complete_vdj=False, v_germline_start=250),
        _seq("C_contig-0", annotation_status="unassigned", failure_reason="no compatible V gene assignment",
             locus=None, productive=None, junction=None),
    ]


def test_annotation_frame_and_stage_counts():
    ann = qc.annotation_frame(_ann_seqs())
    assert ann.height == 4 and set(qc.ANN_SCHEMA) == set(ann.columns)
    out = qc.annotation_stage_counts(ann, n_sent=6, n_record_failures=2)
    assert out == {
        "consensus_to_annotation": 6, "annotation_returned": 4, "annotated": 3,
        "failed_unassigned": 1, "failed_record_error": 2, "failed_annotation": 3,
        "productive": 1, "unproductive": 2, "no_junction": 1,
    }


def test_annotation_frame_single_sequence_and_none():
    assert qc.annotation_frame(_seq("A_contig-0", annotation_status="annotated")).height == 1
    assert qc.annotation_frame(None).height == 0


def test_record_errors_inferred_from_missing_rows():
    ann = qc.annotation_frame(_ann_seqs())
    assert qc.annotation_stage_counts(ann, n_sent=7)["failed_record_error"] == 3


def test_chain_composition_counts():
    pairs = [_Pair(1, 1), _Pair(1, 1), _Pair(1, 0), _Pair(0, 1), _Pair(1, 2), _Pair(2, 1),
             _Pair(2, 2), _Pair(0, 0), _Pair(3, 1), _Pair(0, 2)]
    out = qc.chain_composition_counts(pairs)
    assert out["barcodes_1H1L"] == 2
    assert out["barcodes_0H0L"] == 1
    assert out["barcodes_other"] == 2
    assert out["barcodes_with_chains"] == 10
    assert sum(out[f"barcodes_{c}"] for c in qc.COMPOSITIONS) == 10


def test_stage_counts_frame_total():
    a, b = qc.new_counts("w1"), qc.new_counts("w2")
    a["input_reads"], b["input_reads"], b["emitted_pairs"] = 3, 4, 2
    df = qc.stage_counts_frame([a, b])
    assert df["well"].to_list() == ["w1", "w2", "TOTAL"]
    tot = df.filter(pl.col("well") == "TOTAL").row(0, named=True)
    assert tot["input_reads"] == 7 and tot["emitted_pairs"] == 2
    assert df.columns == ["well", *qc.STAGE_FIELDS]


# ---------------------------------------------------------------- abstar failures


def test_annotation_failure_rows_and_table():
    ann = qc.annotation_frame(_ann_seqs())
    rec = [{"sequence_id": "D_contig-0", "stage": "annotation", "category": "internal_error",
            "exception_type": "ValueError", "message": "CDR3 coordinates fall outside"}]
    rows = qc.annotation_failure_rows("w1", ann, rec)
    got = {(r["kind"], r["mode"]): r["count"] for r in rows}
    assert got[("record_error", "annotation/internal_error ValueError: CDR3 coordinates fall outside")] == 1
    assert got[("unassigned", "no compatible V gene assignment")] == 1
    assert got[("unproductive", "junction does not end with conserved F")] == 1
    assert got[("unproductive", "V/J junction is out of frame")] == 1
    assert got[("unproductive", "unspecified")] == 1
    assert got[("no_junction", "junction is null")] == 1
    table = qc.failure_table(rows + qc.annotation_failure_rows("w2", ann, []))
    tot = table.filter((pl.col("well") == "TOTAL") & (pl.col("kind") == "unassigned"))
    assert tot["count"].to_list() == [2]
    assert qc.failure_table([]).height == 0


def test_capture_abstar_failures_reads_tsv(tmp_path):
    tsv = tmp_path / "failures.tsv"
    tsv.write_text(
        "input_file\tsample\trow_id\tsequence_id\tstage\tcategory\texception_type\tmessage\ttraceback_location\tdiagnostic_path\n"
        "f\ts\t1\tX_contig-0\tannotation\tinternal_error\tValueError\tboom\tloc\tp\n"
    )
    with warnings.catch_warnings(record=True) as outer:
        warnings.simplefilter("always")
        with qc.capture_abstar_failures() as fails:
            warnings.warn(f"1 sequences failed annotation; see {tsv}", RuntimeWarning)
    assert fails == [{"sequence_id": "X_contig-0", "stage": "annotation", "category": "internal_error",
                      "exception_type": "ValueError", "message": "boom"}]
    assert any("failed annotation" in str(w.message) for w in outer)  # re-emitted


# ---------------------------------------------------------------- fragments


def test_fragment_counts_by_locus():
    ann = qc.annotation_frame(_ann_seqs())
    frag = qc.fragment_counts(ann, n_record_failures=1)
    assert frag["unknown"] == {"n": 2, "truncated": 2, "incomplete_vdj": 2}
    assert frag["IGH"] == {"n": 2, "truncated": 1, "incomplete_vdj": 1}
    assert frag["IGK"] == {"n": 1, "truncated": 0, "incomplete_vdj": 1}
    acc = qc.merge_fragment_counts({}, frag)
    acc = qc.merge_fragment_counts(acc, frag)
    summ = qc.fragment_summary(acc)
    assert summ["passing_consensus"] == 10
    assert summ["fragment_fraction"] == pytest.approx(6 / 10)
    assert summ["by_locus"]["IGH"]["fragment_fraction"] == pytest.approx(0.5)


# ---------------------------------------------------------------- UMI signature


def test_hamming1_pairs_exact():
    u = pl.DataFrame({
        "barcode": ["A", "A", "A", "A", "B", "B"],
        "umi": ["AAAA", "AAAT", "AATT", "CCCC", "AAAA", "GGGG"],
    })
    pairs, with_nb = qc._hamming1_pairs(u)
    # A: AAAA-AAAT, AAAT-AATT; cross-barcode AAAA (B) must not count
    assert pairs == 2 and with_nb == 3


def test_umi_hamming1_stats_excess_over_null():
    import random

    rng = random.Random(1)
    rows = []
    for b in range(200):
        bc = f"BC{b:04d}"
        for _ in range(5):
            umi = "".join(rng.choice("ACGT") for _ in range(10))
            rows.append((bc, umi))
            err = umi[:3] + ("A" if umi[3] != "A" else "C") + umi[4:]
            rows.append((bc, err))  # one error copy per molecule
    parsed = pl.DataFrame(rows, schema=["barcode", "umi"], orient="row")
    st = qc.umi_hamming1_stats(parsed)
    assert st["h1_pairs"] >= 1000 and st["h1_pairs_null"] < 50
    summ = qc.umi_summary([st])
    assert summ["excess_error_like_umi_fraction"] > 0.9
    sub = qc.umi_hamming1_stats(parsed, max_umis=500)
    assert sub["sampled"] and sub["umis"] < parsed.height


# ---------------------------------------------------------------- run-level metrics


def test_ambient_proxy():
    rows = [("IGH", "J1", f"B{i}") for i in range(10)] + [("IGH", "J2", "B0"), ("IGH", "J2", "B1"),
                                                          ("IGH", "J3", "B5"), ("IGK", "J1", "B9")]
    df = pl.DataFrame(rows, schema=["locus", "junction", "barcode"], orient="row")
    out = qc.ambient_proxy(df)
    assert out["heavy_junctions"] == 3
    assert out["share_ge2_barcodes"] == pytest.approx(2 / 3, abs=1e-6)
    assert out["share_ge10_barcodes"] == pytest.approx(1 / 3, abs=1e-6)


def test_clonal_coherence_and_chance():
    df = pl.DataFrame({
        "v_gene:0": ["H1", "H1", "H1", "H2", "H2", "H3"],
        "cdr3_aa:0": ["C1", "C1", "C1", "C2", "C2", "C3"],
        "v_gene:1": ["IGKV1-39", "IGKV1D-39", "IGKV3-20", "IGLV2-14", "IGLV2-14", "IGKV3-20"],
    })
    out = qc.clonal_coherence(df)
    # clone H1/C1: KV1-39 x2 (D- normalised) + KV3-20 -> top 2/3; clone H2/C2: 2/2
    assert out["clones_ge2"] == 2 and out["cells_in_clones"] == 5
    assert out["purity"] == pytest.approx(4 / 5)
    assert out["purity_mean_clone"] == pytest.approx((2 / 3 + 1) / 2, abs=1e-4)
    # light V frequencies 2/6, 2/6, 2/6
    assert out["chance"] == pytest.approx(1 / 3, abs=1e-4)
    assert qc.clonal_coherence(pl.DataFrame())["purity"] is None


def test_confidence_distribution():
    assert qc.confidence_distribution(None) is None
    out = qc.confidence_distribution(pl.Series([0.05, 0.5, 0.95, 1.0, None]))
    assert out["n"] == 4
    assert out["bins"] == {"<0.1": 1, "0.1-0.3": 0, "0.3-0.6": 1, "0.6-0.9": 0, ">=0.9": 2}


def _write_well(ann_dir, meta_dir, well, rows):
    """rows: (barcode, locus, junction, umis)"""
    name = f"well{well:03d}.fastq"
    df = pl.DataFrame({
        "sequence_id": [f"{b}_contig-{i}" for i, (b, *_r) in enumerate(rows)],
        "locus": [r[1] for r in rows],
        "productive": [True] * len(rows),
        "v_gene": ["IGHV1-2" if r[1] == "IGH" else "IGKV1-39" for r in rows],
        "j_gene": ["IGHJ4" if r[1] == "IGH" else "IGKJ1" for r in rows],
        "junction": [r[2] for r in rows],
        "cdr3_aa": ["CARW" for _ in rows],
        "sequence": ["ACGT" for _ in rows],
    })
    df.write_parquet(ann_dir / f"{name}_unpaired.parquet")
    # pooled_pairing skips files < 100 bytes, so carry a realistic consensus column
    pl.DataFrame({"name": df["sequence_id"], "umis": [r[3] for r in rows],
                  "consensus": ["ACGT" * 30 for _ in rows]}).write_csv(meta_dir / f"{name}.csv")
    return name


def test_leakage_by_chain_needs_8_wells(tmp_path):
    ann, meta = tmp_path / "annotated", tmp_path / "metadata"
    ann.mkdir()
    meta.mkdir()
    # heavy species: 9 UMIs at home, 1 elsewhere; light: 5 at home, 5 spread
    for w in range(8):
        rows = [(f"B{w}", "IGH", f"TGTH{w}", 9), (f"B{w}", "IGK", f"TGTK{w}", 5)]
        other = (w + 1) % 8
        rows += [(f"B{other}", "IGH", f"TGTH{other}", 1), (f"B{other}", "IGK", f"TGTK{other}", 5)]
        _write_well(ann, meta, w, rows)
    out = qc.leakage_by_chain(tmp_path)
    assert out["wells"] == 8
    assert out["heavy"] == pytest.approx(0.1 / (1 - 1 / 8), abs=1e-3)
    assert out["light"] == pytest.approx(0.5 / (1 - 1 / 8), abs=1e-3)
    few = qc.leakage_by_chain(tmp_path, min_wells=9)
    assert "skipped" in few


def test_write_qc_end_to_end(tmp_path):
    ann, meta = tmp_path / "annotated", tmp_path / "metadata"
    ann.mkdir()
    meta.mkdir()
    name = _write_well(ann, meta, 0, [("B0", "IGH", "TGTA", 3), ("B1", "IGH", "TGTA", 2)])
    pl.DataFrame({"name": ["B0"], "v_gene:0": ["IGHV1-2"], "cdr3_aa:0": ["CARW"], "v_gene:1": ["IGKV1-39"],
                  "confidence": [0.95]}).write_parquet(ann / f"{name}_paired.parquet")
    row = qc.new_counts(name)
    row["input_reads"] = 10
    rep = qc.write_qc(tmp_path, [row], [], {}, [], [name])
    for f in ("stage_counts.csv", "annotation_failures.csv", "report.json", "report.md"):
        assert (tmp_path / "qc" / f).exists()
    assert json.loads((tmp_path / "qc" / "report.json").read_text())["stage_totals"]["input_reads"] == 10
    assert rep["ambient_proxy"]["share_ge2_barcodes"] == 1.0
    assert rep["pair_confidence"]["n"] == 1
    assert "memory" in (tmp_path / "qc" / "report.md").read_text()


# ---------------------------------------------------------------- support columns


def test_add_support_columns():
    seqs = [_seq("AAAC_contig-0"), _seq("AAAC_contig-1"), _seq("GGGT_contig-0")]
    meta = pl.DataFrame({"name": ["AAAC_contig-0", "AAAC_contig-1"], "reads": [10, 3], "umis": [4, 1],
                         "cluster_fraction": [0.77, 0.23], "consensus": ["A", "C"], "pass_filters": [True, True]})
    qc.add_support_columns(seqs, meta)
    a = seqs[0].annotations
    assert (a["umis"], a["reads"], a["cluster_fraction"]) == (4, 10, 0.77)
    assert (a["barcode"], a["contig"]) == ("AAAC", "contig-0")
    assert "umi" not in a
    assert seqs[2]["umis"] is None and seqs[2]["barcode"] == "GGGT"


def test_support_columns_reach_paired_output():
    import abutils

    seqs = [_seq("AAAC_contig-0", locus="IGH"), _seq("AAAC_contig-1", locus="IGK")]
    meta = pl.DataFrame({"name": ["AAAC_contig-0", "AAAC_contig-1"], "reads": [10, 3], "umis": [4, 1],
                         "cluster_fraction": [0.77, 0.23]})
    qc.add_support_columns(seqs, meta)
    pairs = abutils.tl.assign_pairs(seqs, delim="_", delim_occurance=-1)
    df = abutils.io.to_polars(pairs) if hasattr(abutils.io, "to_polars") else None
    if df is None:
        pytest.skip("abutils.io.to_polars unavailable")
    assert df["umis:0"].to_list() == [4] and df["umis:1"].to_list() == [1]
    assert "umi:0" not in df.columns
