import os
from pathlib import Path

import abutils
import polars as pl
import pytest

from pairplex.pairing import (
    Chain,
    contract_frame,
    exact_pairs,
    pair_well,
    resolve_barcode,
    scored_pairs,
    well_number,
    write_contract_pairs,
)


# ---------------------------------------------------------------- resolve_barcode


def test_1h1l_strong_support_scores_high():
    r = resolve_barcode([Chain("bc_h", 30, 300)], [Chain("bc_l", 40, 400)])
    assert r.reason == "paired" and (r.heavy.name, r.light.name) == ("bc_h", "bc_l")
    assert r.confidence == pytest.approx(30 / 31 * 40 / 41)
    assert r.confidence > 0.9
    assert r.margin == pytest.approx(min(30 / 31, 40 / 41))


def test_1h1l_single_umi_scores_low():
    r = resolve_barcode([Chain("h", 1)], [Chain("l", 1)])
    assert r.reason == "paired" and r.confidence == pytest.approx(0.25)


def test_1h2l_dominant_light_is_resolved():
    r = resolve_barcode([Chain("h", 10)], [Chain("l1", 2), Chain("l2", 8)])
    assert r.reason == "paired" and r.light.name == "l2"
    assert r.confidence == pytest.approx(10 / 11 * 8 / 11)
    assert r.margin == pytest.approx(min(10 / 11, 6 / 11))
    # lower than the same barcode without the competing light
    alone = resolve_barcode([Chain("h", 10)], [Chain("l2", 8)])
    assert r.confidence < alone.confidence


def test_ambiguous_2h2l_not_emitted():
    r = resolve_barcode([Chain("h1", 5), Chain("h2", 4)], [Chain("l1", 6), Chain("l2", 5)])
    assert r.heavy is None and r.confidence is None and r.reason == "both_ambiguous"
    r = resolve_barcode([Chain("h1", 10), Chain("h2", 1)], [Chain("l1", 6), Chain("l2", 5)])
    assert r.heavy is None and r.reason == "light_ambiguous"


def test_kappa_and_lambda_compete_in_one_group():
    lights = [Chain("k", 9), Chain("lam", 3)]
    r = resolve_barcode([Chain("h", 5)], lights)
    assert r.light.name == "k"
    assert resolve_barcode([Chain("h", 5)], [Chain("k", 4), Chain("lam", 3)]).heavy is None


def test_threshold_boundary_and_tie_break_by_reads():
    assert resolve_barcode([Chain("h", 4)], [Chain("l1", 4), Chain("l2", 2)]).reason == "paired"
    assert resolve_barcode([Chain("h", 4)], [Chain("l1", 4), Chain("l2", 3)]).reason == "light_ambiguous"
    r = resolve_barcode([Chain("h", 4)], [Chain("l1", 3, 10), Chain("l2", 3, 50)], min_dominance=1.0)
    assert r.light.name == "l2" and r.margin == pytest.approx(0.0)


def test_confidence_is_monotonic():
    def conf(h1, h2, l1):
        return resolve_barcode([Chain("a", h1)] + ([Chain("b", h2)] if h2 else []), [Chain("l", l1)],
                               min_dominance=1.0).confidence
    assert conf(5, 0, 5) < conf(6, 0, 5) < conf(6, 0, 6)
    assert conf(6, 1, 5) < conf(6, 0, 5)
    assert conf(6, 3, 5) < conf(6, 1, 5)
    for h1, h2, l1 in [(1, 0, 1), (100, 0, 100), (3, 3, 2)]:
        assert 0.0 <= conf(h1, h2, l1) <= 1.0


def test_missing_locus_group():
    assert resolve_barcode([], [Chain("l", 3)]).reason == "no_heavy"
    assert resolve_barcode([Chain("h", 3)], []).reason == "no_light"


# ---------------------------------------------------------------- well-level


def _seq(bc, n, locus, productive=True, cdr3=None):
    return abutils.Sequence({"sequence_id": f"{bc}_contig-{n}", "sequence": "ACGT" * 10, "locus": locus,
                             "productive": productive, "v_gene": f"{locus}V1", "j_gene": f"{locus}J1",
                             "v_call": f"{locus}V1*01", "junction": "TGT" + "A" * n,
                             "cdr3_aa": cdr3 or "CAR" + "W" * n})


def _well():
    seqs = [
        _seq("AAA", 0, "IGH"), _seq("AAA", 1, "IGK"),  # 1H+1L
        _seq("BBB", 0, "IGH"), _seq("BBB", 1, "IGK"), _seq("BBB", 2, "IGL"),  # 1H+2L, lambda dominant
        _seq("CCC", 0, "IGH"), _seq("CCC", 1, "IGH"), _seq("CCC", 2, "IGK"), _seq("CCC", 3, "IGK"),  # ambiguous
        _seq("DDD", 0, "IGH"), _seq("DDD", 1, "IGH", productive=False), _seq("DDD", 2, "IGL"),  # non-productive extra H
    ]
    support = {"AAA_contig-0": (5, 50), "AAA_contig-1": (6, 60),
               "BBB_contig-0": (4, 40), "BBB_contig-1": (1, 10), "BBB_contig-2": (7, 70),
               "CCC_contig-0": (3, 30), "CCC_contig-1": (3, 31), "CCC_contig-2": (2, 20), "CCC_contig-3": (2, 21),
               "DDD_contig-0": (2, 20), "DDD_contig-1": (9, 90), "DDD_contig-2": (2, 20)}
    return seqs, support


def test_scored_pairs_on_a_well():
    seqs, support = _well()
    pairs, reasons = scored_pairs(seqs, support)
    got = {p.name: (p.heavy["sequence_id"], p.light["sequence_id"]) for p in pairs}
    assert got == {"AAA": ("AAA_contig-0", "AAA_contig-1"), "BBB": ("BBB_contig-0", "BBB_contig-2"),
                   "DDD": ("DDD_contig-0", "DDD_contig-2")}
    assert reasons == {"paired": 3, "both_ambiguous": 1}
    bbb = next(p for p in pairs if p.name == "BBB")
    assert bbb.light["umis"] == bbb.light["chain_umis"] == 7 and bbb.confidence == pytest.approx(4 / 5 * 7 / 9)
    # the legacy rule keeps only the strict 1H+1L barcode
    assert {p.name for p in exact_pairs(seqs)} == {"AAA"}


def test_clonal_relatives_do_not_compete():
    # 2H+1L where both heavies are the same clonotype (split consensus / clonal relatives)
    seqs = [_seq("EEE", 0, "IGH", cdr3="CARDY"), _seq("EEE", 1, "IGH", cdr3="CARDY"), _seq("EEE", 2, "IGK")]
    support = {"EEE_contig-0": (3, 30), "EEE_contig-1": (3, 40), "EEE_contig-2": (5, 50)}
    pairs, reasons = scored_pairs(seqs, support)
    assert reasons == {"paired": 1}
    p = pairs[0]
    assert p.heavy["sequence_id"] == "EEE_contig-1"
    # umis = the emitted contig's own count; chain_umis = ranking support summed over merged contigs
    assert p.heavy["umis"] == 3 and p.heavy["chain_umis"] == 6
    assert p.confidence == pytest.approx(6 / 7 * 5 / 6)
    # without collapsing, the even split is ambiguous and the lineage is lost
    _, reasons = scored_pairs(seqs, support, collapse_clonal=False)
    assert reasons == {"heavy_ambiguous": 1}


def test_different_clonotypes_still_compete():
    seqs = [_seq("FFF", 0, "IGH", cdr3="CARDY"), _seq("FFF", 1, "IGH", cdr3="CTTGG"), _seq("FFF", 2, "IGK")]
    support = {"FFF_contig-0": (3, 30), "FFF_contig-1": (3, 40), "FFF_contig-2": (5, 50)}
    assert scored_pairs(seqs, support)[1] == {"heavy_ambiguous": 1}


# ---------------------------------------------------------------- non-productive chains


def _np(bc, n, locus, cdr3=None, v=True):
    s = _seq(bc, n, locus, productive=False)
    s["cdr3_aa"] = cdr3
    if not v:
        s["v_gene"] = None
    return s


def test_nonproductive_only_barcode_is_paired():
    # heavy group has only a non-productive chain (no CDR3 aa), light is productive
    seqs = [_np("NNN", 0, "IGH"), _seq("NNN", 1, "IGK")]
    support = {"NNN_contig-0": (4, 40), "NNN_contig-1": (5, 50)}
    pairs, reasons = scored_pairs(seqs, support)
    assert reasons == {"paired": 1}
    assert (pairs[0].heavy["sequence_id"], pairs[0].light["sequence_id"]) == ("NNN_contig-0", "NNN_contig-1")
    assert pairs[0].confidence == pytest.approx(4 / 5 * 5 / 6)
    # both groups non-productive also pair
    seqs = [_np("MMM", 0, "IGH"), _np("MMM", 1, "IGL", cdr3="CQQ*Y")]
    assert scored_pairs(seqs, {})[1] == {"paired": 1}


def test_productive_heavy_beats_nonproductive_without_ambiguity():
    # the non-productive heavy has more UMIs, but productive chains are the only competitors
    seqs = [_seq("PPP", 0, "IGH"), _np("PPP", 1, "IGH", cdr3="CAR*W"), _seq("PPP", 2, "IGK")]
    support = {"PPP_contig-0": (3, 30), "PPP_contig-1": (8, 80), "PPP_contig-2": (5, 50)}
    pairs, reasons = scored_pairs(seqs, support)
    assert reasons == {"paired": 1}
    p = pairs[0]
    assert p.heavy["sequence_id"] == "PPP_contig-0" and p.heavy["chain_umis"] == 3
    assert p.confidence == pytest.approx(3 / 4 * 5 / 6)  # the non-productive UMIs are not counted


def test_null_cdr3_or_v_chains_are_not_merged():
    # two non-productive heavies with null CDR3 aa: they compete (even split -> ambiguous), never merge
    seqs = [_np("QQQ", 0, "IGH"), _np("QQQ", 1, "IGH"), _seq("QQQ", 2, "IGK")]
    support = {"QQQ_contig-0": (3, 30), "QQQ_contig-1": (3, 31), "QQQ_contig-2": (5, 50)}
    assert scored_pairs(seqs, support)[1] == {"heavy_ambiguous": 1}
    # same CDR3 aa but a null V gene: not merged either
    seqs = [_np("RRR", 0, "IGH", cdr3="CARDY", v=False), _np("RRR", 1, "IGH", cdr3="CARDY", v=False),
            _seq("RRR", 2, "IGK")]
    support = {"RRR_contig-0": (3, 30), "RRR_contig-1": (3, 31), "RRR_contig-2": (5, 50)}
    assert scored_pairs(seqs, support)[1] == {"heavy_ambiguous": 1}
    # identical non-null V, J, CDR3 aa: merged (non-productive chains collapse like productive ones)
    seqs = [_np("SSS", 0, "IGH", cdr3="CAR*Y"), _np("SSS", 1, "IGH", cdr3="CAR*Y"), _seq("SSS", 2, "IGK")]
    support = {"SSS_contig-0": (3, 30), "SSS_contig-1": (3, 31), "SSS_contig-2": (5, 50)}
    pairs, reasons = scored_pairs(seqs, support)
    assert reasons == {"paired": 1} and pairs[0].heavy["chain_umis"] == 6


def test_productive_only_restores_previous_behaviour():
    seqs, support = _well()
    seqs += [_np("NNN", 0, "IGH"), _seq("NNN", 1, "IGK"),
             _seq("PPP", 0, "IGH"), _np("PPP", 1, "IGH", cdr3="CAR*W"), _seq("PPP", 2, "IGK")]
    support.update({"NNN_contig-0": (4, 40), "NNN_contig-1": (5, 50),
                    "PPP_contig-0": (3, 30), "PPP_contig-1": (8, 80), "PPP_contig-2": (5, 50)})
    old, old_reasons = scored_pairs(seqs, support, productive_only=True)
    new, new_reasons = scored_pairs(seqs, support)
    key = lambda ps: {p.name: (p.heavy["sequence_id"], p.light["sequence_id"], p.confidence) for p in ps}
    # productive_only: NNN has no heavy; PPP resolved on its productive heavy only
    assert old_reasons == {"paired": 4, "both_ambiguous": 1, "no_heavy": 1}
    assert set(key(old)) == {"AAA", "BBB", "DDD", "PPP"}
    # default: NNN recovered, every productive-only pair unchanged
    assert set(key(new)) == set(key(old)) | {"NNN"}
    assert {k: v for k, v in key(new).items() if k != "NNN"} == key(old)


def test_productive_flag_in_paired_outputs_and_contract(tmp_path):
    seqs = [_np("NNN", 0, "IGH"), _seq("NNN", 1, "IGK"), _seq("AAA", 0, "IGH"), _seq("AAA", 1, "IGK")]
    pq = tmp_path / "well003_paired.parquet"
    pair_well(seqs, tmp_path / "well003_paired.tsv", pq, mode="scored", support={})
    df = pl.read_parquet(pq)
    assert {"productive:0", "productive:1"} <= set(df.columns)
    out = tmp_path / "pairs.parquet"
    write_contract_pairs([contract_frame(pq, 3)], out)
    p = pl.read_parquet(out).sort("name")
    assert p.schema["productive:0"] == pl.Boolean
    assert p["productive:0"].to_list() == [True, False] and p["productive:1"].to_list() == [True, True]
    write_contract_pairs([], tmp_path / "e.parquet")
    assert pl.read_parquet(tmp_path / "e.parquet").schema["productive:1"] == pl.Boolean


def test_pair_well_writes_outputs_and_contract(tmp_path):
    seqs, support = _well()
    tsv, pq = tmp_path / "well007_paired.tsv", tmp_path / "well007_paired.parquet"
    pairs = pair_well(seqs, tsv, pq, mode="scored", support=support)
    df = pl.read_parquet(pq)
    assert df.height == len(pairs) == 3
    for c in ("confidence", "margin", "umis:0", "umis:1", "sequence_id:0", "sequence_id:1"):
        assert c in df.columns
    assert "confidence" in pl.read_csv(tsv, separator="\t").columns
    cf = contract_frame(pq, well_number("well007.fastq"))
    out = tmp_path / "pairs" / "pairs.parquet"
    assert write_contract_pairs([cf, None], out) == 3
    p = pl.read_parquet(out)
    for c in ("barcode", "well", "name", "confidence") + tuple(
            f"{c}:{s}" for s in (0, 1) for c in ("locus", "v_gene", "j_gene", "junction", "cdr3_aa",
                                                 "productive", "sequence_id", "sequence", "umis", "chain_umis")):
        assert c in p.columns
    assert set(p["well"]) == {7} and p["confidence"].is_between(0, 1).all()
    assert set(p["locus:0"]) == {"IGH"}


def test_empty_contract_has_schema(tmp_path):
    out = tmp_path / "pairs.parquet"
    assert write_contract_pairs([], out) == 0
    assert {"well", "name", "confidence", "sequence_id:0", "sequence:1"} <= set(pl.read_parquet(out).columns)


def test_exact_mode_matches_legacy_columns(tmp_path):
    seqs, support = _well()
    pq = tmp_path / "w_paired.parquet"
    pair_well(seqs, tmp_path / "w_paired.tsv", pq, mode="exact", support=support)
    df = pl.read_parquet(pq)
    assert df["name"].to_list() == ["AAA"] and "confidence" not in df.columns


# ---------------------------------------------------------------- end-to-end smoke (opt-in)

SMOKE = os.environ.get("PAIRPLEX_SMOKE_READS")


@pytest.mark.skipif(not SMOKE, reason="set PAIRPLEX_SMOKE_READS to a directory of per-well FASTQs")
def test_run_smoke_scored_vs_legacy(tmp_path):
    from pairplex.pairplex import run

    run(SMOKE, tmp_path / "legacy", legacy=True, quiet=True)
    run(SMOKE, tmp_path / "scored", quiet=True)
    assert not (tmp_path / "legacy" / "pairs").exists()
    pairs = pl.read_parquet(tmp_path / "scored" / "pairs" / "pairs.parquet")
    n_scored = n_legacy = 0
    for f in sorted((tmp_path / "scored" / "annotated").glob("*_paired.parquet")):
        df = pl.read_parquet(f)
        n_scored += df.height
        assert {"confidence", "margin", "umis:0", "umis:1"} <= set(df.columns)
        leg = Path(str(f).replace("/scored/", "/legacy/"))
        n_legacy += pl.read_parquet(leg).height
        assert "confidence" not in pl.read_parquet(leg).columns
    assert pairs.height == n_scored > 0 and n_legacy > 0
