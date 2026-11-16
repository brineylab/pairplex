import polars as pl
import pytest

from pairplex.pooled_pairing import estimate_leakage, home_posteriors, pair_pooled, SPECIES


def _occ(rows):
    base = dict(v_gene="V", j_gene="J", cdr3_aa="CAR", sequence="ACGT")
    return pl.DataFrame([{**base, **r} for r in rows])


def _c(bc, well, locus, junc, umis, n):
    return dict(barcode=bc, well=well, locus=locus, junction=junc, umis=umis, sequence_id=f"{bc}_contig-{n}")


def test_posterior_closed_form():
    sw = pl.DataFrame({"barcode": ["A"] * 2, "locus": ["IGH"] * 2, "v_gene": ["V"] * 2, "j_gene": ["J"] * 2,
                       "junction": ["X"] * 2, "well": [1, 2], "n": [3, 1]})
    W, f = 10, 0.8
    out = home_posteriors(sw, W, f).sort("well")
    r = (1 - f + f / W) / (f / W)
    z = r ** 3 + r + (W - 2)
    assert out["p_home"].to_list() == pytest.approx([r ** 3 / z, r / z])


def test_pairs_follow_shared_home_well_not_well_cooccurrence():
    # barcode A: cell 1 (H1, L1) lives in well 1; cell 2 (H2, L2) in well 2; each leaks a molecule elsewhere
    rows = [
        _c("A", 1, "IGH", "H1", 4, 0), _c("A", 1, "IGK", "L1", 5, 1), _c("A", 2, "IGH", "H1", 1, 2),
        _c("A", 2, "IGH", "H2", 4, 3), _c("A", 2, "IGK", "L2", 4, 4), _c("A", 1, "IGK", "L2", 1, 5),
    ]
    pairs, info = pair_pooled(_occ(rows), n_wells=10, leakage=0.8)
    got = {(r["junction:0"], r["junction:1"], r["well"]) for r in pairs.to_dicts()}
    assert got == {("H1", "L1", 1), ("H2", "L2", 2)}
    assert (pairs["confidence"] > 0.5).all() and (pairs["margin"] > 0).all()


def test_single_molecule_chains_get_low_confidence():
    rows = [_c("B", 3, "IGH", "H", 1, 0), _c("B", 3, "IGL", "L", 1, 1)]
    pairs, _ = pair_pooled(_occ(rows), n_wells=76, leakage=0.8)
    assert pairs.height == 1 and pairs["confidence"][0] < 0.1


def test_leakage_estimate():
    # one species, 10 UMIs: 3 home + 7 spread over 7 wells -> outside 0.7 -> f = 0.7/(1-1/10)
    sw = pl.DataFrame({"barcode": ["A"] * 8, "locus": ["IGH"] * 8, "v_gene": ["V"] * 8, "j_gene": ["J"] * 8,
                       "junction": ["X"] * 8, "well": list(range(8)), "n": [3] + [1] * 7})
    assert estimate_leakage(sw, 10) == pytest.approx(0.7 / 0.9)
