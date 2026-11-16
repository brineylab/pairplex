"""SimPlex v2: fixes and knobs motivated by the real-data diagnosis."""
import numpy as np
import polars as pl
import pytest

from simplex.cells import assign_droplets_and_barcodes
from simplex.config import SimplexConfig
from simplex.molecules import generate_molecules
from simplex.reads import build_merged
from simplex.routing import route_and_amplify


def _cells(n, light_locus="IGK"):
    return pl.DataFrame({"cell_id": np.arange(n), "source_pair_id": [f"s{i}" for i in range(n)],
                         "chain0_id": [f"h{i}" for i in range(n)], "chain0_seq": ["ACGT" * 80] * n, "chain0_locus": ["IGH"] * n,
                         "chain1_id": [f"l{i}" for i in range(n)], "chain1_seq": ["TTGCA" * 60] * n, "chain1_locus": [light_locus] * n,
                         "droplet_id": np.arange(n) // 2, "barcode": [f"BC{i // 2:06d}" for i in range(n)],
                         "resident_well": np.arange(n) % 8})


# --- S7: barcode_pool_size --------------------------------------------------------------
def test_barcode_pool_size_changes_reuse():
    base = pl.DataFrame({"cell_id": np.arange(4000)})
    reuse = []
    for pool in (500, 5000, 500000):
        c = assign_droplets_and_barcodes(base, 1.0, 0.0, "v2", pool, seed=0)
        per_droplet = c.select("droplet_id", "barcode").unique()
        reuse.append(1 - per_droplet["barcode"].n_unique() / per_droplet.height)
    assert reuse[0] > reuse[1] > reuse[2]          # larger pool -> less reuse (v1: identical for all pools >= K)
    assert reuse[2] < 0.01


# --- S2: truncation ---------------------------------------------------------------------
def _reads(n_mol=50, depth=6):
    rows = []
    for m in range(n_mol):
        for d in range(depth):
            rows.append({"read_id": f"r{m}_{d}", "final_well": 0, "barcode": "ACGTACGTACGTACGT",
                         "umi": f"{m:010d}".replace("0", "A").replace("1", "C").replace("2", "G").replace("3", "T")[:10],
                         "cdna": "GATTACAGGT" * 40, "n_seq_errors": 0})
    return pl.DataFrame(rows)


def test_truncation_fragment_5p_keeps_3p_end_and_most_molecules_intact():
    r = _reads(200, 3)
    full = 39 + 400
    b = build_merged(r, "TTTCTTATATGGG", 0.0, True, 0, fragment_rate=0.25)   # default fragment_5p
    b = b.join(r.select("read_id", "umi"), on="read_id")
    per = b.group_by("umi").agg(pl.col("read_seq").str.len_chars().n_unique().alias("nlen"), pl.col("read_seq").str.len_chars().first().alias("L"))
    assert (per["nlen"] == 1).all()                                   # one length per molecule
    frag = (per["L"] < full).mean()
    assert 0.15 < frag < 0.35                                         # ~fragment_rate of molecules fragmented
    cdna = "GATTACAGGT" * 40
    assert all(cdna.endswith(s[39:]) for s in b["read_seq"])          # 3' end always intact


def test_truncation_molecule_3p_shared_within_molecule_and_5p_anchored():
    r = _reads()
    b = build_merged(r, "TTTCTTATATGGG", 0.0, True, 0, truncation="molecule_3p")
    b = b.join(r.select("read_id", "umi"), on="read_id")
    per = b.group_by("umi").agg(pl.col("read_seq").str.len_chars().n_unique().alias("nlen"))
    assert (per["nlen"] == 1).all()                                   # one length per molecule
    assert b["read_seq"].str.len_chars().n_unique() > 1               # but lengths vary across molecules
    assert all(s[39:49] == ("GATTACAGGT" * 40)[:10] for s in b["read_seq"])  # 5' end intact (bc16+umi10+TSO13)


def test_truncation_legacy_still_available():
    r = _reads(5, 4)
    b = build_merged(r, "TTTCTTATATGGG", 0.0, True, 0, truncation="per_read_both")
    assert b.height == r.height
    with pytest.raises(ValueError):
        build_merged(r, "TTTCTTATATGGG", 0.0, True, 0, truncation="nope")


# --- chain-specific release / light abundance -------------------------------------------
def test_chain_specific_release_and_light_molecules():
    m, _ = generate_molecules(_cells(3000), 1.0, 4.0, 0.5, 10, 0.0, 0.0, 0,
                              release_rate_heavy=0.1, release_rate_light=0.9, molecules_per_chain_mean_light=12.0)
    h, l = m.filter(pl.col("locus") == "IGH"), m.filter(pl.col("locus") == "IGK")
    assert 0.07 < h["is_free"].mean() < 0.13 and 0.87 < l["is_free"].mean() < 0.93
    assert l.height > 2.5 * h.height


# --- ambient axis -----------------------------------------------------------------------
def test_ambient_adds_free_molecules_under_other_barcodes():
    cells = _cells(2000)
    m, _ = generate_molecules(cells, 1.0, 5.0, 0.0, 10, 0.0, 0.0, 0, ambient_rate=0.3)
    amb, res = m.filter(pl.col("is_ambient")), m.filter(~pl.col("is_ambient"))
    assert 0.25 < amb.height / res.height < 0.35
    assert amb["is_free"].all() and (~res["is_free"]).all()
    home = cells.select(pl.col("cell_id").alias("origin_cell_id"), pl.col("barcode").alias("home_bc"))
    moved = amb.join(home, on="origin_cell_id")
    assert (moved["barcode"] != moved["home_bc"]).mean() > 0.99      # copies land under other barcodes
    # ambient survives routing as free molecules in random wells, with the flag carried to reads
    _, reads = route_and_amplify(m, 8, 1.0, 2.0, 0.0, 0)
    assert reads.filter(pl.col("is_ambient")).height > 0


def test_ambient_zero_is_v1_identical():
    a, _ = generate_molecules(_cells(200), 0.7, 5.0, 0.2, 10, 0.0, 0.0, 3)
    b, _ = generate_molecules(_cells(200), 0.7, 5.0, 0.2, 10, 0.0, 0.0, 3, ambient_rate=0.0)
    assert a.drop("is_ambient").equals(b.drop("is_ambient")) and not a["is_ambient"].any()


# --- config validation ------------------------------------------------------------------
def test_config_validates_v2_knobs(tmp_path):
    ok = SimplexConfig(input_data="x", output_directory=str(tmp_path), ambient_rate=0.3, release_rate_light=0.8)
    ok.validate(actual_n_cells=10)
    for bad in (dict(ambient_rate=-1), dict(release_rate_heavy=1.5), dict(truncation="x"), dict(molecules_per_chain_mean_light=0)):
        with pytest.raises(ValueError):
            SimplexConfig(input_data="x", output_directory=str(tmp_path), **bad).validate(actual_n_cells=10)
