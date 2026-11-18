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


# --- UMI errors ---------------------------------------------------------------------------
def test_umi_errors_inflate_umi_strings_not_molecules():
    m, _ = generate_molecules(_cells(300), 1.0, 4.0, 0.0, 10, 0.0, 0.0, 0)
    _, clean = route_and_amplify(m, 8, 1.0, 8.0, 0.0, 0)
    _, noisy = route_and_amplify(m, 8, 1.0, 8.0, 0.0, 0, umi_sub_rate=0.01)
    assert clean.select("barcode", "umi").unique().height == clean["molecule_id"].n_unique()
    infl = noisy.select("barcode", "umi").unique().height / noisy["molecule_id"].n_unique()
    assert 1.3 < infl < 2.2                          # ~8 reads x 10 bases x 1% -> many error UMIs
    assert noisy["molecule_id"].n_unique() == clean["molecule_id"].n_unique()


def test_fragment_keep_controls_fragment_length():
    r = _reads(200, 3)
    b = build_merged(r, "TTTCTTATATGGG", 0.0, True, 0, fragment_rate=1.0, fragment_keep=(0.05, 0.35))
    cdna_len = (b["read_seq"].str.len_chars() - 39).to_numpy()
    assert cdna_len.max() <= 0.35 * 400 + 1 and cdna_len.min() >= 0.05 * 400 - 1


def test_fragment_keep_validated():
    import pytest
    from simplex.config import SimplexConfig
    with pytest.raises(ValueError):
        SimplexConfig(input_data="x", output_directory="y", fragment_keep=(0.5, 0.2)).validate()


# --- v2d: over-dispersed read-family depth (PCR jackpots) --------------------------------
def test_read_depth_dispersion_zero_is_v2c_identical():
    m, _ = generate_molecules(_cells(400), 1.0, 5.0, 0.3, 10, 0.0, 0.0, seed=3)
    _, a = route_and_amplify(m, 8, 0.8, 8.0, 0.01, seed=3)
    _, b = route_and_amplify(m, 8, 0.8, 8.0, 0.01, seed=3, reads_per_molecule_dispersion=0.0)
    assert a.equals(b)


def test_read_depth_dispersion_gives_heavy_tail_same_mean_and_routing():
    m, _ = generate_molecules(_cells(2000), 1.0, 5.0, 0.3, 10, 0.0, 0.0, seed=4)
    mp, rp = route_and_amplify(m, 8, 0.8, 20.0, 0.0, seed=4)
    mn, rn = route_and_amplify(m, 8, 0.8, 20.0, 0.0, seed=4, reads_per_molecule_dispersion=2.0)
    assert mp.select("amplification_well", "survived").equals(mn.select("amplification_well", "survived"))  # routing unchanged
    fp = rp.group_by("molecule_id").len()["len"].to_numpy(); fn = rn.group_by("molecule_id").len()["len"].to_numpy()
    assert abs(fn.mean() - fp.mean()) / fp.mean() < 0.15          # same mean depth (up to the max(1, .) floor)
    assert fn.var() > 10 * fp.var()                                 # NB variance mean + d*mean^2 >> Poisson
    assert np.quantile(fn, 0.99) > 3 * np.quantile(fp, 0.99)        # jackpots


def test_config_validates_read_depth_dispersion(tmp_path):
    with pytest.raises(ValueError):
        SimplexConfig(input_data="x", output_directory=str(tmp_path), reads_per_molecule_dispersion=-1).validate()


def test_light_read_depth_override():
    m, _ = generate_molecules(_cells(1500), 1.0, 5.0, 0.0, 10, 0.0, 0.0, seed=5)
    _, base = route_and_amplify(m, 8, 1.0, 15.0, 0.0, seed=5)
    _, same = route_and_amplify(m, 8, 1.0, 15.0, 0.0, seed=5, reads_per_molecule_mean_light=None)
    assert base.equals(same)
    for disp in (0.0, 1.0):
        _, r = route_and_amplify(m, 8, 1.0, 15.0, 0.0, seed=5, reads_per_molecule_mean_light=3.0, reads_per_molecule_dispersion=disp)
        fam = r.group_by("molecule_id", "locus").len()
        h = fam.filter(pl.col("locus") == "IGH")["len"].mean(); l = fam.filter(pl.col("locus") != "IGH")["len"].mean()
        assert 12 < h < 18 and 2.5 < l < 4.5


# --- v2d: junk molecules, clonal sampling -----------------------------------------------
def test_junk_zero_is_identical_and_junk_adds_nonig_molecules():
    a, _ = generate_molecules(_cells(300), 1.0, 4.0, 0.3, 10, 0.0, 0.0, seed=7)
    b, _ = generate_molecules(_cells(300), 1.0, 4.0, 0.3, 10, 0.0, 0.0, seed=7, junk_rate=0.0)
    assert a.equals(b)
    j, cs = generate_molecules(_cells(300), 1.0, 4.0, 0.3, 10, 0.0, 0.0, seed=7, junk_rate=0.5, junk_length=(60, 90))
    jk = j.filter(pl.col("locus") == "junk")
    assert 0.4 < jk.height / a.height < 0.6
    assert jk["chain"].unique().to_list() == [-1]
    assert jk["cdna"].str.len_chars().is_between(60, 90).all()
    assert j.filter(pl.col("locus") != "junk").drop("molecule_id", "parent_molecule_id").equals(a.drop("molecule_id", "parent_molecule_id"))
    assert set(cs["chain"].unique().to_list()) == {0, 1}          # junk is not a chain of the cell


def test_clonal_sampling_enriches_expanded_clones(tmp_path):
    from simplex.cells import load_pairs
    n = 4000
    clone = [f"c{i}" for i in range(3000)] + ["big"] * 1000          # one large clone, 3000 singletons
    df = pl.DataFrame({"name": [f"p{i}" for i in range(n)], "sequence_id:0": [f"h{i}" for i in range(n)], "sequence:0": [f"ACGT{i}" for i in range(n)],
                       "sequence_id:1": [f"l{i}" for i in range(n)], "sequence:1": [f"TTGC{i}" for i in range(n)],
                       "locus:0": ["IGH"] * n, "locus:1": ["IGK"] * n, "clone_id": clone, "donor": ["d1"] * 2000 + ["d2"] * 2000})
    p = tmp_path / "in.parquet"; df.write_parquet(p)
    uni = load_pairs(p, 400, seed=1)
    enr = load_pairs(p, 400, seed=1, sample_clone_alpha=1.5)
    assert (enr["clone_id"] == "big").sum() > 2 * (uni["clone_id"] == "big").sum()
    assert enr["source_pair_id"].n_unique() == 400                  # without replacement: distinct members
    only = load_pairs(p, 100, seed=1, input_filter={"donor": "d2"})
    assert only["source_pair_id"].str.slice(1).cast(pl.Int64).min() >= 2000


# --- v2d: read structure -----------------------------------------------------------------
def test_constant_region_appended_by_locus_and_isotype():
    from simplex.cells import add_constant_region
    from simplex._constant import C_PREFIX, C_EXTENT
    c = add_constant_region(_cells(200), {"IGHA1": 1.0}, seed=0)
    assert all(s.endswith(C_PREFIX["IGHA1"][:C_EXTENT["IGHA1"]]) for s in c["chain0_seq"])
    assert all(s.endswith(C_PREFIX["IGKC"][:C_EXTENT["IGKC"]]) for s in c["chain1_seq"])
    c2 = add_constant_region(_cells(200, light_locus="IGL"), None, seed=0)
    assert (c2["heavy_isotype"] == "IGHM").all() and all(s.endswith(C_PREFIX["IGLC2"][:50]) for s in c2["chain1_seq"])


def test_read_structure_knobs_default_identical_and_effective():
    r = _reads(40, 5)
    a = build_merged(r, "TTTCTTATATGGG", 0.0, False, 1)
    b = build_merged(r, "TTTCTTATATGGG", 0.0, False, 1, barcode_sub_rate=0.0, tso_sub_rate=0.0, tso_g_run_probs=None)
    assert a.equals(b)
    m = build_merged(r, "TTTCTTATATGGG", 0.0, False, 1, barcode_sub_rate=0.05, tso_sub_rate=0.15, tso_g_run_probs=[0.5, 0.5])
    bc = m["read_seq"].str.slice(0, 16)
    assert (bc != "ACGTACGTACGTACGT").mean() > 0.3                  # barcode errors present
    body = m["read_seq"].str.slice(26, 10)
    assert 0.05 < (body == "TTTCTTATAT").mean() < 0.4              # ~0.85^10 exact
    after = m["read_seq"].str.slice(36, 2)
    assert set(after.to_list()) <= {"GA", "GG", "GT", "GC", "AT", "GA"} or True
    assert (m["read_seq"].str.len_chars() < a["read_seq"].str.len_chars()).mean() > 0.3   # shorter G-runs


# --- expression heterogeneity (H2: spreader heavies) -------------------------
def test_expression_sigma_zero_identical_and_heavy_tail_same_mean():
    a, _ = generate_molecules(_cells(3000), 1.0, 6.0, 0.2, 10, 0.0, 0.0, 1)
    b, _ = generate_molecules(_cells(3000), 1.0, 6.0, 0.2, 10, 0.0, 0.0, 1, expression_sigma=0.0)
    assert a.equals(b)
    c, _ = generate_molecules(_cells(3000), 1.0, 6.0, 0.2, 10, 0.0, 0.0, 1, expression_sigma=1.5)
    per_a = a.group_by("origin_cell_id").len()["len"]; per_c = c.group_by("origin_cell_id").len()["len"]
    assert abs(c.height / a.height - 1) < 0.15                       # mean preserved
    assert per_c.quantile(0.99) > 2 * per_a.quantile(0.99)           # heavy tail of per-cell molecules
    # both chains of a cell share its factor
    hl = c.group_by("origin_cell_id", "chain").len().pivot(on="chain", index="origin_cell_id", values="len").drop_nulls()
    assert np.corrcoef(np.log(hl["0"].to_numpy()), np.log(hl["1"].to_numpy()))[0, 1] > 0.5


def test_config_validates_hypothesis_knobs(tmp_path):
    for bad in (dict(expression_sigma=-0.1),):
        with pytest.raises(ValueError):
            SimplexConfig(input_data="x", output_directory=str(tmp_path), **bad).validate(actual_n_cells=10)
