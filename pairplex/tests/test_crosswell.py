import gzip
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import pytest

from pairplex.crosswell import (
    ShardWriter,
    assign_molecule_wells,
    auto_shard_count,
    contig_well_counts,
    directional_collapse,
    parse_to_shards,
    process_droplet_crosswell,
    shard_of,
    well_from_path,
)
from pairplex.pooled_pairing import estimate_leakage, leakage_from_totals

SIM_READS = Path("/home/jovyan/work/PairPlex/diagnosis/results/simplex_20k96/calibrated/sim/reads")
TSO = "TTTCTTATATGGG"
WL = Path(__file__).resolve().parents[1] / "barcodes" / "737K-august-2016.txt"


# ---- wells ----------------------------------------------------------------------------------

@pytest.mark.parametrize("name,well", [
    ("VDJ_123.fastq.gz", 123), ("well042.fastq.gz", 42), ("well000.fq", 0), ("VDJ_7.fasta", 7),
    ("sample1_VDJ_96.fastq", 96), ("/a/b/run2/VDJ_5.fa.gz", 5),
])
def test_well_from_path(name, well):
    assert well_from_path(name) == well


def test_well_from_path_without_number_raises():
    with pytest.raises(ValueError):
        well_from_path("reads.fastq.gz")


# ---- sharding -------------------------------------------------------------------------------

def test_shard_is_deterministic_and_spreads():
    bcs = pl.Series([f"{i:016d}" for i in range(4000)])
    a, b = shard_of(bcs, 16), shard_of(bcs, 16)
    assert a.to_list() == b.to_list()
    counts = a.value_counts()["count"]
    assert counts.len() == 16 and counts.min() > 150  # roughly uniform (expect 250)


def test_shard_writer_keeps_every_barcode_in_one_shard(tmp_path):
    rng = random.Random(1)
    bcs = ["".join(rng.choice("ACGT") for _ in range(16)) for _ in range(300)]
    w = ShardWriter(tmp_path, 7)
    for well in range(4):  # several batches appended to the same shard files
        w.write(pl.DataFrame({"barcode": bcs, "umi": ["A" * 10] * 300, "well": [well] * 300, "sequence": ["ACGT"] * 300}))
    shards = w.close()
    seen = {}
    total = 0
    for i in shards:
        df = pl.read_parquet(w.path(i))
        total += df.height
        for bc in df["barcode"].unique():
            assert bc not in seen
            seen[bc] = i
        assert df.group_by("barcode").agg(pl.col("well").n_unique())["well"].min() == 4
    assert total == 1200 and len(seen) == 300


def test_auto_shard_count(tmp_path):
    f = tmp_path / "well1.fastq.gz"
    f.write_bytes(b"x" * 1000)
    assert auto_shard_count([f]) == 1
    assert auto_shard_count([f, f, f], target_gz_bytes=1000) == 3


def _fastq(path, reads):
    with gzip.open(path, "wt") as fh:
        for i, (bc, umi, seq) in enumerate(reads):
            s = bc + umi + TSO + seq
            fh.write(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n")


def test_parse_to_shards_adds_well_and_streams(tmp_path):
    wl = [l.strip() for _, l in zip(range(50), open(WL))]
    files = []
    for well in (3, 12):
        rng = random.Random(well)
        reads = [(wl[i % 50], "".join(rng.choice("ACGT") for _ in range(10)), "CAGGTGCAG" * 5) for i in range(250)]
        f = tmp_path / f"VDJ_{well}.fastq.gz"
        _fastq(f, reads)
        files.append(str(f))
    w = ShardWriter(tmp_path / "shards", 4)
    with ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as ex:
        st = parse_to_shards(files, w, ex, tmp_path / "work", chunk_reads=60, max_in_flight=120, flush_reads=100)
    shards = w.close()
    df = pl.concat([pl.read_parquet(w.path(i)) for i in shards])
    assert st["wells"] == [3, 12] and st["input_reads"] == 500 and st["valid_reads"] == 500
    assert df.height == 500 and sorted(df["well"].unique().to_list()) == [3, 12]
    assert df["sequence"].str.starts_with("CAGGTG").all()
    assert not (tmp_path / "work" / "chunks").exists()


# ---- UMI collapsing -------------------------------------------------------------------------

def test_directional_collapse_follows_count_rule():
    counts = {"AAAAAAAAAA": 10, "AAAAAAAAAC": 3, "AAAAAAAACC": 1, "TTTTTTTTTT": 4, "TTTTTTTTTA": 4}
    fix = directional_collapse(counts)
    assert fix["AAAAAAAAAC"] == "AAAAAAAAAA"   # 10 >= 2*3-1
    assert fix["AAAAAAAACC"] == "AAAAAAAAAA"   # absorbed transitively via AC (3 >= 2*1-1)
    # 4 vs 4: 4 < 2*4-1, so neither absorbs the other
    assert fix["TTTTTTTTTT"] == "TTTTTTTTTT" and fix["TTTTTTTTTA"] == "TTTTTTTTTA"


def test_directional_collapse_needs_hamming_one():
    fix = directional_collapse({"AAAAAAAAAA": 50, "AAAAAAAACC": 1})
    assert fix["AAAAAAAACC"] == "AAAAAAAACC"


def test_collapse_counts_reads_across_wells():
    # parent UMI has 1 read in each of 3 wells (3 total); error UMI 2 reads in one well.
    # Per well, parent (1) could not absorb the error (2); pooled, 3 >= 2*2-1 does.
    reads = pl.DataFrame({
        "umi": ["AAAAAAAAAA"] * 3 + ["AAAAAAAAAT"] * 2,
        "well": [1, 2, 3, 1, 1],
        "sequence": ["ACGTACGTAC" * 20] * 5,
    })
    r = process_droplet_crosswell("BC", reads, min_cluster_reads=1)
    assert r["stats"]["raw_umis"] == 2 and r["stats"]["molecules"] == 1
    assert r["contigs"][0]["umis"] == 1 and r["contigs"][0]["raw_umis"] == 2


def test_umi_collapse_off_counts_raw_umis():
    reads = pl.DataFrame({
        "umi": ["AAAAAAAAAA"] * 3 + ["AAAAAAAAAT"] * 2,
        "well": [1, 2, 3, 1, 1],
        "sequence": ["ACGTACGTAC" * 20] * 5,
    })
    r = process_droplet_crosswell("BC", reads, min_cluster_reads=1, umi_collapse=False)
    assert r["stats"]["raw_umis"] == 2 and r["stats"]["molecules"] == 2
    assert r["contigs"][0]["umis"] == 2
    # AAAAAAAAAA: 1 read in each of wells 1-3 -> tie -> well 1; AAAAAAAAAT: well 1
    assert {(x["well"], x["umis"]) for x in r["wells"]} == {(1, 2), (2, 0), (3, 0)}


# ---- molecule wells -------------------------------------------------------------------------

def test_assign_molecule_wells_majority_and_tie():
    mol = assign_molecule_wells(["u", "u", "u", "v", "v"], [5, 5, 9, 2, 7])
    assert mol["u"] == (5, 2, 1)
    assert mol["v"] == (2, 1, 1)  # tie -> lowest well


def test_contig_well_counts_excludes_hops_from_umis():
    # molecule a: 9 reads in well 1, 1 hopped read in well 2; molecule b: 2 reads in well 2
    umis = ["a"] * 10 + ["b"] * 2
    wells = [1] * 9 + [2] + [2, 2]
    rows = {r["well"]: r for r in contig_well_counts(umis, wells)}
    assert rows[1] == {"well": 1, "umis": 1, "reads": 9, "umis_present": 1}
    assert rows[2] == {"well": 2, "umis": 1, "reads": 3, "umis_present": 2}


def test_process_droplet_crosswell_clusters_and_counts_per_well():
    rng = random.Random(0)
    heavy = "".join(rng.choice("ACGT") for _ in range(300))
    light = "".join(rng.choice("ACGT") for _ in range(280))
    rows = []
    # heavy: 3 molecules in well 4 (home), 1 leaked molecule in well 9; light: 2 molecules in well 4
    for umi, well, seq, n in [("AAAACCCCGG", 4, heavy, 4), ("AAAACCCCTT", 4, heavy, 3), ("GGGGCCCCAA", 4, heavy, 3),
                              ("TTTTGGGGAA", 9, heavy, 3), ("CACACACACA", 4, light, 5), ("GTGTGTGTGT", 4, light, 4)]:
        rows += [{"umi": umi, "well": well, "sequence": seq}] * n
    rows.append({"umi": "AAAACCCCGG", "well": 9, "sequence": heavy})  # one index-hopped read
    r = process_droplet_crosswell("BC", pl.DataFrame(rows), clustering_threshold=0.9)
    assert len(r["contigs"]) == 2
    h, l = r["contigs"]
    assert h["name"] == "BC_contig-0" and h["consensus"] == heavy and h["umis"] == 4 and h["main_well"] == 4
    assert l["consensus"] == light and l["n_wells"] == 1
    cw = {(x["name"], x["well"]): x for x in r["wells"]}
    assert cw[("BC_contig-0", 4)]["umis"] == 3 and cw[("BC_contig-0", 9)]["umis"] == 1
    assert cw[("BC_contig-0", 9)]["reads"] == 4 and cw[("BC_contig-0", 9)]["umis_present"] == 2
    assert r["stats"]["molecules_multiwell"] == 1 and r["stats"]["reads_off_main_well"] == 1


def test_leakage_from_totals_matches_estimate():
    sw = pl.DataFrame({"barcode": ["A"] * 8, "locus": ["IGH"] * 8, "v_gene": ["V"] * 8, "j_gene": ["J"] * 8,
                       "junction": ["X"] * 8, "well": list(range(8)), "n": [3] + [1] * 7})
    tot = pl.DataFrame({"N": [10, 2], "nmax": [3, 2]})  # the 2-UMI species is below min_umis
    assert leakage_from_totals(tot, 10) == pytest.approx(estimate_leakage(sw, 10))


# ---- end to end -----------------------------------------------------------------------------

CONTRACT = ["barcode", "well", "name", "confidence"] + [
    f"{c}:{s}" for s in (0, 1) for c in ("locus", "v_gene", "j_gene", "junction", "cdr3_aa", "sequence_id", "sequence", "umis")]
EXTRA = [f"{c}:{s}" for s in (0, 1) for c in ("umis_home", "umis_pooled", "n_wells", "p_home")] + ["margin"]


@pytest.mark.skipif(not SIM_READS.exists(), reason="SimPlex reads not available")
def test_end_to_end_across_wells(tmp_path, monkeypatch):
    import shutil

    import pairplex.pairplex as ppx

    reads = tmp_path / "reads"
    reads.mkdir()
    for w in ("well000", "well001"):
        shutil.copy(SIM_READS / f"{w}.fastq.gz", reads)
    monkeypatch.setenv("PAIRPLEX_MAX_WORKERS", "2")
    out = tmp_path / "out"
    ppx.run(str(reads), str(out), quiet=True, pooling="across_wells")
    pairs = pl.read_parquet(out / "pairs" / "pairs.parquet")
    assert pairs.height > 0
    assert set(CONTRACT + EXTRA) <= set(pairs.columns)
    assert pairs["well"].dtype.is_integer() and set(pairs["well"].unique()) <= {0, 1}
    assert (pairs["locus:0"] == "IGH").all() and pairs["locus:1"].is_in(["IGK", "IGL"]).all()
    assert pairs["confidence"].is_between(0, 1).all()
    assert (pairs["name"] == pairs["barcode"]).all()
    assert pairs["sequence_id:0"].str.starts_with(pairs["barcode"] + "_contig-").all()
    contigs = pl.read_parquet(out / "contigs" / "contigs.parquet")
    wells = pl.read_parquet(out / "contigs" / "contig_wells.parquet")
    assert contigs["name"].n_unique() == contigs.height
    assert {"locus", "v_gene", "junction", "pass_filters", "umis", "reads"} <= set(contigs.columns)
    assert set(wells.columns) == {"name", "well", "umis", "reads", "umis_present"}
    assert set(wells["name"]) <= set(contigs["name"])
    assert not (out / "annotated").exists() and not (out / "temp").exists()


@pytest.mark.parametrize("kwargs,crosswell", [({}, False), ({"legacy": True, "pooling": "across_wells"}, False),
                                               ({"pooling": "across_wells"}, True)])
def test_pooling_routing(tmp_path, monkeypatch, kwargs, crosswell):
    """Default and legacy=True take the untouched per-well path; only pooling='across_wells'
    (without legacy) dispatches to the cross-well pipeline."""
    import pairplex.crosswell as cw
    import pairplex.pairplex as ppx

    called = []
    monkeypatch.setattr(cw, "run_crosswell", lambda *a, **k: called.append(k))
    monkeypatch.setenv("PAIRPLEX_MAX_WORKERS", "1")
    (tmp_path / "reads").mkdir()
    ppx.run(str(tmp_path / "reads"), str(tmp_path / "out"), quiet=True, **kwargs)
    assert bool(called) is crosswell
    assert (tmp_path / "out" / "annotated").exists() is (not crosswell)


def test_invalid_pooling_raises(tmp_path):
    import pairplex.pairplex as ppx

    with pytest.raises(ValueError):
        ppx.run(str(tmp_path), str(tmp_path / "out"), quiet=True, pooling="bogus")
