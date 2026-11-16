"""Cross-well residency with each molecule counted ONCE (at its top well).

Fixes the inflation in residency.py: post-pool swapping puts stray copies of a
molecule (same barcode+UMI) into other wells, which per-well contig UMI counts
treat as extra molecules. Here every (barcode, UMI) is assigned to the well with
most of its reads, represented by its most frequent read, and only then grouped
into chain species within the barcode. The outside-modal-well fraction of DISTINCT
molecules is then a direct estimate of pre-split displacement (leakage).

Usage: python residency_dedup.py PARSED_DIR OUT.parquet --wells 1-96 [--sample-mod 400] [--species-id 0.94]
"""
import argparse
import glob
import os
import re

import edlib
import polars as pl


def hw_identity(a, b):
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if not s:
        return 0.0
    return 1 - edlib.align(s, l, mode="HW", task="distance")["editDistance"] / len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parsed_dir")
    ap.add_argument("out")
    ap.add_argument("--wells", default="1-96")
    ap.add_argument("--sample-mod", type=int, default=400)
    ap.add_argument("--species-id", type=float, default=0.94)
    ap.add_argument("--min-len", type=int, default=200)
    a = ap.parse_args()
    lo, hi = map(int, a.wells.split("-"))
    frames = []
    for f in glob.glob(os.path.join(a.parsed_dir, "VDJ_*.parquet")):
        w = int(re.search(r"VDJ_(\d+)\.parquet$", f).group(1))
        if lo <= w <= hi:
            frames.append(pl.scan_parquet(f).select("barcode", "umi", "sequence")
                          .filter(pl.col("barcode").hash(seed=0) % a.sample_mod == 0)
                          .with_columns(pl.lit(w).alias("well")).collect())
    reads = pl.concat(frames)
    W = reads["well"].n_unique()
    # molecule -> top well
    top = (reads.group_by("barcode", "umi", "well").len()
                .sort("len", descending=True).group_by("barcode", "umi").agg(
                    pl.col("well").first().alias("well"), pl.col("len").sum().alias("reads")))
    # molecule -> representative read (mode, from its top well)
    rep = (reads.join(top.select("barcode", "umi", "well"), on=["barcode", "umi", "well"])
                .group_by("barcode", "umi", "sequence").len()
                .sort("len", descending=True).group_by("barcode", "umi").agg(pl.col("sequence").first()))
    mol = top.join(rep, on=["barcode", "umi"]).filter(pl.col("sequence").str.len_chars() >= a.min_len)
    print(f"{W} wells | barcodes {mol['barcode'].n_unique()} | molecules {mol.height}")

    rows = []
    for (bc,), g in mol.sort("reads", descending=True).group_by("barcode", maintain_order=True):
        reps, members = [], []
        for well, seq in g.select("well", "sequence").iter_rows():
            for i, r in enumerate(reps):
                if hw_identity(seq, r) >= a.species_id:
                    members[i].append(well)
                    break
            else:
                reps.append(seq)
                members.append([well])
        for m in members:
            counts = {}
            for w in m:
                counts[w] = counts.get(w, 0) + 1
            rows.append(dict(barcode=bc, umis=len(m), n_wells=len(counts), modal=max(counts.values())))
    r = pl.DataFrame(rows)
    r.write_parquet(a.out)
    summ = (r.with_columns(pl.col("umis").cut([1, 2, 3, 5, 10, 20], labels=["1", "2", "3", "4-5", "6-10", "11-20", ">20"]).alias("umi_bin"))
             .group_by("umi_bin").agg(pl.len().alias("species"), pl.col("n_wells").median().alias("median_wells"),
                                      (pl.col("n_wells") == 1).mean().round(3).alias("frac_single_well"),
                                      (1 - pl.col("modal").sum() / pl.col("umis").sum()).round(3).alias("distinct_umis_outside_modal"))
             .sort("umi_bin"))
    pl.Config.set_tbl_rows(20)
    print(f"species-id {a.species_id}")
    print(summ)
    big = r.filter(pl.col("umis") >= 6)
    if big.height:
        out = 1 - big["modal"].sum() / big["umis"].sum()
        print(f">=6 distinct UMIs: outside-modal {out:.3f} -> implied displacement f ~= {out / (1 - 1 / W):.3f} (W={W})")


if __name__ == "__main__":
    main()
