"""Separate pre-split leakage from post-pool well swapping.

Before the 96-well split there is only reverse transcription (no amplification),
so each (droplet barcode, UMI) is ONE physical molecule and can be in only one well.
- The same (barcode, UMI) observed in >1 well  => post-pool event (index hopping,
  cross-well PCR chimera / well-index swap within a pool of 24, or demux error).
- Distinct UMIs of one chain spread across wells => the molecules were already in
  different wells at the split => barcoded cDNA was outside its cell (leakage).

Also reports whether cross-well copies stay within the same i7 pool of 24 wells
(VDJ_1-24, 25-48, ...), which distinguishes PCR-in-pool swapping (same pool) from
index hopping on the flow cell (any well).

Usage: python umi_cross_well.py PARSED_DIR OUT.parquet --wells 1-96 [--sample-mod 400]
"""
import argparse
import glob
import os
import re

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parsed_dir")
    ap.add_argument("out")
    ap.add_argument("--wells", default="1-96")
    ap.add_argument("--sample-mod", type=int, default=400)
    ap.add_argument("--pool-size", type=int, default=24)
    a = ap.parse_args()
    lo, hi = map(int, a.wells.split("-"))
    frames = []
    for f in glob.glob(os.path.join(a.parsed_dir, "VDJ_*.parquet")):
        m = re.search(r"VDJ_(\d+)\.parquet$", f)  # skips stray files such as VDJ_103-checkpoint.parquet
        if not m or not lo <= int(m.group(1)) <= hi:
            continue
        w = int(m.group(1))
        frames.append(
            pl.scan_parquet(f).select("barcode", "umi")
            .filter(pl.col("barcode").hash(seed=0) % a.sample_mod == 0)
            .group_by("barcode", "umi").agg(pl.len().alias("reads"))
            .with_columns(pl.lit(w).alias("well")).collect())
    df = pl.concat(frames)
    W = df["well"].n_unique()
    df = df.with_columns(((pl.col("well") - 1) // a.pool_size).alias("pool"))
    u = df.group_by("barcode", "umi").agg(
        pl.col("well").n_unique().alias("n_wells"), pl.col("pool").n_unique().alias("n_pools"),
        pl.col("reads").sum().alias("reads"), pl.col("reads").max().alias("top_well_reads"))
    u.write_parquet(a.out)
    multi = u.filter(pl.col("n_wells") > 1)
    print(f"{W} wells | sampled barcodes {u['barcode'].n_unique()} | (barcode,UMI) molecules {u.height}")
    print(f"  molecules seen in >1 well: {multi.height} ({multi.height / u.height:.2%})")
    if multi.height:
        print(f"    ... of these, all copies within ONE pool of {a.pool_size}: {(multi['n_pools'] == 1).mean():.1%} "
              f"(random expectation for 2 wells ~{(a.pool_size - 1) / (W - 1):.1%})")
        print(f"    read share of the top well: median {float((multi['top_well_reads'] / multi['reads']).median()):.3f}")
    for k in (2, 5, 10):
        s = u.filter(pl.col("reads") >= k)
        print(f"  molecules with >= {k} reads: {s.height}, in >1 well: {(s['n_wells'] > 1).mean():.2%}")
    rd = u.filter(pl.col("n_wells") > 1)["reads"].sum() - u.filter(pl.col("n_wells") > 1)["top_well_reads"].sum()
    print(f"  reads outside the molecule's top well: {rd / u['reads'].sum():.3%} of all reads")


if __name__ == "__main__":
    main()
