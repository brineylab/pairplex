"""Clonal light-chain purity of `pairplex pair-pooled` output by confidence bin, vs the per-well
pairs of the same wells (real-data validation of the production pooled pairing).

Usage: python pooled_purity.py POOLED_PAIRS.parquet PAIRPLEX_RUN_DIR --wells 97-192 --out OUT.csv
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import polars as pl  # noqa: E402

BINS = [0.1, 0.3, 0.6, 0.9]
LABELS = ["<0.1", "0.1-0.3", "0.3-0.6", "0.6-0.9", ">=0.9"]


def norm_lv(e):
    return e.str.replace(r"^(IG[KL]V\d+)D-", "${1}-")


def purity(df):
    c = df.group_by("hv", "hcdr3", "lv").len()
    cl = c.group_by("hv", "hcdr3").agg(pl.col("len").sum().alias("size"), pl.col("len").max().alias("top")).filter(pl.col("size") >= 2)
    return (float(cl["top"].sum() / cl["size"].sum()) if cl.height else None), cl.height, int(cl["size"].sum()) if cl.height else 0


def chance(df):
    p = df["lv"].value_counts()["count"].to_numpy().astype(float)
    p /= p.sum()
    return float((p ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pooled")
    ap.add_argument("run_dir")
    ap.add_argument("--wells", default="97-192")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.wells.split("-"))
    p = pl.read_parquet(a.pooled).select(
        pl.col("v_gene:0").alias("hv"), pl.col("cdr3_aa:0").alias("hcdr3"), norm_lv(pl.col("v_gene:1")).alias("lv"),
        "confidence", pl.col("umis_home:0"), pl.col("umis_home:1"))
    p = p.with_columns(pl.col("confidence").cut(BINS, labels=LABELS, left_closed=True).cast(pl.Utf8).alias("bin"))
    rows = []
    for lab in LABELS:
        b = p.filter(pl.col("bin") == lab)
        pu, ncl, nc = purity(b)
        rows.append(dict(set=f"pooled conf {lab}", pairs=b.height, clones_ge2=ncl, cells_in_clones=nc, purity=pu,
                         chance=chance(b) if b.height else None))
    for lab, f in (("pooled UMIs home >=3 both", (pl.col("umis_home:0") >= 3) & (pl.col("umis_home:1") >= 3)),
                   ("pooled all", pl.lit(True))):
        b = p.filter(f)
        pu, ncl, nc = purity(b)
        rows.append(dict(set=lab, pairs=b.height, clones_ge2=ncl, cells_in_clones=nc, purity=pu, chance=chance(b) if b.height else None))
    base = []
    for f in glob.glob(os.path.join(a.run_dir, "annotated", "*_paired.parquet")):
        m = re.search(r"(\d+)", os.path.basename(f))
        if m and lo <= int(m.group(1)) <= hi and os.path.getsize(f) > 100 and "v_gene:0" in pl.read_parquet_schema(f):
            base.append(pl.read_parquet(f, columns=["v_gene:0", "cdr3_aa:0", "v_gene:1", "locus:0", "locus:1"]))
    base = pl.concat(base).filter((pl.col("locus:0") == "IGH") & pl.col("locus:1").is_in(["IGK", "IGL"])).select(
        pl.col("v_gene:0").alias("hv"), pl.col("cdr3_aa:0").alias("hcdr3"), norm_lv(pl.col("v_gene:1")).alias("lv"))
    pu, ncl, nc = purity(base)
    rows.append(dict(set="current PairPlex per-well", pairs=base.height, clones_ge2=ncl, cells_in_clones=nc, purity=pu, chance=chance(base)))
    r = pl.DataFrame(rows)
    r.write_csv(a.out)
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(200)
    print(r.with_columns(pl.col("purity").round(3), pl.col("chance").round(3)))


if __name__ == "__main__":
    main()
