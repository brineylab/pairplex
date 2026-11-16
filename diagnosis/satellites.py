"""(a) pre-RT ambient capture vs (b) post-break re-barcoding: where do satellite copies sit?

For chains found under many droplet barcodes (candidates from across_barcode.py), define
  parent    = the barcode with the most UMIs for that chain (putative real cell)
  home well = the parent's modal well for that chain
  satellites = occurrences of the chain under OTHER barcodes.
Predictions for the fraction of satellite occurrences located in the parent's home well:
  (a) ambient mRNA barcoded in other droplets before RT -> free cDNA, independent of the
      parent cell's well: ~ depth share of that well (~1/W).
  (b) re-barcoding after the emulsion break (free gel-bead oligos) -> the copy is made where
      its template is; templates still in the cell (~22% given ~78% leakage) are in the home
      well: ~ 0.22 + 0.78/W.
Expectation is depth-weighted (share of all occurrences of that locus in each well). A
permutation control shuffles home wells across keys (should give enrichment ~1).

Usage: python satellites.py ROOT DONOR KEYS.parquet --wells 1-96 --out OUT.parquet
KEYS.parquet = output of across_barcode.py (uses its `key`, `contaminant`, `n_barcodes`).
"""
import argparse
import glob
import os
import re

import numpy as np
import polars as pl

COLS = ["sequence_id", "locus", "productive", "v_gene", "j_gene", "junction"]


def files(root, donor, kind, lo, hi):
    pat = {"ann": "annotated/VDJ_*_unpaired.parquet", "meta": "metadata/VDJ_*.csv"}[kind]
    out = []
    for f in glob.glob(os.path.join(root, donor, "2_pairplexed", pat)):
        w = int(re.search(r"VDJ_(\d+)", os.path.basename(f)).group(1))
        if lo <= w <= hi and os.path.getsize(f) > 100:
            out.append((w, f))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("donor")
    ap.add_argument("keys")
    ap.add_argument("--wells", default="1-96")
    ap.add_argument("--min-barcodes", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.wells.split("-"))

    keys = pl.read_parquet(a.keys).filter(pl.col("contaminant") & (pl.col("n_barcodes") >= a.min_barcodes)).select("key")
    print(f"candidate keys: {keys.height}")

    occ, depth = [], []
    for w, f in files(a.root, a.donor, "ann", lo, hi):
        lf = (pl.scan_parquet(f).select(COLS)
              .filter(pl.col("productive") & pl.col("locus").is_in(["IGH", "IGK", "IGL"]) & pl.col("junction").is_not_null())
              .with_columns(pl.concat_str(["locus", "v_gene", "j_gene", "junction"], separator="|").alias("key"),
                            pl.col("sequence_id").str.split("_").list.first().alias("barcode"), pl.lit(w).alias("well")))
        depth.append(lf.group_by("locus", "well").len().collect())
        occ.append(lf.join(keys.lazy(), on="key", how="semi").select("key", "locus", "barcode", "well", "sequence_id").collect())
    occ = pl.concat(occ)
    depth = pl.concat(depth).with_columns((pl.col("len") / pl.col("len").sum().over("locus")).alias("w_share")).select("locus", "well", "w_share")
    W = depth["well"].n_unique()

    names = occ.select("sequence_id").unique()
    umis = pl.concat([pl.scan_csv(f).select("name", "umis").join(names.lazy(), left_on="name", right_on="sequence_id", how="semi")
                        .with_columns(pl.lit(w).alias("well")).collect() for w, f in files(a.root, a.donor, "meta", lo, hi)])
    occ = occ.join(umis, left_on=["sequence_id", "well"], right_on=["name", "well"], how="left").with_columns(pl.col("umis").fill_null(1))

    # parent barcode per key (most UMIs summed over wells), its home well, and its margin over the runner-up
    bc = occ.group_by("key", "barcode").agg(pl.col("umis").sum().alias("bc_umis"))
    ranked = bc.sort(["key", "bc_umis"], descending=[False, True]).group_by("key", maintain_order=True).agg(
        pl.col("barcode").first().alias("parent"), pl.col("bc_umis").first().alias("parent_umis"),
        pl.col("bc_umis").slice(1, 1).first().fill_null(0).alias("second_umis"))
    home = (occ.join(ranked.select("key", "parent"), left_on=["key", "barcode"], right_on=["key", "parent"])
               .sort("umis", descending=True).group_by("key").agg(pl.col("well").first().alias("home")))
    sat = (occ.join(ranked, on="key").filter(pl.col("barcode") != pl.col("parent"))
              .join(home, on="key").join(depth, left_on=["locus", "home"], right_on=["locus", "well"], how="left"))
    sat = sat.with_columns((pl.col("well") == pl.col("home")).alias("in_home"))
    sat.write_parquet(a.out)

    def report(df, label):
        if df.height == 0:
            return
        obs, exp = df["in_home"].mean(), df["w_share"].mean()
        print(f"  {label:42s} satellites={df.height:7d}  in home well: obs {obs:.4f}  exp(a) {exp:.4f}  "
              f"enrichment {obs / exp:6.2f}x   [pred (b) ~ {0.22 + 0.78 / W:.3f}]")

    print(f"W={W} wells | keys with satellites: {sat['key'].n_unique()} | satellite occurrences: {sat.height}")
    report(sat, "all candidates")
    for loc in ("IGH", "IGK", "IGL"):
        report(sat.filter(pl.col("locus") == loc), f"locus {loc}")
    clear = sat.filter((pl.col("parent_umis") >= 3) & (pl.col("parent_umis") >= 2 * pl.col("second_umis")))
    report(clear, "clear parent (>=3 UMIs, >=2x runner-up)")
    report(sat.filter(pl.col("umis") == 1), "satellites with 1 UMI")
    report(sat.filter(pl.col("umis") >= 2), "satellites with >=2 UMIs")

    # permutation control: shuffle home wells across keys within locus
    rng = np.random.default_rng(0)
    h = home.join(ranked.select("key"), on="key").join(sat.select("key", "locus").unique(), on="key")
    enr = []
    for _ in range(a.n_perm):
        perm = h.with_columns(pl.col("home").shuffle(seed=int(rng.integers(1 << 31))).over("locus").alias("home_p"))
        s = (sat.drop("home", "w_share", "in_home").join(perm.select("key", "home_p"), on="key")
                .join(depth, left_on=["locus", "home_p"], right_on=["locus", "well"], how="left"))
        enr.append((s["well"] == s["home_p"]).mean() / s["w_share"].mean())
    enr = np.array(enr)
    print(f"  permutation control (home wells shuffled across keys, n={a.n_perm}): enrichment mean {enr.mean():.2f}x, "
          f"95% range {np.quantile(enr, 0.025):.2f}-{np.quantile(enr, 0.975):.2f}x")


if __name__ == "__main__":
    main()
