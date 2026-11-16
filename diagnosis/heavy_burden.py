"""Per-donor ambient burden from naive heavy chains (no comparator needed).

In a naive compartment, the same heavy V|J|junction(nt) under several droplet barcodes is not
explained by clonality or convergence; leaked cDNA keeps its barcode, so cross-barcode copies
are ambient (pre-RT capture in other droplets; see satellites.py). Burden metrics, computed on
hash-sampled keys (each keeps its full barcode count):
  sat_frac      = sum(n_barcodes - 1) / sum(n_barcodes)  over all keys  (copies beyond one per chain)
  sat_frac_ge5  = same, restricted to keys with >= 5 barcodes
  occ_frac_geK  = share of heavy barcode-occurrences in keys seen under >= K barcodes
Writes one JSON line per donor.

Usage: python heavy_burden.py ROOT DONOR --wells 1-96 --out OUT.json [--sample-mod 10]
"""
import argparse
import glob
import json
import os
import re
import time

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("donor")
    ap.add_argument("--wells", default="1-96")
    ap.add_argument("--sample-mod", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.wells.split("-"))
    t0 = time.time()
    occ, n_total, n_wells = [], 0, 0
    for f in sorted(glob.glob(os.path.join(a.root, a.donor, "2_pairplexed", "annotated", "VDJ_*_unpaired.parquet"))):
        w = int(re.search(r"VDJ_(\d+)_", os.path.basename(f)).group(1))
        if not (lo <= w <= hi) or os.path.getsize(f) < 100:
            continue
        lf = (pl.scan_parquet(f).select("sequence_id", "locus", "productive", "v_gene", "j_gene", "junction")
              .filter(pl.col("productive") & (pl.col("locus") == "IGH") & pl.col("junction").is_not_null())
              .with_columns(pl.col("sequence_id").str.split("_").list.first().alias("barcode"),
                            pl.concat_str(["v_gene", "j_gene", "junction"], separator="|").alias("key")))
        n_total += lf.select(pl.len()).collect().item()
        n_wells += 1
        occ.append(lf.filter(pl.col("key").hash(seed=0) % a.sample_mod == 0)
                     .select("key", "barcode").with_columns(pl.lit(w).alias("well")).collect())
    occ = pl.concat(occ)
    k = occ.group_by("key").agg(pl.col("barcode").n_unique().alias("nb"), pl.len().alias("occ"))
    res = {
        "donor": a.donor, "wells": a.wells, "n_wells": n_wells,
        "heavy_occurrences_total": n_total, "sampled_keys": k.height, "sampled_occurrences": int(k["occ"].sum()),
        "sat_frac": float((k["nb"] - 1).sum() / k["nb"].sum()),
        "sat_frac_ge5": float(((k.filter(pl.col("nb") >= 5)["nb"] - 1).sum()) / max(k["nb"].sum(), 1)),
        **{f"occ_frac_ge{K}": float(k.filter(pl.col("nb") >= K)["occ"].sum() / k["occ"].sum()) for K in (2, 5, 10)},
        "max_barcodes_per_key": int(k["nb"].max()),
        "sec": round(time.time() - t0, 1),
    }
    with open(a.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
