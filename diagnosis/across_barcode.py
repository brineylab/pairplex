"""Across-barcode axis: chains found under far more droplet barcodes than clonality allows.

Leaked barcoded cDNA keeps its droplet barcode, so it spreads across WELLS, not
barcodes. A V(D)J found under many DIFFERENT barcodes is either
  (a) many distinct cells: clonal expansion or convergent/public recombination, or
  (b) ambient mRNA captured into many droplets before RT (each droplet stamps its own barcode).

Two null models, both from the cohort itself (no external reference):
  * heavy chains in a NAIVE reaction: expansion and convergence are negligible (diverse
    CDR3, N additions), so the heavy n_barcodes distribution is the clonality baseline;
  * light chains recur naturally (few N additions). Expected barcodes for a key in the
    target = its barcode frequency in comparator donors (same compartment) x target size.
    Ambient contaminants specific to the target reaction are enriched vs that expectation;
    public chains are not.

Key = locus | V gene | J gene | junction (nt). Keys are hash-sampled (key hash %% mod == 0),
so each sampled key keeps its FULL barcode count. Single process, lazy scans.

Usage:
  python across_barcode.py ROOT TARGET_DONOR --comparators donor02 donor09 --wells 1-96 \
      --out OUT.parquet [--sample-mod 10]
ROOT contains <donor>/2_pairplexed/{annotated,metadata}.
"""
import argparse
import glob
import math
import os
import re

import polars as pl

COLS = ["sequence_id", "locus", "productive", "v_gene", "j_gene", "junction", "junction_aa"]


def files(root, donor, kind, lo, hi):
    pat = {"ann": "annotated/VDJ_*_unpaired.parquet", "meta": "metadata/VDJ_*.csv"}[kind]
    out = []
    for f in glob.glob(os.path.join(root, donor, "2_pairplexed", pat)):
        w = int(re.search(r"VDJ_(\d+)", os.path.basename(f)).group(1))
        if lo <= w <= hi and os.path.getsize(f) > 100:
            out.append((w, f))
    return sorted(out)


def scan_donor(root, donor, lo, hi, mod):
    """Returns (sampled occurrences, total productive occurrences per locus)."""
    occ, tot = [], []
    for w, f in files(root, donor, "ann", lo, hi):
        lf = (pl.scan_parquet(f).select(COLS)
              .filter(pl.col("productive") & pl.col("locus").is_in(["IGH", "IGK", "IGL"]) & pl.col("junction").is_not_null())
              .with_columns(pl.col("sequence_id").str.split("_").list.first().alias("barcode"),
                            pl.concat_str(["locus", "v_gene", "j_gene", "junction"], separator="|").alias("key"),
                            pl.lit(w).alias("well")))
        tot.append(lf.group_by("locus").agg(pl.col("barcode").n_unique().alias("n")).collect())
        occ.append(lf.filter(pl.col("key").hash(seed=0) % mod == 0)
                     .select("key", "locus", "v_gene", "j_gene", "junction_aa", "barcode", "well", "sequence_id").collect())
    # per-locus totals: barcode-occurrences summed over wells (a barcode in 2 wells counts twice; same for keys below)
    tot = pl.concat(tot).group_by("locus").agg(pl.col("n").sum())
    return pl.concat(occ), dict(tot.iter_rows())


def poisson_sf(k, lam):
    """P(X >= k) for X ~ Poisson(lam); stable enough for reporting."""
    if k <= 0:
        return 1.0
    term = math.exp(-lam)
    cdf = term
    for i in range(1, k):
        term *= lam / i
        cdf += term
    return max(0.0, 1.0 - cdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("target")
    ap.add_argument("--comparators", nargs="+", required=True)
    ap.add_argument("--wells", default="1-96")
    ap.add_argument("--sample-mod", type=int, default=10)
    ap.add_argument("--min-barcodes", type=int, default=5)
    ap.add_argument("--min-ratio", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.wells.split("-"))

    occ, tot_t = scan_donor(a.root, a.target, lo, hi, a.sample_mod)
    print(f"target {a.target} wells {lo}-{hi}: sampled occurrences {occ.height}; total barcode-occurrences by locus {tot_t}")

    # UMIs per occurrence from metadata (only sampled contig names)
    names = occ.select("sequence_id").unique()
    umis = []
    for w, f in files(a.root, a.target, "meta", lo, hi):
        umis.append(pl.scan_csv(f).select("name", "umis").join(names.lazy(), left_on="name", right_on="sequence_id", how="semi")
                      .with_columns(pl.lit(w).alias("well")).collect())
    umis = pl.concat(umis)
    occ = occ.join(umis, left_on=["sequence_id", "well"], right_on=["name", "well"], how="left")

    keys = occ.group_by("key", "locus", "v_gene", "j_gene", "junction_aa").agg(
        pl.struct("barcode", "well").n_unique().alias("obs"),
        pl.col("barcode").n_unique().alias("n_barcodes"),
        pl.col("well").n_unique().alias("n_wells"),
        pl.col("umis").median().alias("umis_per_occ_median"),
        pl.col("umis").mean().alias("umis_per_occ_mean"))

    # comparator frequencies
    comp_counts, comp_tot = [], {}
    for d in a.comparators:
        o, t = scan_donor(a.root, d, lo, hi, a.sample_mod)
        comp_counts.append(o.group_by("key").agg(pl.struct("barcode", "well").n_unique().alias("c")))
        for k, v in t.items():
            comp_tot[k] = comp_tot.get(k, 0) + v
        print(f"comparator {d}: total barcode-occurrences by locus {t}")
    comp = pl.concat(comp_counts).group_by("key").agg(pl.col("c").sum().alias("comp_obs"))
    keys = keys.join(comp, on="key", how="left").with_columns(pl.col("comp_obs").fill_null(0))
    keys = keys.with_columns(
        ((pl.col("comp_obs") + 0.5) / pl.col("locus").replace_strict(comp_tot, return_dtype=pl.Float64)
         * pl.col("locus").replace_strict(tot_t, return_dtype=pl.Float64)).alias("expected"))
    keys = keys.with_columns((pl.col("obs") / pl.col("expected")).alias("ratio"))
    keys = keys.with_columns(pl.struct("obs", "expected").map_elements(
        lambda s: poisson_sf(int(s["obs"]), float(s["expected"])), return_dtype=pl.Float64).alias("p_enriched"))
    keys = keys.with_columns(((pl.col("n_barcodes") >= a.min_barcodes) & (pl.col("ratio") >= a.min_ratio)
                              & (pl.col("p_enriched") < 1e-6)).alias("contaminant"))
    keys.write_parquet(a.out)

    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_fmt_str_lengths(40)
    print(f"\n== n_barcodes distribution per key (hash-sampled 1/{a.sample_mod} of keys) ==")
    dist = (keys.with_columns(pl.col("n_barcodes").cut([1, 2, 4, 9, 49, 199], labels=["1", "2", "3-4", "5-9", "10-49", "50-199", ">=200"]).alias("bin"))
                .group_by("locus", "bin").agg(pl.len().alias("keys"), pl.col("obs").sum().alias("occurrences"))
                .with_columns((pl.col("occurrences") / pl.col("occurrences").sum().over("locus")).round(4).alias("frac_occ"))
                .sort("locus", "bin"))
    print(dist)

    print(f"\n== contaminant candidates: n_barcodes >= {a.min_barcodes}, obs/expected >= {a.min_ratio}, Poisson p < 1e-6 ==")
    summ = (keys.group_by("locus").agg(
        pl.len().alias("keys"), pl.col("contaminant").sum().alias("cand_keys"),
        pl.col("obs").filter(pl.col("contaminant")).sum().alias("cand_occ"), pl.col("obs").sum().alias("all_occ"),
        pl.col("umis_per_occ_mean").filter(pl.col("contaminant")).mean().round(2).alias("cand_umis_per_occ"),
        pl.col("umis_per_occ_mean").filter(~pl.col("contaminant")).mean().round(2).alias("other_umis_per_occ"),
        pl.col("n_wells").filter(pl.col("contaminant")).median().alias("cand_median_wells"))
        .with_columns((pl.col("cand_occ") / pl.col("all_occ")).round(4).alias("cand_frac_occ")).sort("locus"))
    print(summ)
    c = keys.filter("contaminant")
    if c.height:
        lc = c.filter(pl.col("locus") != "IGH")["obs"].sum()
        print(f"candidate occurrences that are light chains: {lc}/{c['obs'].sum()} ({lc / c['obs'].sum():.1%})")
    print("\ntop candidates:")
    print(keys.filter("contaminant").sort("obs", descending=True)
              .select("locus", "v_gene", "j_gene", "junction_aa", "n_barcodes", "n_wells", "comp_obs", "expected", "ratio", "umis_per_occ_mean").head(15))
    print("\nmost-observed keys overall (for the public-chain check):")
    print(keys.sort("obs", descending=True)
              .select("locus", "v_gene", "j_gene", "junction_aa", "n_barcodes", "comp_obs", "expected", "ratio", "contaminant").head(10))


if __name__ == "__main__":
    main()
