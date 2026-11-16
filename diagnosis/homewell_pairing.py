"""Prototype: pool by barcode across wells, assign each chain a home well, pair H/L by
shared home well, and validate with clonal light-chain purity by confidence bin.

Model (measured, residency_dedup.py): each molecule of a cell stays in the cell's well with
prob 1-f, otherwise lands in a uniformly random well of W. For a chain species with UMI
counts n_w over wells, the posterior that well w is home (uniform prior) is
    P(home = w) = r^{n_w} / (sum_{observed} r^{n_v} + (W - k)),   r = (1 - f + f/W) / (f/W)
Chain species = (barcode, locus, V gene, J gene, junction nt), pooled over wells.
f is estimated from the reaction itself (species with >= 6 UMIs; outside-modal fraction).

Pairing: candidate (H, L, w) for H and L of the same barcode both observed in well w;
confidence = P_H(w) * P_L(w); keep reciprocal best pairs.

Validation: clonal light-chain purity (WETLAB_HANDOFF method): clone = (heavy V, CDRH3 aa)
within the reaction; purity = fraction of a clone's cells carrying the clone's most common
light V (D paralogs collapsed). Reported (i) within each confidence bin, (ii) against a
leave-one-out consensus light V from high-confidence pairs, plus the chance level and the
current per-well PairPlex pairs of the same wells as baseline.

Usage: python homewell_pairing.py PAIRPLEX_DIR --wells 97-192 --out-prefix PREFIX
"""
import argparse
import glob
import os
import re

import numpy as np
import polars as pl

ANN = ["sequence_id", "locus", "productive", "v_gene", "j_gene", "junction", "cdr3_aa"]
BINS = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0001]
LABELS = ["<0.1", "0.1-0.3", "0.3-0.6", "0.6-0.9", ">=0.9"]


def wells(d, kind, lo, hi):
    pat = {"ann": "annotated/VDJ_*_unpaired.parquet", "pair": "annotated/VDJ_*_paired.parquet", "meta": "metadata/VDJ_*.csv"}[kind]
    out = []
    for f in glob.glob(os.path.join(d, pat)):
        w = int(re.search(r"VDJ_(\d+)", os.path.basename(f)).group(1))
        if lo <= w <= hi and os.path.getsize(f) > 100:
            out.append((w, f))
    return sorted(out)


def norm_lv(expr):
    return expr.str.replace(r"^(IG[KL]V\d+)D-", "${1}-")


def purity(df, hv="hv", hcdr3="hcdr3", lv="lv", min_size=2):
    """cells-weighted fraction carrying the clone's modal light V, clones with >= min_size cells."""
    c = df.group_by(hv, hcdr3, lv).len()
    cl = c.group_by(hv, hcdr3).agg(pl.col("len").sum().alias("size"), pl.col("len").max().alias("top"))
    cl = cl.filter(pl.col("size") >= min_size)
    if cl.height == 0:
        return None, 0, 0
    return float(cl["top"].sum() / cl["size"].sum()), cl.height, int(cl["size"].sum())


def chance(df, lv="lv"):
    p = df[lv].value_counts()["count"].to_numpy().astype(float)
    p /= p.sum()
    return float((p ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pp_dir")
    ap.add_argument("--wells", default="97-192")
    ap.add_argument("--f", type=float, default=None, help="leakage; estimated from data if omitted")
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.wells.split("-"))

    occ = []
    for w, f in wells(a.pp_dir, "ann", lo, hi):
        occ.append(pl.scan_parquet(f).select(ANN)
                   .filter(pl.col("productive") & pl.col("locus").is_in(["IGH", "IGK", "IGL"]) & pl.col("junction").is_not_null())
                   .with_columns(pl.col("sequence_id").str.split("_").list.first().alias("barcode"), pl.lit(w).alias("well"))
                   .drop("productive").collect())
    occ = pl.concat(occ)
    names = occ.select("sequence_id", "well")
    umis = pl.concat([pl.scan_csv(f).select("name", "umis").with_columns(pl.lit(w).alias("well"))
                        .join(names.lazy(), left_on=["name", "well"], right_on=["sequence_id", "well"], how="semi").collect()
                      for w, f in wells(a.pp_dir, "meta", lo, hi)])
    occ = occ.join(umis, left_on=["sequence_id", "well"], right_on=["name", "well"], how="left").with_columns(pl.col("umis").fill_null(1))
    W = occ["well"].n_unique()
    print(f"wells {lo}-{hi}: W={W} | productive contigs {occ.height} | barcodes {occ['barcode'].n_unique()}")

    sp_cols = ["barcode", "locus", "v_gene", "j_gene", "junction"]
    sw = occ.group_by(*sp_cols, "well").agg(pl.col("umis").sum().alias("n"), pl.col("cdr3_aa").first())
    tot = sw.group_by(sp_cols).agg(pl.col("n").sum().alias("N"), pl.col("n").max().alias("nmax"), pl.len().alias("k"))
    big = tot.filter(pl.col("N") >= 6)
    out_frac = 1 - big["nmax"].sum() / big["N"].sum()
    f = a.f if a.f is not None else min(0.99, out_frac / (1 - 1 / W))
    r = (1 - f + f / W) / (f / W)
    print(f"chain species {tot.height} | outside-modal fraction (N>=6) {out_frac:.3f} -> f={f:.3f}, r={r:.1f}")

    # posterior per (species, well) in log space: log P = n log r - logZ, Z = sum_obs r^n + (W - k)
    lr = np.log(r)
    sw = sw.join(tot.select(*sp_cols, "k"), on=sp_cols)
    sw = sw.with_columns((pl.col("n") * lr).alias("ln"))
    z = sw.group_by(sp_cols).agg(pl.col("ln").max().alias("m"), pl.col("ln").alias("lns"), pl.col("k").first())
    z = z.with_columns(
        (pl.col("m") + (pl.col("lns").list.eval((pl.element() - pl.element().max()).exp()).list.sum()
                        + (W - pl.col("k")) * (-pl.col("m")).exp()).log()).alias("logZ")).select(*sp_cols, "logZ")
    sw = sw.join(z, on=sp_cols).with_columns((pl.col("ln") - pl.col("logZ")).exp().alias("p_home"))

    H = sw.filter(pl.col("locus") == "IGH").select("barcode", "well", pl.col("v_gene").alias("hv"), pl.col("j_gene").alias("hj"),
                                                   pl.col("junction").alias("hjunc"), pl.col("cdr3_aa").alias("hcdr3"),
                                                   pl.col("p_home").alias("pH"), pl.col("n").alias("nH"))
    L = sw.filter(pl.col("locus") != "IGH").select("barcode", "well", pl.col("locus").alias("llocus"), pl.col("v_gene").alias("lv_raw"),
                                                   pl.col("j_gene").alias("lj"), pl.col("junction").alias("ljunc"),
                                                   pl.col("p_home").alias("pL"), pl.col("n").alias("nL"))
    cand = H.join(L, on=["barcode", "well"]).with_columns((pl.col("pH") * pl.col("pL")).alias("conf"))
    hkey, lkey = ["barcode", "hv", "hj", "hjunc"], ["barcode", "llocus", "lv_raw", "lj", "ljunc"]
    cand = cand.with_columns(pl.col("conf").rank("ordinal", descending=True).over(hkey).alias("rh"),
                             pl.col("conf").rank("ordinal", descending=True).over(lkey).alias("rl"))
    pairs = cand.filter((pl.col("rh") == 1) & (pl.col("rl") == 1))
    pairs = pairs.with_columns(norm_lv(pl.col("lv_raw")).alias("lv"),
                               pl.col("conf").cut(BINS[1:-1], labels=LABELS, left_closed=True).alias("bin"))
    pairs.write_parquet(f"{a.out_prefix}_pairs.parquet")
    print(f"home-well pairs (reciprocal best): {pairs.height} | barcodes {pairs['barcode'].n_unique()}")

    # reference consensus light V per clone from high-confidence pairs (leave-one-out applied below)
    hi_ = pairs.filter(pl.col("conf") >= 0.9)
    ref = hi_.group_by("hv", "hcdr3", "lv").len().rename({"len": "votes"})
    ref_tot = hi_.group_by("hv", "hcdr3").len().rename({"len": "ref_n"})

    def loo_agree(df):
        d = df.join(ref_tot, on=["hv", "hcdr3"], how="inner")
        if d.height == 0:
            return None, 0
        self_vote = (pl.col("conf") >= 0.9).cast(pl.Int64)
        # votes for this pair's own light V, minus its own vote; compare to best competing light V
        d = d.join(ref, on=["hv", "hcdr3", "lv"], how="left").with_columns(pl.col("votes").fill_null(0))
        best = ref.sort("votes", descending=True).group_by("hv", "hcdr3").agg(pl.col("lv").first().alias("ref_lv"), pl.col("votes").first().alias("ref_votes"),
                                                                              pl.col("votes").slice(1, 1).first().fill_null(0).alias("second_votes"))
        d = d.join(best, on=["hv", "hcdr3"]).with_columns((pl.col("ref_n") - self_vote).alias("ref_n_loo"))
        d = d.filter(pl.col("ref_n_loo") >= 2)
        own = pl.col("votes") - self_vote
        other_best = pl.when(pl.col("lv") == pl.col("ref_lv")).then(pl.col("second_votes")).otherwise(pl.col("ref_votes"))
        agree = (own > other_best) | ((own == other_best) & (own > 0) & (pl.col("lv") == pl.col("ref_lv")))
        d = d.with_columns(agree.alias("agree"))
        return (float(d["agree"].mean()) if d.height else None), d.height

    rows = []
    for lab in LABELS:
        b = pairs.filter(pl.col("bin") == lab)
        pur, ncl, ncell = purity(b)
        loo, nref = loo_agree(b)
        rows.append(dict(set=f"home-well conf {lab}", pairs=b.height, clones_ge2=ncl, cells_in_clones=ncell,
                         purity_within=pur, agree_with_hiconf_ref=loo, n_ref_scored=nref, chance=chance(b) if b.height else None))
    allp = pairs
    pur, ncl, ncell = purity(allp)
    rows.append(dict(set="home-well all", pairs=allp.height, clones_ge2=ncl, cells_in_clones=ncell, purity_within=pur,
                     agree_with_hiconf_ref=loo_agree(allp)[0], n_ref_scored=loo_agree(allp)[1], chance=chance(allp)))

    # baseline: current per-well PairPlex pairs, same wells
    base = []
    for w, fp in wells(a.pp_dir, "pair", lo, hi):
        base.append(pl.read_parquet(fp, columns=["v_gene:0", "cdr3_aa:0", "v_gene:1", "locus:0", "locus:1"]).with_columns(pl.lit(w).alias("well")))
    base = pl.concat(base).filter((pl.col("locus:0") == "IGH") & pl.col("locus:1").is_in(["IGK", "IGL"]))
    base = base.select(pl.col("v_gene:0").alias("hv"), pl.col("cdr3_aa:0").alias("hcdr3"), norm_lv(pl.col("v_gene:1")).alias("lv"),
                       pl.lit(0.0).alias("conf"))
    pur, ncl, ncell = purity(base)
    bl, bn = loo_agree(base)
    rows.append(dict(set="current PairPlex per-well", pairs=base.height, clones_ge2=ncl, cells_in_clones=ncell, purity_within=pur,
                     agree_with_hiconf_ref=bl, n_ref_scored=bn, chance=chance(base)))
    res = pl.DataFrame(rows)
    res.write_csv(f"{a.out_prefix}_purity.csv")
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(220)
    print(res.with_columns(pl.col(c).round(3) for c in ("purity_within", "agree_with_hiconf_ref", "chance")))


if __name__ == "__main__":
    main()
