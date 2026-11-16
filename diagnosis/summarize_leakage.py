"""Aggregate per-reaction leakage (residency_dedup.py --json-out) and test the follow-up questions.

1. Does light leak more than heavy?  Paired comparison across reactions (f6_K, f6_L vs f6_H),
   sign-flip permutation test on the per-reaction differences.
2. Naive vs memory reactions: paired by donor.
3. Leakage vs ambient burden (heavy_burden.py JSONL, naive): Spearman with permutation p.
   Prediction if both come from leaky fixed cells: positive correlation.

Usage: python summarize_leakage.py LEAKAGE.jsonl --burden BURDEN.jsonl [--min-n6 50] [--out OUT.csv]
"""
import argparse
import json

import numpy as np
import polars as pl


def signflip_p(diff, n=20000, seed=0):
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    rng = np.random.default_rng(seed)
    obs = diff.mean()
    null = np.array([(diff * rng.choice([-1, 1], size=diff.size)).mean() for _ in range(n)])
    return float(obs), float((np.abs(null) >= abs(obs) - 1e-12).mean()), int(diff.size)


def spearman(x, y, n=20000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    rho = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n)])
    return float(rho), float((np.abs(null) >= abs(rho) - 1e-12).mean()), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("leakage")
    ap.add_argument("--burden", default=None)
    ap.add_argument("--min-n6", type=int, default=50, help="min species (>=6 UMIs) per chain to use its f")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = pl.DataFrame([json.loads(l) for l in open(a.leakage) if l.strip()])
    d = d.with_columns(pl.col("label").str.split("_").list.get(0).alias("donor"),
                       pl.col("label").str.split("_").list.get(1).alias("reaction"))
    for c in ("H", "K", "L", "all"):
        d = d.with_columns(pl.when(pl.col(f"n6_{c}") >= a.min_n6).then(pl.col(f"f6_{c}")).otherwise(None).alias(f"f_{c}"))
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(200)
    print(d.select("donor", "reaction", "W", "n6_all", *[pl.col(f"f_{c}").round(3) for c in ("all", "H", "K", "L")]).sort("reaction", "donor"))
    print("\nmedian leakage by reaction:")
    print(d.group_by("reaction").agg(*[pl.col(f"f_{c}").median().round(3) for c in ("all", "H", "K", "L")], pl.len()))

    print("\n1) light vs heavy (per-reaction differences, sign-flip permutation):")
    for c in ("K", "L"):
        m, p, n = signflip_p((d[f"f_{c}"] - d["f_H"]).to_numpy())
        print(f"   f_{c} - f_H: mean {m:+.3f}  p={p:.4f}  n={n} reactions")
    w = d.pivot(index="donor", on="reaction", values="f_all")
    if {"naive", "memory"} <= set(w.columns):
        m, p, n = signflip_p((w["memory"] - w["naive"]).to_numpy())
        print(f"\n2) memory - naive (paired by donor): mean {m:+.3f}  p={p:.4f}  n={n}")
    if a.burden:
        b = pl.DataFrame([json.loads(l) for l in open(a.burden) if l.strip()])
        j = d.filter(pl.col("reaction") == "naive").join(b, on="donor")
        print("\n3) naive leakage vs ambient burden (Spearman, permutation p):")
        for y in ("sat_frac", "occ_frac_ge5", "occ_frac_ge10"):
            for x in ("f_all", "f_H", "f_K", "f_L"):
                rho, p, n = spearman(j[x].to_numpy(), j[y].to_numpy())
                print(f"   {y:14s} ~ {x:6s} rho={rho:+.2f} p={p:.3f} n={n}")
    if a.out:
        d.write_csv(a.out)


if __name__ == "__main__":
    main()
