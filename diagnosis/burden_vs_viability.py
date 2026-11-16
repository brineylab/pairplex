"""Correlate per-donor ambient burden (heavy_burden.py JSONL) with sample viability.

Prediction of the ambient-mRNA hypothesis (satellites.py): free mRNA comes from dead/lysed
cells, so burden should rise as viability falls. Viability columns are read from the study
sheet (Sheet1, rows 4-15, one row per donor): PBMC viability (J), naive B viability (M),
memory B viability (P); naive cells loaded post-fixation (Q); lab user (C) and processing date (E)
are tested as processing-group effects. Spearman rho with an exact-ish
permutation p-value (n = 12 donors, so treat as exploratory). Depth (total heavy occurrences)
and cells loaded are reported as possible confounders.

Usage: python burden_vs_viability.py BURDEN.jsonl STUDY.xlsx [--out OUT.csv]
"""
import argparse
import json

import numpy as np
import openpyxl
import polars as pl


def spearman(x, y, n_perm=20000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    rho = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)])
    return rho, float((np.abs(null) >= abs(rho) - 1e-12).mean()), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("burden")
    ap.add_argument("xlsx")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    b = pl.DataFrame([json.loads(l) for l in open(a.burden) if l.strip()])
    b = b.with_columns(pl.col("donor").str.extract(r"(\d+)").cast(pl.Int64).alias("donor_n"))

    ws = openpyxl.load_workbook(a.xlsx, data_only=True)["Sheet1"]
    rows = []
    for r in range(4, 16):
        v = lambda c: ws[f"{c}{r}"].value
        rows.append(dict(donor_n=int(v("B")), donor_id=v("D"), user=v("C"), date=str(v("E"))[:10], pbmc_viability=v("J"), naive_viability=v("M"),
                         memory_viability=v("P"), naive_loaded=v("Q")))
    s = pl.DataFrame(rows, schema_overrides={"naive_loaded": pl.Float64, "pbmc_viability": pl.Float64,
                                             "naive_viability": pl.Float64, "memory_viability": pl.Float64})
    d = b.join(s, on="donor_n", how="left").sort("donor_n")
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(200)
    print(d.select("donor", "pbmc_viability", "naive_viability", "naive_loaded", "heavy_occurrences_total",
                   pl.col("sat_frac").round(4), pl.col("occ_frac_ge5").round(4), pl.col("occ_frac_ge10").round(4)))
    print("\nSpearman rho (permutation p), n donors:")
    for y in ("sat_frac", "occ_frac_ge5", "occ_frac_ge10"):
        for x in ("pbmc_viability", "naive_viability", "naive_loaded", "heavy_occurrences_total"):
            rho, p, n = spearman(d[x].to_numpy(), d[y].to_numpy())
            print(f"  {y:14s} ~ {x:24s} rho={rho:+.2f}  p={p:.3f}  n={n}")
    print("\nprocessing-group differences (median difference, permutation p; exploratory, uncorrected):")
    rng = np.random.default_rng(0)
    for y in ("sat_frac", "occ_frac_ge5", "occ_frac_ge10"):
        vals, grp = d[y].to_numpy(), d["user"].to_numpy()
        g0 = grp[0]
        stat = lambda g: np.median(vals[g == g0]) - np.median(vals[g != g0])
        obs = stat(grp)
        null = np.array([stat(rng.permutation(grp)) for _ in range(20000)])
        print(f"  {y:14s} by lab user ({g0} - other) = {obs:+.3f}  p={(np.abs(null) >= abs(obs) - 1e-12).mean():.3f}")
    print(d.group_by("date").agg(pl.len(), *[pl.col(y).median() for y in ("sat_frac", "occ_frac_ge5", "occ_frac_ge10")]).sort("date"))
    if a.out:
        d.write_csv(a.out)


if __name__ == "__main__":
    main()
