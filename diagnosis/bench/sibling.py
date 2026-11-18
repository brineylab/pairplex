"""Model-free pairing precision on real data from clonal siblings (from diagnosis/simplex_v2d/sibling_agreement.py).

Two pairs from different barcodes whose heavies share V gene + CDR3 aa but differ by 5-30 nt of V(D)J are distinct
cells of one lineage. If both pairs are correct their light V agrees with probability S_SAME (~0.9); if either is
wrong, at chance c (sum of squared light-V frequencies). So agreement a = p^2 S + (1 - p^2) c, and
    p = sqrt((a - c) / (S - c)).
Validated on SimPlex v2d against truth (within 0.02-0.05). It measures pairs whose heavy belongs to an expanded
lineage and is extrapolated to the whole set: est_correct = pairs x p.
"""
import edlib
import numpy as np
import polars as pl

S_SAME = 0.9
MAX_PER_GROUP = 40
COLS = ["sequence_id:0", "v_gene:0", "cdr3_aa:0", "junction:0", "sequence:0", "v_gene:1"]


def _vdj(s, j):
    i = s.find(j) if s and j else -1
    return s[: i + len(j) + 30] if i >= 0 else None


def sibling_precision(df: pl.DataFrame, lo: int = 5, hi: int = 30) -> dict:
    """``df``: heavy-first pairs with COLS (and ``barcode`` or barcode-prefixed ``sequence_id:0``)."""
    if df is None or df.height == 0 or any(c not in df.columns for c in COLS):
        return {"p_est": None}
    bc = pl.col("barcode") if "barcode" in df.columns else pl.col("sequence_id:0").str.split("_").list.first()
    d = (df.select(bc.alias("bc"), "v_gene:0", "cdr3_aa:0",
                   pl.col("v_gene:1").str.split("*").list.first().str.replace(r"^(IG[KL]V\d+)D-", "${1}-").alias("lv"),
                   pl.struct("sequence:0", "junction:0").map_elements(lambda r: _vdj(r["sequence:0"], r["junction:0"]),
                                                                       return_dtype=pl.Utf8).alias("vdj"))
         .drop_nulls())
    if d.height == 0:
        return {"p_est": None}
    p = d["lv"].value_counts()["count"].to_numpy().astype(float)
    p /= p.sum()
    chance = float((p ** 2).sum())
    d = d.filter(pl.len().over("v_gene:0", "cdr3_aa:0") >= 2)
    rng = np.random.default_rng(0)
    agree = []
    for _, g in d.sort("v_gene:0", "cdr3_aa:0", "bc", "vdj").group_by("v_gene:0", "cdr3_aa:0", maintain_order=True):
        r = g.select("bc", "vdj", "lv").rows()
        if len(r) > MAX_PER_GROUP:
            r = [r[k] for k in sorted(rng.choice(len(r), MAX_PER_GROUP, replace=False))]
        for a in range(len(r)):
            for b in range(a + 1, len(r)):
                if r[a][0] == r[b][0]:
                    continue
                x, y = sorted((r[a][1], r[b][1]), key=len)
                e = edlib.align(x, y, mode="HW", task="distance")["editDistance"]
                if lo <= e <= hi:
                    agree.append(r[a][2] == r[b][2])
    if not agree:
        return {"p_est": None, "n_sibling_pairs": 0, "chance": round(chance, 4)}
    a = float(np.mean(agree))
    pe = float(np.sqrt(max(a - chance, 0.0) / (S_SAME - chance)))
    return {"p_est": round(min(pe, 1.0), 4), "n_sibling_pairs": len(agree), "agreement": round(a, 4), "chance": round(chance, 4),
            "est_correct": int(round(min(pe, 1.0) * df.height))}
