"""Summary metrics on top of `simplex.score` output.

Key-level recall (singleton, reference-pairable, resident-correct) is the v1 definition, but it
collapses when leakage is high (few cells keep >= REF_MIN support at home). `cell_recall` is the
leakage-robust alternative: distinct source pairs recovered by at least one *correct, uniquely
resolved* emitted pair anywhere, over cells whose two chains both survived to amplification.
"""
from pathlib import Path

import polars as pl


def summarize(pair_scores, key_scores, truth_dir=None, confidence=None):
    """Return a dict of precision / mispair rate / recalls.

    `confidence` (optional): a Series aligned with `pair_scores` rows; adds metrics for the
    subset with confidence >= 0.5 and >= 0.9 (for pooled/home-well pairing outputs).
    """
    ps, ks = pair_scores, key_scores
    n = ps.height
    out = {"emitted": n}
    for s in ("correct", "mispaired", "unmatchable", "ambiguous"):
        out[f"n_{s}"] = int((ps["pairing_status"] == s).sum()) if n else 0
    out["precision"] = out["n_correct"] / n if n else None
    out["mispair_rate"] = out["n_mispaired"] / n if n else None
    ck = {(r["well"], r["barcode"]) for r in ps.filter(
        (pl.col("pairing_status") == "correct") & (pl.col("origin_status") == "resident")
        & (pl.col("output_status") == "unique") & (pl.col("key_status") == "singleton")).to_dicts()} if n else set()
    ref = ks.filter(pl.col("reference_pairable_both") & (pl.col("key_status") == "singleton"))
    out["key_recall"] = (sum((r["well"], r["barcode"]) in ck for r in ref.to_dicts()) / ref.height) if ref.height else None
    out["n_reference_pairable_singleton_keys"] = ref.height
    if truth_dir is not None:
        tc = pl.read_parquet(Path(truth_dir) / "truth_cells.parquet")
        denom = tc.filter(pl.col("survived_0") & pl.col("survived_1"))["source_pair_id"].n_unique()
        rec = ps.filter((pl.col("pairing_status") == "correct") & pl.col("resolved_source").is_not_null())["resolved_source"].n_unique() if n else 0
        out["cell_recall"] = rec / denom if denom else None
        out["n_cells_survived_both"] = denom
    if confidence is not None and n:
        c = pl.Series(confidence)
        for t in (0.5, 0.9):
            sub = ps.filter(c >= t)
            out[f"emitted_conf{t}"] = sub.height
            out[f"precision_conf{t}"] = (int((sub["pairing_status"] == "correct").sum()) / sub.height) if sub.height else None
    return out


__all__ = ["summarize"]
