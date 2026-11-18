"""Benchmark stage 2: score one dataset's outputs with the fixed harness code (this checkout).

Simulated datasets (SimPlex truth available): precision, cell recall, high-confidence pairs and
drop reasons for per-well pairs, pooled pairs and branch pairs (out/pairs/pairs.parquet).
Real datasets: clonal light-chain purity (memory clones = heavy V + CDRH3 aa) for the same sets.

Usage: python stage_score.py OUT --kind sim|real [--truth TRUTH_DIR] --metrics METRICS_JSON
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # harness checkout: simplex for scoring

BINS = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0001]
LABELS = ["<0.1", "0.1-0.3", "0.3-0.6", "0.6-0.9", ">=0.9"]
SCORE_COLS = ["well", "name", "sequence_id:0", "sequence_id:1", "sequence:0", "sequence:1"]


def readable_pairs(files, cols):
    """Paired files that actually hold rows with ``cols``. Wells without pairs can write column-less or
    empty parquet files (> 100 bytes because of the footer); skip them instead of crashing."""
    out = []
    for f in files:
        try:
            schema = pl.read_parquet_schema(f)
        except Exception:
            continue
        if all(c in schema for c in cols) and pl.scan_parquet(f).select(pl.len()).collect().item() > 0:
            out.append(f)
    return out


def norm_lv(expr):
    return expr.str.replace(r"^(IG[KL]V\d+)D-", "${1}-")


def purity(df: pl.DataFrame) -> dict:
    """Cells-weighted share of a clone's cells carrying its modal light V; clones >= 2 cells."""
    d = df.select(pl.col("v_gene:0").alias("hv"), pl.col("cdr3_aa:0").alias("hcdr3"), norm_lv(pl.col("v_gene:1")).alias("lv")).drop_nulls()
    c = d.group_by("hv", "hcdr3", "lv").len()
    cl = c.group_by("hv", "hcdr3").agg(pl.col("len").sum().alias("size"), pl.col("len").max().alias("top")).filter(pl.col("size") >= 2)
    if cl.height == 0:
        return {"pairs": df.height, "clones_ge2": 0, "purity": None}
    p = d["lv"].value_counts()["count"].to_numpy().astype(float)
    p /= p.sum()
    return {"pairs": df.height, "clones_ge2": cl.height, "cells_in_clones": int(cl["size"].sum()),
            "purity": round(float(cl["top"].sum() / cl["size"].sum()), 4), "chance": round(float((p ** 2).sum()), 4)}


def heavy_first(df: pl.DataFrame) -> pl.DataFrame:
    """Orient pairs so :0 is the heavy chain (per-well paired files are already heavy-first)."""
    if "locus:0" not in df.columns:
        return df
    sw = df.filter(pl.col("locus:0") != "IGH")
    if sw.height == 0:
        return df
    ren = {}
    for c in df.columns:
        if c.endswith(":0"):
            ren[c] = c[:-2] + ":1"
        elif c.endswith(":1"):
            ren[c] = c[:-2] + ":0"
    return pl.concat([df.filter(pl.col("locus:0") == "IGH"), sw.rename(ren).select(df.columns)], how="vertical_relaxed")


def per_well_pairs(out: Path) -> pl.DataFrame | None:
    fs = sorted((out / "annotated").glob("*_paired.parquet")) if (out / "annotated").exists() else []
    cols = ["name", "sequence_id:0", "sequence_id:1", "v_gene:0", "v_gene:1", "cdr3_aa:0", "locus:0", "locus:1"]
    fs = readable_pairs(fs, cols)
    if fs:  # sibling-agreement columns when present
        extra = [c for c in ("junction:0", "sequence:0", "junction:1", "sequence:1") if all(c in pl.read_parquet_schema(f) for f in fs)]
        cols = cols + extra
    if not fs:
        return None
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def conf_bins(df: pl.DataFrame, fn) -> dict:
    d = df.with_columns(pl.col("confidence").cut(BINS[1:-1], labels=LABELS, left_closed=True).cast(pl.Utf8).alias("bin"))
    return {lab: fn(d.filter(pl.col("bin") == lab)) for lab in LABELS}


def score_sim(out: Path, truth: Path) -> dict:
    import simplex

    res = {}
    ann = out / "annotated"
    if per_well_pairs(out) is not None:
        ps, ks = simplex.score(ann, truth, quiet=True)
        res["per_well"] = simplex.summarize(ps, ks, truth)
        csv = out / "drop_reasons.csv"
        sim_link = out / "sim"
        if not sim_link.exists():
            sim_link.mkdir()
            (sim_link / "truth").symlink_to(truth.resolve())
        pp_link = out / "pp"
        if not pp_link.exists():
            pp_link.mkdir()
            (pp_link / "annotated").symlink_to(ann.resolve())
        subprocess.run([sys.executable, str(HERE.parent / "simplex_drop_reasons.py"), str(out), str(csv)],
                       check=True, capture_output=True)
        res["drop_reasons"] = {r["reason"]: r["frac"] for r in pl.read_csv(csv).iter_rows(named=True)}
    for key, f in (("pooled", out / "pooled" / "pooled_pairs.parquet"), ("pairs", out / "pairs" / "pairs.parquet")):
        if not f.exists():
            continue
        p = pl.read_parquet(f)
        tmp = f.with_name(f.stem + "_for_scoring_paired.parquet")
        p.select(SCORE_COLS + ["confidence"]).write_parquet(tmp)
        ps, ks = simplex.score(tmp, truth, quiet=True)
        conf = p["confidence"].to_list()
        res[key] = simplex.summarize(ps, ks, truth, confidence=conf)
        sc = ps.with_columns(pl.Series("confidence", conf))
        res[key]["n_conf0.9"] = int((sc["confidence"] >= 0.9).sum())
        res[key + "_by_conf"] = conf_bins(sc, lambda b: {"n": b.height, "precision": (round(float((b["pairing_status"] == "correct").mean()), 4) if b.height else None)})
    return res


def real_metrics(df: pl.DataFrame) -> dict:
    """Clonal purity plus sibling-agreement precision and the estimated number of correct pairs."""
    from sibling import sibling_precision

    r = purity(df)
    sib = sibling_precision(df)
    r.update({"sib_" + k: v for k, v in sib.items() if k != "est_correct"})
    r["est_correct"] = sib.get("est_correct")
    return r


def score_real(out: Path) -> dict:
    res = {}
    pw = per_well_pairs(out)
    if pw is not None:
        res["per_well"] = real_metrics(heavy_first(pw))
    for key, f in (("pooled", out / "pooled" / "pooled_pairs.parquet"), ("pairs", out / "pairs" / "pairs.parquet")):
        if not f.exists():
            continue
        p = heavy_first(pl.read_parquet(f))
        res[key] = real_metrics(p)
        res[key + "_by_conf"] = conf_bins(p, purity)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--kind", choices=["sim", "real"], required=True)
    ap.add_argument("--truth")
    ap.add_argument("--metrics", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    res = score_sim(out, Path(a.truth)) if a.kind == "sim" else score_real(out)
    run_info = out / "stage_run.json"
    if run_info.exists():
        res["run"] = json.loads(run_info.read_text())
    Path(a.metrics).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
