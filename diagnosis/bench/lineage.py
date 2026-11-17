"""Lineage retention on real memory reactions: does a new version still recover the clonally
expanded lineages that the baseline found?

Clone = heavy V gene + heavy J gene (alleles stripped) + CDRH3 aa, within one reaction.
Cell = unique droplet barcode carrying that heavy chain (leaked copies of one cell in several wells
count once).

Expanded clones are defined on the BASELINE pairs (same pair set; a branch's "pairs" set is compared
with the baseline "pooled" set). For clone-size cutoffs of 3, 5 and 10 cells:
  * base_clones / base_cells: expanded clones and their cells in the baseline;
  * cells_recovered: cells of those clones present in the new set (any light chain);
  * clones_lost: baseline expanded clones with 0 cells in the new set (listed for review);
  * coherence: share of a clone's cells carrying its modal light V in the new set, for those clones
    (light-chain coherence restricted to expanded lineages).

Verdict (2026-11-17): any lost lineage of >= 5 cells -> REVIEW (listed; not an automatic fail, because
baseline "expanded clones" include ambient pseudo-lineages - one heavy chain under many barcodes with
random lights - that a correct ambient filter should remove). The cells-recovered ratio is reported
only until its normal range is known.

Usage: python lineage.py NEW_RUN [--baseline v2-base] [--datasets real_d02,real_d07]
"""
import argparse
import json
from pathlib import Path

import polars as pl

RUNS = Path("/home/jovyan/work/PairPlex/bench/runs")
SIZES = (3, 5, 10)
COLS = {"barcode": None, "v_gene:0": None, "j_gene:0": None, "cdr3_aa:0": None, "v_gene:1": None, "locus:0": None}


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


def _gene(expr):
    return expr.str.split("*").list.first()


def load_set(run: str, ds: str, which: str) -> pl.DataFrame | None:
    """Pairs of one set as (barcode, clone, lv); heavy chain in :0."""
    root = RUNS / run / ds / "pp"
    if which == "per_well":
        fs = sorted((root / "annotated").glob("*_paired.parquet")) if (root / "annotated").exists() else []
        cols = ["name", "v_gene:0", "j_gene:0", "cdr3_aa:0", "v_gene:1", "locus:0"]
        fs = readable_pairs(fs, cols)
        if not fs:
            return None
        df = pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed").rename({"name": "barcode"})
    else:
        f = root / ("pooled/pooled_pairs.parquet" if which == "pooled" else "pairs/pairs.parquet")
        if not f.exists():
            return None
        df = pl.read_parquet(f)
        cols = [c for c in ("barcode", "name", "v_gene:0", "j_gene:0", "cdr3_aa:0", "v_gene:1", "locus:0",
                            "v_gene:1", "j_gene:1", "cdr3_aa:1", "locus:1") if c in df.columns]
        df = df.select(list(dict.fromkeys(cols)))
        if "barcode" not in df.columns:
            df = df.with_columns(pl.col("name").str.split("_").list.first().alias("barcode"))
        if "locus:1" in df.columns:  # orient heavy-first
            swap = df.filter(pl.col("locus:0") != "IGH")
            if swap.height:
                ren = {c: c[:-2] + (":1" if c.endswith(":0") else ":0") for c in df.columns if c[-2:] in (":0", ":1")}
                df = pl.concat([df.filter(pl.col("locus:0") == "IGH"), swap.rename(ren).select(df.columns)])
    df = df.filter(pl.col("locus:0") == "IGH") if "locus:0" in df.columns else df
    return df.select(
        pl.col("barcode").str.split("_").list.first().alias("barcode"),
        pl.concat_str([_gene(pl.col("v_gene:0")), _gene(pl.col("j_gene:0")), pl.col("cdr3_aa:0")], separator="|").alias("clone"),
        _gene(pl.col("v_gene:1")).str.replace(r"^(IG[KL]V\d+)D-", "${1}-").alias("lv"),
    ).drop_nulls(["barcode", "clone"])


def retention(base: pl.DataFrame, new: pl.DataFrame) -> dict:
    bcells = base.select("barcode", "clone").unique()
    sizes = bcells.group_by("clone").len().rename({"len": "size"})
    ncells = new.select("barcode", "clone").unique()
    rec = ncells.join(sizes, on="clone").group_by("clone").len().rename({"len": "recovered"})
    out = {}
    for k in SIZES:
        cl = sizes.filter(pl.col("size") >= k).join(rec, on="clone", how="left").with_columns(pl.col("recovered").fill_null(0))
        lost = cl.filter(pl.col("recovered") == 0).sort("size", descending=True)
        # coherence of the new set within these clones
        sub = new.filter(pl.col("clone").is_in(cl["clone"].implode())).drop_nulls("lv").unique(["barcode", "clone", "lv"])
        c = sub.group_by("clone", "lv").len().group_by("clone").agg(pl.col("len").sum().alias("n"), pl.col("len").max().alias("top"))
        c = c.filter(pl.col("n") >= 2)
        out[f">={k}"] = {
            "base_clones": cl.height, "base_cells": int(cl["size"].sum()),
            "cells_recovered": int(cl["recovered"].clip(upper_bound=cl["size"]).sum()),
            "cells_ratio": round(float(cl["recovered"].clip(upper_bound=cl["size"]).sum() / max(1, cl["size"].sum())), 4),
            "clones_lost": lost.height,
            "lost_examples": lost.head(10).rows(),
            "coherence": round(float(c["top"].sum() / c["n"].sum()), 4) if c.height else None,
        }
    return out


def compare_lineages(new_run: str, base_run: str = "v2-base", datasets=("real_d02", "real_d07")) -> dict:
    res = {}
    for ds in datasets:
        for s in ("per_well", "pooled", "pairs"):
            new = load_set(new_run, ds, s)
            if new is None:
                continue
            bs = s if load_set(base_run, ds, s) is not None else ("pooled" if s == "pairs" else None)
            base = load_set(base_run, ds, bs) if bs else None
            if base is None:
                continue
            r = retention(base, new)
            r["base_coherence_ref"] = retention(base, base)  # same metric on the baseline itself
            r["vs"] = bs
            r["ok"] = r[">=5"]["clones_lost"] == 0
            res[f"{ds}/{s}"] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new_run")
    ap.add_argument("--baseline", default="v2-base")
    ap.add_argument("--datasets", default="real_d02,real_d07")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = compare_lineages(a.new_run, a.baseline, a.datasets.split(","))
    if a.json:
        print(json.dumps(res, indent=1, default=str))
        return
    for key, r in res.items():
        print(f"\n[{'PASS' if r['ok'] else 'REVIEW'}] lineages {key} (vs baseline {r['vs']})")
        for k in (f">={s}" for s in SIZES):
            x, b = r[k], r["base_coherence_ref"][k]
            print(f"   clones {k:<4} base {x['base_clones']:>5} clones / {x['base_cells']:>6} cells | recovered {x['cells_recovered']:>6} "
                  f"({x['cells_ratio']:.3f}) | lost {x['clones_lost']:>3} | coherence {b['coherence']} -> {x['coherence']}")
        if r[">=5"]["clones_lost"]:
            print("   lost lineages (>=5 cells), clone | size | recovered:", r[">=5"]["lost_examples"])


if __name__ == "__main__":
    main()
