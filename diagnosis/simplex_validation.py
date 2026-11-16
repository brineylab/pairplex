"""Ground-truth validation of per-well vs pooled (home-well) pairing with SimPlex v2.

For each scenario: simplex.run (calibrated profile + overrides) -> pairplex.run (real defaults:
clustering_threshold 0.9, min_cluster_reads 3, min_cluster_umis 1) -> score per-well pairs ->
pooled pairing (pairplex.pooled_pairing) -> score pooled pairs, overall and by confidence bin.
Appends one JSON line per scenario; resumable (skips scenarios with a metrics file).
Run from the repo root so the repo's pairplex/simplex are imported.

Usage: python diagnosis/simplex_validation.py OUT_DIR --input PAIRS.parquet [--n-cells 4000 --wells 48]
       [--scenarios calibrated,clean,...]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: use this checkout's pairplex/simplex

import polars as pl

import pairplex.pairplex as ppx
import simplex
from pairplex.pooled_pairing import run_pooled_pairing

SCENARIOS = {
    # measured regime
    "calibrated": {},
    # one axis at a time
    "no_ambient": dict(ambient_rate=0.0),
    "no_leakage": dict(release_rate=0.02, release_rate_heavy=None, release_rate_light=None),
    "clean": dict(release_rate=0.02, release_rate_heavy=None, release_rate_light=None, ambient_rate=0.0),
    "leak_0.5": dict(release_rate=0.5, release_rate_heavy=None, release_rate_light=None),
    # what a fixed protocol could look like: leakage 0.2, ambient 0.05, more molecules
    "improved_protocol": dict(release_rate=0.2, release_rate_heavy=None, release_rate_light=None, ambient_rate=0.05,
                              molecules_per_chain_mean=6.0, molecules_per_chain_mean_light=8.0),
}
BINS = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0001]
LABELS = ["<0.1", "0.1-0.3", "0.3-0.6", "0.6-0.9", ">=0.9"]


def by_bin(ps, conf):
    ps = ps.with_columns(pl.Series("conf", conf)).with_columns(
        pl.col("conf").cut(BINS[1:-1], labels=LABELS, left_closed=True).cast(pl.Utf8).alias("bin"))
    out = {}
    for lab in LABELS:
        b = ps.filter(pl.col("bin") == lab)
        out[lab] = {"n": b.height, "precision": (float((b["pairing_status"] == "correct").mean()) if b.height else None),
                    "mispair_rate": (float((b["pairing_status"] == "mispaired").mean()) if b.height else None)}
    return out


def run_one(name, overrides, a, out_root):
    d = out_root / name
    mfile = d / "metrics.json"
    if mfile.exists():
        print(f"SKIP {name}")
        return json.loads(mfile.read_text())
    if d.exists():
        shutil.rmtree(d)
    t0 = time.time()
    prof = {**simplex.PRINCEPS_2025, **overrides}
    reads = simplex.run(a.input, d / "sim", n_cells=a.n_cells, wells=a.wells, seed=a.seed, quiet=True, **prof)
    t_sim = time.time() - t0
    ppx.run(str(reads), str(d / "pp"), quiet=True)
    t_pp = time.time() - t0 - t_sim
    truth = d / "sim" / "truth"
    res = {"scenario": name, "overrides": overrides, "n_cells": a.n_cells, "wells": a.wells, "seed": a.seed,
           "sec_sim": round(t_sim), "sec_pairplex": round(t_pp)}
    ps, ks = simplex.score(d / "pp" / "annotated", truth, quiet=True)
    res["per_well"] = simplex.summarize(ps, ks, truth)
    pairs, info = run_pooled_pairing(d / "pp", d / "pooled_pairs.parquet")
    res["pooled_info"] = info
    # score pooled pairs: rows carry well (home well) + sequence ids/sequences
    pp_file = d / "pooled_for_scoring_paired.parquet"
    pairs.select("well", "name", "sequence_id:0", "sequence_id:1", "sequence:0", "sequence:1", "confidence").write_parquet(pp_file)
    ps2, ks2 = simplex.score(pp_file, truth, quiet=True)
    conf = pairs["confidence"].to_list()
    res["pooled"] = simplex.summarize(ps2, ks2, truth, confidence=conf)
    res["pooled_by_conf"] = by_bin(ps2, conf)
    # same reads, PairPlex with min_cluster_umis=2 (SimPlex v1 claimed this removes all mispairs)
    ppx.run(str(reads), str(d / "pp_mcu2"), min_cluster_umis=2, quiet=True)
    ps3, ks3 = simplex.score(d / "pp_mcu2" / "annotated", truth, quiet=True)
    res["per_well_mcu2"] = simplex.summarize(ps3, ks3, truth)
    for sub in ("temp", "parsed", "consensus"):
        shutil.rmtree(d / "pp_mcu2" / sub, ignore_errors=True)
    ps.write_parquet(d / "per_well_pair_scores.parquet")
    ps2.with_columns(pl.Series("confidence", conf)).write_parquet(d / "pooled_pair_scores.parquet")
    # free disk: reads + PairPlex intermediates
    shutil.rmtree(d / "sim" / "reads", ignore_errors=True)
    for sub in ("temp", "parsed", "consensus"):
        shutil.rmtree(d / "pp" / sub, ignore_errors=True)
    res["sec_total"] = round(time.time() - t0)
    mfile.write_text(json.dumps(res, indent=1, default=str))
    with open(out_root / "simplex_validation.jsonl", "a") as fh:
        fh.write(json.dumps(res, default=str) + "\n")
    pw, po = res["per_well"], res["pooled"]
    print(f"DONE {name}: per-well emitted={pw['emitted']} prec={pw['precision']} cell_recall={pw.get('cell_recall')} | "
          f"mcu2 emitted={res['per_well_mcu2']['emitted']} prec={res['per_well_mcu2']['precision']} | "
          f"pooled emitted={po['emitted']} prec={po['precision']} prec@0.9={po.get('precision_conf0.9')} "
          f"cell_recall={po.get('cell_recall')} ({res['sec_total']}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--input", required=True)
    ap.add_argument("--n-cells", type=int, default=4000)
    ap.add_argument("--wells", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    a = ap.parse_args()
    out_root = Path(a.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for name in a.scenarios.split(","):
        try:
            run_one(name, SCENARIOS[name], a, out_root)
        except Exception as e:  # keep the queue going; record the failure
            import traceback
            print(f"FAIL {name}: {e}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
