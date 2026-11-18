"""Compare a benchmark run with a baseline run and apply the acceptance rule.

Acceptance (agreed 2026-11-17), per comparable pair set:
  * simulations (sim_cal, sim_imp): cell recall OR pairs at confidence >= 0.9 go up, and precision
    does not drop by more than 1 point;
  * real memory reactions (real_d02, real_d07): clonal purity does not drop by more than 1 point;
  * runtime: total PairPlex seconds <= 1.5 x baseline.
A set whose metrics are identical to the baseline is reported SAME and accepted (changes that are
neutral by design, e.g. extra output columns).
Noise band (agreed 2026-11-18): v2-base is not deterministic (as_completed chunk order -> clustering/POA
tie-breaks), so a set whose every change stays within the measured run-to-run noise is reported SAME~
and accepted. Measured on v2-base reruns (C-v2base-rerun-sim, main-a8b6376, C-support-qc-v1 vs v2-base):
sim per-well sets were exactly identical (no band: any change there is real); sim pooled sets moved by
<= 0.0013 precision, <= 0.0009 recall, <= 2 emitted, 0 pairs at conf >= 0.9; real sets by <= 0.0008
purity and <= 12 pairs (0.02%). Band = about 1.5-2.5x those maxima (NOISE below). A branch set "pairs" (out/pairs/pairs.parquet) is compared with the baseline "pooled" set when the
baseline has no "pairs".

Usage: python compare.py NAME [--baseline v2-base] [--json]
"""
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
RUNS = Path("/home/jovyan/work/PairPlex/bench/runs")
SETS = ("per_well", "pooled", "pairs")
TOL = 0.01
# run-to-run noise band (see docstring); sim per-well sets are deterministic and get no band
NOISE = {"precision": 0.002, "cell_recall": 0.002, "emitted_rel": 0.001, "emitted_min": 2,
         "n_conf_rel": 0.001, "n_conf_min": 1, "purity": 0.002, "pairs_rel": 0.0005, "pairs_min": 10}


def within(cur, ref, key, abs_tol=None, rel=None, minimum=0):
    if cur is None or ref is None:
        return cur == ref
    tol = abs_tol if abs_tol is not None else max(minimum, rel * abs(ref))
    return abs(cur - ref) <= tol + 1e-12


def sim_noise(cur, ref):
    return (within(cur["precision"], ref["precision"], "p", NOISE["precision"])
            and within(cur["cell_recall"], ref["cell_recall"], "r", NOISE["cell_recall"])
            and within(cur["emitted"], ref["emitted"], "e", rel=NOISE["emitted_rel"], minimum=NOISE["emitted_min"])
            and within(cur["n_conf0.9"], ref["n_conf0.9"], "c", rel=NOISE["n_conf_rel"], minimum=NOISE["n_conf_min"]))


def real_noise(cur, ref):
    return (within(cur["purity"], ref["purity"], "u", NOISE["purity"])
            and within(cur["pairs"], ref["pairs"], "n", rel=NOISE["pairs_rel"], minimum=NOISE["pairs_min"]))


def load(name):
    d = RUNS / name
    out = {}
    for m in sorted(d.glob("*/metrics.json")):
        out[m.parent.name] = json.loads(m.read_text())
    return out


def g(x, *ks):
    for k in ks:
        if not isinstance(x, dict) or x.get(k) is None:
            return None
        x = x[k]
    return x


def fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--baseline", default="v2-base")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--pairs-vs", default="pooled", choices=["pooled", "per_well"],
                    help="baseline set to compare a branch 'pairs' set with (per_well for branches whose pairs/ are per-well pairs, e.g. scored pairing)")
    a = ap.parse_args()
    new, base = load(a.name), load(a.baseline)
    rows, verdicts = [], []
    for ds in sorted(set(new) | set(base)):
        n, b = new.get(ds, {}), base.get(ds, {})
        for s in SETS:
            bs = s if s in b else (a.pairs_vs if s == "pairs" else None)
            if s not in n:
                continue
            if ds.startswith("sim"):
                cur = {"precision": g(n, s, "precision"), "cell_recall": g(n, s, "cell_recall"),
                       "emitted": g(n, s, "emitted"), "n_conf0.9": g(n, s, "n_conf0.9"), "prec_conf0.9": g(n, s, "precision_conf0.9")}
                ref = {k: (g(b, bs, k2) if bs else None) for k, k2 in
                       (("precision", "precision"), ("cell_recall", "cell_recall"), ("emitted", "emitted"),
                        ("n_conf0.9", "n_conf0.9"), ("prec_conf0.9", "precision_conf0.9"))}
                ok = None
                if ref["precision"] is not None and cur["precision"] is not None:
                    gain = (cur["cell_recall"] or 0) > (ref["cell_recall"] or 0) or (cur["n_conf0.9"] or 0) > (ref["n_conf0.9"] or 0)
                    same = all(cur[k] == ref[k] for k in ("precision", "cell_recall", "emitted", "n_conf0.9"))
                    ok = "same" if same else ("noise" if s == "pooled" and sim_noise(cur, ref) else
                                              (gain and cur["precision"] >= ref["precision"] - TOL))
            else:
                cur = {"purity": g(n, s, "purity"), "pairs": g(n, s, "pairs"), "clones_ge2": g(n, s, "clones_ge2"),
                       "purity_conf0.9": g(n, s + "_by_conf", ">=0.9", "purity"), "n_conf0.9": g(n, s + "_by_conf", ">=0.9", "pairs")}
                ref = {"purity": g(b, bs, "purity") if bs else None, "pairs": g(b, bs, "pairs") if bs else None,
                       "clones_ge2": g(b, bs, "clones_ge2") if bs else None,
                       "purity_conf0.9": g(b, bs + "_by_conf", ">=0.9", "purity") if bs else None,
                       "n_conf0.9": g(b, bs + "_by_conf", ">=0.9", "pairs") if bs else None}
                ok = None if ref["purity"] is None or cur["purity"] is None else (
                    "same" if (cur["purity"], cur["pairs"]) == (ref["purity"], ref["pairs"]) else
                    "noise" if real_noise(cur, ref) else cur["purity"] >= ref["purity"] - TOL)
            rows.append((ds, s, bs, cur, ref, ok))
            if ok is not None:
                verdicts.append(ok)
    t_new = sum(g(v, "run", "sec_pairplex") or 0 for v in new.values())
    t_base = sum(g(v, "run", "sec_pairplex") or 0 for v in base.values())
    rt_ok = None if not t_base else t_new <= 1.5 * t_base
    if a.json:
        print(json.dumps({"rows": [dict(dataset=r[0], set=r[1], vs=r[2], new=r[3], base=r[4], ok=r[5]) for r in rows],
                          "runtime": {"new_s": t_new, "base_s": t_base, "ok": rt_ok}}, indent=1))
        return
    print(f"{a.name} vs {a.baseline}")
    for ds, s, bs, cur, ref, ok in rows:
        mark = {True: "PASS", False: "FAIL", None: "n/a", "same": "SAME", "noise": "SAME~"}[ok]
        print(f"\n[{mark}] {ds} / {s}" + (f" (vs baseline {bs})" if bs != s else ""))
        for k in cur:
            print(f"   {k:<15} {fmt(ref.get(k)):>10} -> {fmt(cur[k]):>10}")
    print(f"\nruntime: {t_base}s -> {t_new}s ({'PASS' if rt_ok else 'FAIL' if rt_ok is False else 'n/a'}, limit 1.5x)")
    overall = all(v in (True, "same", "noise") for v in verdicts) and rt_ok is not False and bool(verdicts)
    # lineage retention on real memory reactions (report + REVIEW flag, not part of the pass rule)
    try:
        from lineage import SIZES, compare_lineages
        lin = compare_lineages(a.name, a.baseline, [d for d in ("real_d02", "real_d07") if d in new and d in base], a.pairs_vs)
    except Exception as e:  # never let the report break the verdict
        lin, SIZES = {}, ()
        print(f"\n(lineage report failed: {e})")
    review = []
    for key, r in lin.items():
        print(f"\n[{'PASS' if r['ok'] else 'REVIEW'}] lineages {key} (vs baseline {r['vs']})")
        for k in (f">={s}" for s in SIZES):
            x, b = r[k], r["base_coherence_ref"][k]
            print(f"   clones {k:<4} {x['base_clones']:>5} clones / {x['base_cells']:>6} cells | recovered {x['cells_ratio']:.3f} "
                  f"| lost {x['clones_lost']:>3} | coherence {b['coherence']} -> {x['coherence']}")
        if not r["ok"]:
            review.append(key)
            print("   lost lineages (>=5 cells) [clone, size, recovered]:", r[">=5"]["lost_examples"])
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}  (a set passes on its own rule; every scored set must pass)"
          + (f"  | LINEAGE REVIEW: {', '.join(review)}" if review else ""))


if __name__ == "__main__":
    main()
