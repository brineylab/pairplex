# PairPlex v2 benchmark harness

One command benchmarks any git ref of PairPlex on four frozen datasets and scores it with fixed code.

```bash
# from any pod, in ~/work/Software/pairplex (branch: diagnosis)
bash diagnosis/bench/submit.sh REF NAME ['{"kwarg": value}'] [sim_cal,sim_imp,real_d02,real_d07]
cat ~/work/PairPlex/bench/runs/NAME/STATUS          # queued / running slotN / DONE / DONE_WITH_FAILURES
/home/jovyan/work/pairplex-simplex/.venv/bin/python diagnosis/bench/compare.py NAME --baseline v2-base
```

- `REF` is a branch, tag or commit in the shared repo (local refs are visible from every worktree; `origin/REF` is tried second). The code is snapshotted with `git archive`, so only **committed** changes are tested.
- `NAME` must be new; results go to `~/work/PairPlex/bench/runs/NAME/<dataset>/{metrics.json,run.log,score.log,STATUS}`.
- Jobs run on the high-CPU pod (240 cores) in tmux. At most **2 benchmarks** run at once (slot locks); the rest queue. Each runs its datasets in parallel with 30 PairPlex workers each.
- Typical time: sims ~30 min, real subsets ~1–1.5 h.

## Datasets
| name | what | scored by |
|---|---|---|
| `sim_cal` | SimPlex v2c, 20k cells / 96 wells, measured regime (`results/simplex_20k96/calibrated`) | truth: precision, cell recall, pairs at confidence ≥0.9, drop reasons |
| `sim_imp` | same, improved protocol (leakage 0.2, ambient 0.05, more molecules) | same |
| `real_d02` | donor02 memory wells 97–192, 1 in 8 droplet barcodes (`~/fast/imm/pairplex_bench/real/`) | clonal light-chain purity (clone = heavy V + CDRH3 aa) |
| `real_d07` | donor07, same | same |

## Acceptance rule (agreed 2026-11-17), applied by `compare.py`
- sims: cell recall **or** pairs at confidence ≥0.9 go up, and precision drops by ≤1 point;
- real: clonal purity drops by ≤1 point;
- runtime: total PairPlex time ≤1.5× baseline.
- a set identical to the baseline is reported SAME and accepted (neutral by design).

## Contract with branches
1. **Entry point:** `pairplex.pairplex.run(reads, output_directory, quiet=True, **kwargs)`. New behaviour is selected through keyword arguments; `legacy=True` must reproduce `v2-base` output.
2. **Per-well outputs** (current layout): `annotated/<well>_paired.parquet` / `_unpaired.parquet`, `metadata/<well>.csv`. Cross-well modes may skip them.
3. **Final pairs (optional, recommended for anything that scores pairs):** `pairs/pairs.parquet` with at least
   `barcode, well (home well, int), name, confidence (0–1, nullable)` and, for `:0` (heavy) and `:1` (light):
   `locus, v_gene, j_gene, junction, cdr3_aa, sequence_id, sequence, umis`. This is the `pair-pooled` schema.
4. If the checkout has `pairplex.pooled_pairing.run_pooled_pairing` and per-well outputs exist, the harness also runs pooled pairing (`pooled/pooled_pairs.parquet`).
5. The harness wraps `abstar.run` so `mmseqs_threads=None` becomes 1 (mmseqs crash on >67-core nodes); an explicit count set by the branch is respected.
6. Do not modify `simplex/` in feature branches: scoring uses this checkout's copy.

## Files
- `make_real_subset.py`: builds the real subsets (one-time; `~/work/PairPlex/bench/make_real_subsets.sh`).
- `stage_run.py`: runs the code under test (branch snapshot first on `sys.path`).
- `stage_score.py`: scores outputs with harness code.
- `bench.sh` / `submit.sh`: job and queue; `compare.py`: deltas and verdict.
