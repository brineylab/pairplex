# SimPlex — simulated PairPlex

**SimPlex generates mechanism-faithful synthetic "raw sequencing" data from real paired
antibody sequences, with known ground truth, so you can run it through PairPlex and *measure*
how well PairPlex recovers the correct heavy/light pairs — and choose pairing thresholds on a
precision/yield basis instead of guessing.**

It is a research / method-development tool. It does not change or wrap the `pairplex` runtime;
it sits beside it (`import simplex`).

---

## Table of contents

1. [Why SimPlex exists](#1-why-simplex-exists)
2. [What biological process it mimics](#2-what-biological-process-it-mimics)
3. [How it works (the pipeline)](#3-how-it-works-the-pipeline)
4. [Install & quickstart](#4-install--quickstart)
5. [`simplex.run()` — the knobs](#5-simplexrun--the-knobs)
6. [Ground-truth outputs](#6-ground-truth-outputs)
7. [Scoring: `simplex.score()`](#7-scoring-simplexscore)
8. [The end-to-end workflow](#8-the-end-to-end-workflow)
9. [How to read the results (interpretation guide)](#9-how-to-read-the-results-interpretation-guide)
10. [What the tests demonstrate, and why](#10-what-the-tests-demonstrate-and-why)
11. [Scope & limitations](#11-scope--limitations)

---

## 1. Why SimPlex exists

PairPlex sequences natively paired antibodies by **massively overloading** 10x reactions
(combinatorial barcoding, UDA-seq–inspired). At scale, some heavy and light chains get **wrongly
paired**. Investigation established that PairPlex's core pairing logic is correct on clean data,
and that mispairing is driven by **contamination (ambient molecules, barcode collisions,
cross-well hopping) meeting PairPlex's permissive default filters**.

That leaves a practical question: **what filter thresholds should you actually use?** You cannot
answer it from real data alone, because real data has no label telling you which emitted pairs
are wrong. And you cannot answer it by hand-injecting the defect you already suspect — the sweep
would just rediscover your assumption (circular).

SimPlex answers it by simulating the **physical wet-lab mechanism** faithfully, emitting
**ground truth** (which chain really came from which cell), and providing a **scorer** that
measures, for any PairPlex configuration, how many emitted pairs are correct, how many are
mispaired, and how many real cells were lost. You sweep thresholds and read off the
precision/yield trade-off.

> **Guiding principle:** simulate the *mechanism*, not the *conclusion you expect*.

---

## 2. What biological process it mimics

The PairPlex wet lab, as modelled:

1. **Bulk fixation + overloaded RT barcoding.** Cells are fixed in bulk; the 10x cell barcode is
   added by reverse transcription inside **overloaded GEMs**, so **many cells share the same 10x
   barcode**. (In standard 10x, one GEM ≈ one cell; here it is intentionally many-to-one.)
2. **Whole-cell redistribution into wells.** GEMs are broken and the **whole fixed cells** are
   randomly distributed across a 96-well plate. Because whole cells move (not free molecules), a
   cell's heavy and light stay together, in one well, under one barcode.
3. **Per-well indexing + sequencing.** A second-stage (Illumina) index is added per well; wells
   are demultiplexed upstream, giving **one FASTA/Q per well**.

The key consequence: **within a single well, the 10x barcode is effectively a unique cell ID**
(two cells sharing a barcode almost never land in the same well), so PairPlex pairs per-well by
barcode. Mispairing arises from the ways that assumption is *violated*:

- **Ambient / free molecules.** Not every molecule stays with its cell. Some are released and,
  during the well split, drift into a *different* well **while keeping their original barcode and
  UMI**. A resident cell's barcode-X reads can then be joined by background molecules that also
  carry barcode X (from a droplet-mate that shared the overloaded GEM).
- **Within-well barcode collisions.** Occasionally two cells that shared a barcode do land in the
  same well; with chain dropout this yields a heavy from cell A + a light from cell B.
- **Index hopping.** A fraction of reads are misassigned to the wrong well at sequencing.

SimPlex reproduces each of these as an explicit, tunable step, so mispairs are a **consequence
of the modelled mechanism**, never fabricated.

---

## 3. How it works (the pipeline)

SimPlex is a staged, vectorized pipeline (polars + numpy). Each stage is a pure function; the
whole run is reproducible from one `seed`.

```
real paired parquet
      │  load_pairs            → cells  (one row per input pair = one "cell")
      ▼
  assign_droplets_and_barcodes → group cells into overloaded droplets; each droplet a 10x barcode
      │  assign_wells          → each whole cell → one resident well
      ▼
  generate_molecules           → per chain: recovery, UMIs; mark molecules resident vs FREE;
      │                          stamp inherited RT error (shared by the molecule's read family)
      ▼
  route_and_amplify            → FREE molecules redistribute to another well (barcode+UMI kept);
      │                          molecule survival (before amplification); expand survivors into
      │                          read families; per-read index hopping → final_well
      ▼
  apply_sequencing_errors      → independent per-read errors
      │  build_merged          → assemble barcode+UMI+TSO+cDNA into reads
      ▼
  write_merged_fastq           → one FASTQ per well  (this is the PairPlex input)
  write_truth                  → truth_components / truth_cells / truth_barcodes parquet
  run_manifest.json            → seed, versions, config hash, counts
```

Then you run PairPlex on the FASTQs and hand its output to `simplex.score()`, which compares
PairPlex's emitted pairs to the truth tables.

---

## 4. Install & quickstart

SimPlex ships inside this repo. Its runtime dependencies (`polars`, `numpy`, `edlib`; and
`abstar`/`abutils`/`pyspoa` via PairPlex) come with a PairPlex install. If needed:

```bash
pip install edlib pyspoa   # if not already present
```

Minimal run:

```python
import simplex

reads_dir = simplex.run(
    input_data="./real_pairs.parquet",   # needs sequence_id:0/sequence:0/sequence_id:1/sequence:1 + locus:0/1
    output_directory="./sim_out",
    wells=96,
    cells_per_droplet_mean=2,              # loading rate lambda (cells per GEM)
    recovery_rate=0.5,
    release_rate=0.02,                    # ambient
    index_hop_rate=0.001,
    seed=0,
)
# reads_dir == ./sim_out/reads  (one FASTQ per well; feed this to pairplex.run)
```

**Input requirements.** `input_data` is a parquet with (at least) `sequence_id:0`, `sequence:0`,
`sequence_id:1`, `sequence:1`, and — required in v1 — `locus:0` / `locus:1`. A `name` column, if
present, is used as the stable `source_pair_id`. (This is exactly the shape PairPlex's own
`*_paired.parquet` output has.)

---

## 5. `simplex.run()` — the knobs

```python
simplex.run(input_data, output_directory, **knobs) -> Path   # returns the reads/ dir
```

Knobs, grouped by what they model. The **Effect** column is the *direction* a change pushes
results when you then run PairPlex and score (all else equal).

> ⚠️ **Defaults are an illustrative baseline, not a claim about any assay.** Calibrate the
> uncertain, high-leverage ones — especially `release_rate` (ambient), `cells_per_droplet_mean`
> (λ), `molecules_per_chain_mean`, and `recovery_rate` — to your real `metadata/*.csv`, and
> **sweep ranges** rather than trusting a single point.

### Sampling
| Knob | Default | Meaning | Effect |
|---|---|---|---|
| `n_cells` | `None` | Subsample/oversample the input to N cells (`None` = use all). | Dataset size. |
| `seed` | `0` | Master seed; all randomness derives from it per stage. | Reproducibility. |

### Overloading (how barcodes get shared)
| Knob | Default | Meaning | Effect |
|---|---|---|---|
| `wells` | `96` | Number of wells cells are distributed into. | More wells → fewer within-well barcode collisions. |
| `cells_per_droplet_mean` | `2` | **Loading rate λ = cells per GEM.** Cells are randomly loaded into `round(n_cells/λ)` droplets → **Poisson** occupancy. All cells in a droplet share one barcode. | Higher λ → more cells share a barcode → more collision risk. Realistic λ ≈ cells-loaded ÷ GEM-barcodes (often ~1–3). |
| `cells_per_droplet_overdispersion` | `0.0` | `0` = pure Poisson; `>0` makes droplet capture propensities vary (Dirichlet, concentration `1/overdispersion`) → Negative-Binomial-like clumping. | Higher → more uneven occupancy (cell clumping / GEM heterogeneity). |
| `barcode_pool_size` | `None` | `None` = unique barcode per droplet; `int` = sample droplet barcodes from a pool of this size (forces reuse). | A stress knob for barcode-collision robustness. |

### Capture & depth (how much signal each cell produces)
| Knob | Default | Meaning | Effect |
|---|---|---|---|
| `recovery_rate` | `0.5` | Per-chain capture probability. | Lower → more chain **dropout** → more unpaired cells and more mispair opportunity. |
| `molecules_per_chain_mean` | `20` | Mean distinct molecules (UMIs) per captured chain. | Higher → more UMI diversity / support. |
| `molecule_survival_rate` | `0.8` | Fraction of molecules surviving **before** amplification. | Lower → less support, more effective dropout. |
| `reads_per_molecule_mean` | `5` | Amplification depth per surviving molecule. | Higher → more reads per contig. |

### Contamination (the mispairing drivers)
| Knob | Default | Meaning | Effect |
|---|---|---|---|
| `release_rate` | `0.02` | Fraction of molecules that are **free** (redistribute to another well, keeping barcode+UMI). This is *ambient*. | Higher → more ambient contigs → more mispairs (with collisions/dropout) and more dropped `1H+2L` cells. |
| `index_hop_rate` | `0.001` | Fraction of reads misassigned to a different well at sequencing. | Higher → more cross-well contamination. (Must be 0 if `wells==1`.) |

### Errors (two layers, on purpose)
| Knob | Default | Meaning | Effect |
|---|---|---|---|
| `rt_sub_rate` / `rt_indel_rate` | `0.0` | RT errors applied to the molecule template — **inherited by the whole read family** (one UMI). | Higher → errors that consensus **cannot** average away → stresses `clustering_threshold`. |
| `sequencing_sub_rate` / `sequencing_indel_rate` | `0.001` / `0.0` | Independent per-read errors. | Higher → per-read noise (consensus removes most). |

### Read structure & output
| Knob | Default | Meaning | Effect |
|---|---|---|---|
| `barcode_length` / `umi_length` / `tso` | `16` / `10` / `TTTCTTATATGGG` | **Fixed** (validated) to match `pairplex.parse_barcodes`. | Changing them raises. |
| `chemistry` | `"v2"` | 10x whitelist to draw barcodes from. | Barcode source. |
| `output_mode` | `"merged"` | v1 supports `"merged"` only (paired-end/fastp is deferred). | — |
| `rc_fraction` | `0.0` | Fraction of reads written reverse-complemented. | Exercises orientation handling. |
| `variable_length` | `True` | Randomly truncate cDNA 5'/3'. | Stresses clustering/consensus. |
| `write_read_truth` | `False` | Also emit per-read provenance (`truth_reads.parquet`; large). | Debugging / route inspection. |

`run()` **refuses a non-empty output directory** (so a stale run can't contaminate a fresh one)
and estimates the total read count up front, raising if it would blow a memory budget.

---

## 6. Ground-truth outputs

Written under `output_directory/truth/`:

- **`truth_components.parquet`** *(primary scorer input)* — one row per
  `(final_well, barcode, origin_cell_id, chain)`: the true source sequence at that key, whether
  it is the **resident** source, and read/UMI/molecule support split into
  `n_reads_resident / n_reads_free / n_reads_index_hopped`.
- **`truth_cells.parquet`** — one row per simulated cell, with per-chain `captured`, `survived`,
  `n_molecules`, and read counts — so "chain never captured" is distinguishable from "captured
  but produced no usable reads".
- **`truth_barcodes.parquet`** — per `(well, barcode)`: physical resident occupancy (built from
  cells, **not** from observed reads), `is_collision`, `is_ambient_only`, per-resident-cell
  observability counts, and per-locus dominant source (by reads and by UMIs).
- **`truth_reads.parquet`** *(optional)* — per-read provenance.
- **`simplex_config.json`** + **`run_manifest.json`** — the exact knobs, seed, versions, and counts.

---

## 7. Scoring: `simplex.score()`

```python
pair_scores, key_scores = simplex.score(pairplex_output, truth_dir)
```

`pairplex_output` may be a PairPlex run directory (it globs `**/*_paired.parquet`), a single
parquet, or a list — it reads **all wells jointly**. `truth_dir` is the SimPlex `truth/` folder.

The scorer keys each emitted pair on `(well, barcode)` — **well is derived from the PairPlex
output filename** (`well000.fastq_paired.parquet`), because real PairPlex merged output has no
`well` column — and matches each chain sequence (bounded edit distance, **orientation-agnostic**,
never trusting PairPlex's own locus call) to a **set** of candidate true sources, then resolves
the pair jointly. Two non-empty candidate sets with an empty intersection ⇒ **mispaired**.

**`pair_scores`** — one row per emitted pair, with orthogonal status axes:

| Axis | Values |
|---|---|
| `pairing_status` | `correct` · `mispaired` · `unmatchable` · `ambiguous` |
| `source_resolution` | `unique` · `ambiguous` · `none` |
| `origin_status` | `resident` · `resident_plus_ambient` · `ambient` · `ambiguous` · `unknown` |
| `key_status` | `singleton` · `collision` · `ambient_only` · `unknown` |
| `output_status` | `unique` · `duplicate` |

**`key_scores`** — one row per truth `(well, barcode)`, **including keys PairPlex returned
nothing for** (`output_status="missing"`) — this is what lets you measure recall / yield loss,
not just precision. It carries fixed, threshold-independent observability levels
(`captured_both`, `survived_both`, `sequenced_both`, `reference_pairable_both`) so denominators
don't move as you change the threshold you're testing.

**Metrics you compute from these** (see `simplex/tests/test_single_factor.py::metrics` for a
reference): **pair precision** (resident-correct among emitted), **mispair rate**, **recall**
(singleton reference-pairable keys that got a unique resident-correct pair), and **yield loss**
(observable cells with no/duplicate output).

---

## 8. The end-to-end workflow

```python
import simplex, pairplex

# 1) generate synthetic reads with known truth
reads_dir = simplex.run(
    input_data="./real_pairs.parquet", output_directory="./sim",
    wells=8, cells_per_droplet_mean=2,
    recovery_rate=0.6, release_rate=0.15, index_hop_rate=0.001,
    seed=1,
)

# 2) run PairPlex under a candidate configuration
pairplex.run(
    sequences=str(reads_dir), output_directory="./pp_run",
    min_cluster_reads=3, min_cluster_umis=1, min_cluster_fraction=0.0,  # <- the thing you're tuning
    quiet=True,
)

# 3) score PairPlex's output against SimPlex truth
pair_scores, key_scores = simplex.score("./pp_run", "./sim/truth")

import polars as pl
mispaired = (pair_scores["pairing_status"] == "mispaired").sum()
correct   = (pair_scores["pairing_status"] == "correct").sum()
print("emitted pairs:", pair_scores.height, "| correct:", correct, "| mispaired:", mispaired)
print("keys with no output (yield loss):",
      (key_scores["output_status"] == "missing").sum())
```

To choose a threshold, wrap steps 2–3 in a loop over `min_cluster_fraction` (and/or
`min_cluster_umis`, `clustering_threshold`) and compare precision vs recall.

---

## 9. How to read the results (interpretation guide)

You are looking for a **precision/yield operating point**, not a single "best" number. "Best
threshold" is a trade-off: permissive filters admit wrong pairs (low precision); aggressive
filters reject real cells (low recall/yield).

**What to plot / compare across a threshold sweep:**

- **Mispair rate** (`mispaired / emitted`) — should fall as you tighten `min_cluster_fraction` /
  raise `min_cluster_umis`, because those remove low-support ambient contigs.
- **Recall** (correct resident pairs / reference-pairable singleton cells) — will eventually fall
  as tightening also deletes real low-support chains.
- **The frontier.** Pick the loosest setting that gets mispair rate under your tolerance, then
  read off the recall you pay for it. That is your production default.

**Signals and gotchas to watch:**

- **Non-monotonicity is real.** More read depth (or a moderate fraction filter) can *raise*
  recall by removing an ambient extra contig that was turning a recoverable `1H+1L` cell into a
  rejected `1H+2L` case — and then *lower* it again once you over-filter. Don't assume any single
  metric moves monotonically with a knob; read the frontier.
- **Mispairs concentrate in low-support minority chains.** In a mispaired pair, one chain is
  typically a small fraction of the barcode's reads/UMIs. That is exactly what a `cluster_fraction`
  filter targets. If your real PairPlex `metadata/*.csv` shows wrong-looking pairs with a dominant
  chain + a tiny minority chain, that is the ambient signature.
- **`key_status`.** Restrict recall to `singleton` keys; `collision` keys are a separate,
  harder regime (two real cells share a barcode in one well) and should be reported on their own.
- **`origin_status`.** `ambient` / `resident_plus_ambient` pairs are contamination even when a
  chain matches *some* real cell — don't count them as successes.

**Match the simulation to your biology.** Set `cells_per_droplet_*`, `wells`, `recovery_rate`,
and `release_rate` to bracket what you believe your assay does, and **sweep ranges** rather than
trusting a single point. SimPlex is deliberately dataset-agnostic; it is not calibrated to any
one dataset, and (v1) it cannot confirm from real data *which* real pairs are wrong — it tells
you how a given PairPlex configuration behaves on data whose truth you control.

---

## 10. What the tests demonstrate, and why

Two layers, both runnable (`pytest simplex/`):

- **Deterministic mechanism fixtures** (`tests/test_mechanism.py`) — seven hand-built scenarios,
  each run through the **real** `pairplex.run` and scored, proving the machinery reacts correctly
  to a *specific* mechanism: a clean run pairs ~perfectly; a free light molecule routed onto
  another cell's barcode produces a real **mispair**; a same-well barcode collision with dropout
  produces a **collision** mispair; an index-hopped read shows `final_well != amplification_well`;
  a shared heavy + distinct light still resolves **correct** (not ambiguous); a contaminant contig
  makes PairPlex emit nothing, which shows up as a **missing** key; and a cell's own free
  molecules co-landing are correctly scored **coherent, not mispaired**. These exist to prove the
  simulator + scorer actually detect the failure modes — a mispair that PairPlex should catch is
  caught, and a coherent pair is not falsely flagged.
- **Single-factor tests** (`tests/test_single_factor.py`) — run the whole generator→PairPlex→score
  loop and assert **regime-specific** trade-offs (not blanket monotonicity): a `min_cluster_fraction`
  filter reduces mispairs under ambient extra-contigs, costs recall under weak real chains, and
  over-filters past a point. This is the harness reproducing, on synthetic data with known truth,
  the precision/yield behaviour you will tune against — and mechanistically confirming that
  ambient contamination + permissive filters is the mispairing driver.

---

## 11. Scope & limitations

- **v1 scale:** in-memory reference simulator, roughly 5k–50k cells. Stages are partitionable by
  well; 1M-scale sharded/streaming execution is deferred (Phase 5).
- **Output:** merged reads only in v1. Paired-end + real fastp merging is deferred (Phase 3).
- **Reproducibility:** identical `seed` + identical input order + identical layout → identical
  output. Chunk/row-order invariance is a Phase-5 (entity-keyed RNG) item.
- **Deferred phenomena** (tracked in the design spec): alternative PairPlex pairing strategies,
  a `barcode_swap` (PCR-recombination) mechanism, PCR chimeras, empirical distribution
  calibration, and UMI sequencing errors.
- **No labelled real truth.** SimPlex measures PairPlex on data whose truth it controls; it does
  not, by itself, tell you which pairs in a *real* run are wrong. Use it to choose thresholds and
  to characterise how a configuration behaves, then apply those thresholds to real runs.

---

*Design of record: `docs/superpowers/specs/2026-08-27-simplex-generator-design-v5.md`. SimPlex was
built to make PairPlex threshold selection an evidence-based, precision/yield decision.*
