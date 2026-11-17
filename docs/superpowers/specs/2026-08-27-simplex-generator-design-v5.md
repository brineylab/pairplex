# SimPlex — synthetic data generator + scorer for PairPlex (design spec, v5 — FROZEN)

**Date:** 2026-08-27 (v5 — final consistency edits; frozen)
**Branch:** `simplex`
**Status:** FROZEN. Drives Phase 0–2 plan v4. No further design revisions expected — subsequent
issues are ordinary implementation/test corrections.

> **v5 edits (freeze):** removed `read_length`/`platform` from the Phase 1–2 API (merged-only;
> reserved for Phase 3); `truth_reads` is a **single parquet in v1** (chunking deferred to
> Phase 5); TSO is **fixed and validated** (`config.tso` must equal the canonical TSO, since the
> parser's `s[36:].lstrip("G")` extraction assumes it).
**Author:** brainstormed with bnemoz

> **Version history.** v1 simulated the *conclusion* (read-level barcode-swap "ambient"). v2
> fixed the mechanism, truth, scorer, metrics, scale honesty. v3 fixed compositional
> provenance, joint/orthogonal scoring, honest Phase 0, API/RNG/locus. **v4** (this doc)
> consistency cleanup: molecule record no longer carries a per-read `is_index_hopped`; one
> reproducibility claim (v1 = same seed+order+layout, NOT chunk-invariant); `barcode` naming
> standardized in truth. Guiding principle unchanged: **simulate the wet-lab mechanism, not the
> hypothesis we want to confirm.**

## 1. Motivation & anti-circularity

PairPlex mispairs heavy/light chains at scale. The core pipeline is correct on clean data
(`INVESTIGATION_NOTES.md`). Clean-data tests rule out several pipeline defects, and
contamination interacting with permissive filters is the **leading hypothesis** because it
reproduces the observed failure pattern — but it has **not** been directly confirmed on labeled
real data (the data we have carry no "which pair is wrong" labels). To choose thresholds we need
synthetic "raw sequencing" data reproducing the **physical mechanism** with **mechanism-faithful
ground truth**, plus a scorer that measures the real failure and the precision/yield tradeoff.

**Circularity guard:** the simulator must not merely inject the defect we expect and let the
sweep rediscover it. Mitigations: keep knobs explicit and **sweep ranges** (never fit the
generator to one dataset); keep SimPlex/PairPlex **dataset-agnostic**; and, where real data is
used at all (Phase 0A), use it only to **bracket plausible ranges**, documenting that no labeled
"which pair is wrong" truth exists in the data we have.

## 2. Scope, decomposition, roadmap

Full vision = harness: generator → scorer → sweep runner. Built in phases; **thresholds are not
selected until Phase 4.** Each phase is its own spec→plan cycle. This document is the design of
record for all phases; the *implementation plan* targets **Phase 0–2**.

- **Phase 0A — Real-data audit (light, optional, agnostic).** Given a real PairPlex `metadata/*.csv`,
  summarize marginal distributions to **bracket knob ranges** for later sweeps. Never fits the
  generator; runs only if such a file is supplied; explicitly records that these files carry no
  labeled truth, so the "wrong pairs have a low-support minority chain" prediction stays
  **unconfirmed** on them. Separates raw/algorithm-independent observables from
  reference-PairPlex-derived observables; freezes a reference extraction config.
- **Phase 0B — Scoring contract (required, dataset-independent).** Exact label-resolution rules,
  key-level outcomes, ambiguity fixtures, metric definitions. Produced before generator effects.
- **Phase 1 — Mechanistic minimal generator.** Cells → overloaded droplets/barcodes → whole-cell
  well assignment → per-chain molecules/UMIs → resident vs free molecules → free molecules
  redistributed across wells **retaining barcode+UMI** → molecule survival → molecule-level
  amplification (read families) → index hopping → merged reads → compact truth. Minimal errors.
- **Phase 2 — Compact truth + scorer.** `truth_components.parquet` + `pair_scores`/`key_scores`
  scorer keyed by `(final_well, barcode)`, orthogonal status axes, joint set resolution.
- **Phase 3 — Golden + single-factor mechanistic tests; realistic paired-end/fastp;
  sequence-similarity battery; ≥1 genuine extra-chain scenario.**
- **Phase 4 — Calibration ranges + robust sweeps** across params, seeds, held-out sequences;
  precision–yield **Pareto** with CIs; never optimize+evaluate on the same seeds. Before any
  **production** threshold recommendation, the sweep must cover ranges that **bracket the real
  marginal distributions** (from Phase 0A if available) and show the chosen threshold is **robust
  across those ranges** — not tuned to a single simulated point.
- **Phase 5 — Scale redesign** (shard by final well; streaming; partitioned truth).

**Deferred (tracked, not dropped):** alternative pairing *strategies* + a PairPlex
`pairing_policy` seam (Phase 4+); `barcode_swap`/PCR recombination; PCR chimeras; empirical
distribution calibration; 1M-scale streaming. See §11 matrix.

**Separate immediate task (own branch, not in the SimPlex plan):** PairPlex **hygiene** — docs'
broken `clustering_threshold=0.0`, CLI/docs `min_cluster_umis` disagreement, fixed-offset parse
fragility. Must land before Phase 4 experiments so we don't measure a misconfigured PairPlex.

## 3. Packaging

Sibling top-level package `simplex/`, imported as `import simplex` → `simplex.run(...)` and
`simplex.score(...)`. May import read-structure constants / whitelists from `pairplex`.

## 4. Corrected biological model (mechanism)

Wet lab: cells fixed in bulk → 10X barcode by **RT in overloaded GEMs (many cells share a
barcode)** → **whole fixed cells** distributed into a 96-well plate → per-well Illumina index →
sequenced, demuxed upstream to one FASTX per well.

Stage order (each stage per-well-partitionable for Phase 5):

1. **Cells → overloaded droplets.** cells/droplet ~ Normal(mean, sd), clamped ≥1. Each droplet
   gets a 10X barcode. Overloading ⇒ many cells share a barcode. Optional `barcode_reuse` lets
   distinct droplets collide on a barcode (default off, but **available** so it can be tested).
2. **RT molecules.** Per cell, per chain: per-chain recovery Bernoulli; molecule count; each
   molecule = one `molecule_id` + one UMI, carrying the cell's droplet barcode (`origin_barcode`).
   Optional molecule-level RT error stamped here (inherited by the whole read family).
3. **Resident vs free.** Each molecule `is_free` with prob `release_rate`. Free molecules are the
   ambient pool; they **retain barcode + UMI**.
4. **Whole-cell → well.** Each cell → `resident_well` uniformly. Resident molecules take
   `amplification_well = resident_well`.
5. **Free-molecule redistribution.** Each free molecule independently picks
   `amplification_well` (uniform for now), **keeping barcode + UMI**.
6. **Molecule survival + amplification.** Each molecule survives with `molecule_survival_rate`
   (Bernoulli, **before** amplification — fixes v1's read-thinning bug); survivors expand into a
   read family sharing one UMI (depth ~ `reads_per_molecule_mean`), inheriting any RT error;
   `parent_molecule_id` retained.
7. **Sequencing error** per read (independent, `sequencing_sub_rate`/`sequencing_indel_rate`).
8. **Index hopping.** Some reads get `final_well ≠ amplification_well` (barcode/UMI unchanged),
   `is_index_hopped=True`. Reads that don't hop have `final_well = amplification_well`.
9. *(Deferred)* **`barcode_swap`** — actually changes `final_barcode ≠ origin_barcode`; separate
   knob, **not** "ambient."

Emergent (no explicit knobs): resident-cell contamination by a **same-barcode droplet mate**;
ambient-only `(well, barcode)` keys; cross-well contamination; UMI-coherent read families;
support values usable to study `min_cluster_reads/umis/fraction`.

**Corrected invariant (precise).** *With `barcode_reuse` off and no `barcode_swap`, one cell per
barcode prevents cross-source **mispairing**.* It does **not** prevent an `ambient_coherent`
false positive: free heavy+light from the same cell can co-land in a well with **no resident
cell** of that barcode — source-coherent, but not a recovered resident pair. "Dropout" means
**the resident chain is absent from the set of accepted contigs**, from any cause (capture loss,
molecule loss, insufficient read/UMI support, filtering, truncation, clustering failure,
annotation failure). With `barcode_reuse` on, even `cells_per_droplet=1` need not yield one
source per barcode.

## 5. Provenance & ground truth

**Molecule-level record** (atomic; one UMI per molecule so molecule-level `n_umis` is
unnecessary — UMI collisions surface as `n_source_molecules > n_umis` on aggregation):

```
molecule_id, parent_molecule_id, origin_cell_id, origin_droplet_id, source_pair_id, chain,
locus, umi, barcode, resident_well, amplification_well, is_free, survived
```

Index hopping is a **per-read** property (it splits one molecule's family across wells), so it
lives on the `reads` record (`is_index_hopped`, `final_well`), **not** on the molecule. Barcode
is a single `barcode` column (origin == final in Phase 1–2 since `barcode_swap` is deferred);
truth uses the name `barcode` everywhere. Component aggregation counts reads by destination.

**Ground-truth outputs:**

- **`truth_components.parquet`** *(primary scorer input; small)* — one row per
  `(final_well, barcode, origin_cell_id, chain)`:
  `final_well, barcode, origin_cell_id, source_pair_id, chain, locus, sequence,
  is_resident_source, n_source_molecules, n_umis, n_reads, n_reads_resident, n_reads_free,
  n_reads_index_hopped`. (Separate resident/free/hopped read counts because one origin+chain can
  contribute both resident and free molecules to the same key — a scalar `route` cannot describe
  it.)
- **`truth_cells.parquet`** — one row per cell: `cell_id, source_pair_id, droplet_id, barcode,
  resident_well, chain{0,1}_id, chain{0,1}_seq, chain{0,1}_locus`, and per chain: `captured,
  survived, n_molecules, n_umis, n_reads_generated, n_reads_resident, n_reads_free_out,
  n_reads_index_hopped_out`.
- **`truth_barcodes.parquet`** — per `(well, barcode)` where the key set is the **union** of
  *physical* resident keys `(resident_well, barcode)` from `cells` and *observed* keys
  `(final_well, barcode)` from `reads`. **Physical occupancy comes from `cells`, never from
  observed components** — otherwise a resident cell that produced no read silently disappears
  (undercounting collisions, mislabeling `ambient_only`). Columns: resident source set,
  `n_resident_cells`, `is_collision` (≥2 resident cells), `is_ambient_only` (observed key with
  zero resident cells), and **collision counts** (not just booleans) computed from truth:
  `n_captured_both_resident_cells`, `n_survived_both_resident_cells`,
  `n_sequenced_both_resident_cells`, `n_reference_pairable_resident_cells` (each reduces to 0/1
  at singleton keys); plus **per-locus** dominance by both reads and UMIs
  (`dominant_heavy_source_by_reads/by_umis`, `dominant_light_source_by_reads/by_umis`, light =
  `locus ∈ {IGK, IGL}`) since the dominant heavy and light source can differ.
- **`truth_reads.parquet`** *(optional; `write_read_truth=False`)* — per-read; **single parquet
  in v1** (per-well chunking deferred to Phase 5).
- **`simplex_config.json`** + **`run_manifest.json`** (resolved seed, versions, stage RNG keys).

## 6. Scorer specification (0B defines; Phase 2 implements)

Two outputs:

- **`pair_scores.parquet`** — one row per PairPlex-returned pair.
- **`key_scores.parquet`** — one row per truth `(final_well, barcode)` evaluation unit
  (**including keys PairPlex returned nothing for** — required for recall/yield).

**Keying.** Barcode = token before `_contig` in the pair's `sequence_id`; well = the `well`
column (tolerant of decorated `name`s like `{bc}_d{dd}_w{ww}` seen in real output — never assume
`name == barcode`).

**Score ALL PairPlex outputs jointly.** `score()` takes a **directory or list of paired
parquets** (not one file), reads them all, and computes `pair_scores`+`key_scores` **once**
against the whole truth — otherwise every other well's truth keys are wrongly marked `missing`.

**Sequence matching returns a SET of candidate `source_pair_id`s** (duplicates are guaranteed:
the reference input has ~40% shared heavies / ~50% shared lights). Matching is **locus-restricted**
(from truth locus, never PairPlex's own annotation — circular) and **restricted to sources
actually present at that key**. Because Phase 1 has inherited RT + independent sequencing errors,
matching is **bounded edit-distance** (via `edlib` infix/HW alignment) with a max-edit-fraction
and a **minimum aligned length** so short sequences don't match many sources; containment is
symmetric (`seq in full` **or** `full in seq`), not one-directional.

**Pairing status and source resolution are separate axes.** Resolve jointly over the candidate
sets: **any two non-empty candidate sets with empty intersection ⇒ `pairing_status=mispaired`**
(a cross-source pair is impossible) even if one side's exact source is ambiguous; a unique
intersection ⇒ `correct` with `source_resolution=unique`; a non-unique intersection ⇒ `correct`
pairing but `source_resolution=ambiguous`; empty candidates ⇒ `unmatchable`.

**Orthogonal status axes (named labels are derived, not primary):**
```
pairing_status         : correct | mispaired | unmatchable | ambiguous
source_resolution      : unique | ambiguous | none
origin_status          : resident | resident_plus_ambient | ambient | ambiguous | unknown
key_status             : singleton | collision | ambient_only | unknown   # unknown = key absent from truth
output_status          : unique | duplicate | missing                     # missing only on key_scores
```
`origin_status` is computed from the resolved source(s)' `is_resident_source`; for an ambiguous
pair, it is `ambiguous` unless every permitted source assignment shares one origin category.
A key absent from truth is `key_status=unknown`, **never** silently `ambient_only`.
`key_scores` also carries `output_count`, and (best-effort, **conditional on PairPlex metadata /
unpaired output being available**) a `no_output` refinement:
`filtered_all | single_chain_only | extra_contigs_rejected | annotation_failure | unknown`.

**Threshold-independent observability levels** (fixed, never the threshold under test — avoids
biasing the denominator). **Computed per resident cell/source first** — `(well, barcode,
origin_cell_id)` — then summarized at key level, so a collision key can never combine cell A's
heavy with cell B's light into a false "both chains present": `captured_both`, `survived_both`,
`sequenced_both` (≥1 final read each chain, same cell), `reference_pairable_both` (a frozen
preregistered minimum, e.g. fixed min reads+UMIs, set only to exclude physically unrecoverable
cases). These come from truth (`captured`/`survived`/`n_molecules` preserved in `truth_cells`),
**not** from `captured_both = sequenced_both`. Report **recall against both `sequenced_both` and
`reference_pairable_both`.**

**Metrics (all reported; "best threshold" is not scalar):** biological recovery, technical
observability (levels above), algorithmic recall, pair precision (resident-correct among
returned), mispair rate, rejection/yield loss, collision performance. Sweeps → precision–yield
**Pareto frontier**; production default chosen for a target precision / max mispair rate.

## 7. Support & error model

**Support (Phase 1 minimal).** Per-chain recovery Bernoulli → molecule count → **molecule
survival Bernoulli before amplification** (`molecule_survival_rate`; renamed from the misleading
`seq_efficiency` so nobody re-implements read thinning) → per-molecule read depth. Phase 4
options (matrix, not now): separate H/L recovery; **cell-level shared abundance** (correlated
dropout/support); chain-specific abundance; negative-binomial/log-normal counts; PCR jackpotting
/ empirical reads-per-UMI; well-depth variation; empirical distributions from real metadata.

**Errors, inheritance-aware.** RT/molecule error (inherited by all reads of a UMI); sequencing
error (independent per read). Phase 1 must support **molecule-level inherited RT error**
(`rt_sub_rate`/`rt_indel_rate`), else consensus erases independent errors and the
`clustering_threshold` sweep is unrealistically easy. PCR-branch error deferred.

## 8. Read assembly, output, paired-end

Merged layout matches `pairplex.parse_barcodes`: `barcode(16)+umi(10)+TSO+cDNA`. **Phase 1–2
default `output_mode="merged"`.** Paired-end realism (R1/R2 overlap, adapter read-through,
`rc_fraction` in paired mode, **long inserts that fail to merge**, Illumina naming verified
against `abstar.pp.merge_fastqs`) is a **Phase 3** deliverable with a real end-to-end test
calling `pairplex.run(..., merge_paired_reads=True)` over mergeable+non-mergeable fragments and
asserting actual fastp retention.

## 9. Public API (v2 config)

```python
simplex.run(
    input_data, output_directory,
    n_cells=None, wells=96,
    cells_per_droplet_mean=5, cells_per_droplet_sd=2,
    barcode_pool_size=None,            # None = unique barcode per droplet; int = sample from a
                                       # pool of this size (enables controlled reuse/collision)
    recovery_rate=0.5, molecules_per_chain_mean=10,
    release_rate=0.02,                 # fraction of molecules released/free (redistributed, not added)
    molecule_survival_rate=0.8,        # molecule Bernoulli survival BEFORE amplification
    reads_per_molecule_mean=5,
    rt_sub_rate=0.0, rt_indel_rate=0.0,            # inherited by a UMI's read family
    sequencing_sub_rate=0.001, sequencing_indel_rate=0.0,  # independent per read
    index_hop_rate=0.001,
    barcode_length=16, umi_length=10, tso="TTTCTTATATGGG", chemistry="v2",  # barcode/umi/tso fixed & validated
    output_mode="merged", rc_fraction=0.0,                                    # read_length/platform NOT in v1 API (Phase 3)
    variable_length=True, write_read_truth=False, seed=0,
)
simplex.score(pairplex_output_dir_or_parquet_list, truth_dir, *, pairplex_metadata=None) -> (pair_scores, key_scores)
```

- `leakage_rate`/`ambient_only_barcodes` (v1) **removed**; ambient is `release_rate`
  (redistribution — moves molecules, does **not** create extra ones). Barcode-changing events are
  the deferred `barcode_swap`. `barcode_reuse` boolean replaced by explicit `barcode_pool_size`.
- **Locus is required for Phase 1–2.** `validate()` fails if the input lacks `locus:0/1` and no
  frozen source-annotation is supplied — never silently proceed with `"unknown"` loci (the scorer
  only searches `IGH/IGK/IGL`). Also validate that repeated `source_pair_id`s describe the **same**
  sequences+loci, so two unrelated records with the same name aren't treated as equivalent.
- **`output_mode="paired"` is rejected in Phase 1–2** (only `"merged"` implemented; paired = Phase 3).
- **`index_hop_rate` must be 0 when `wells==1`** (a hop can't change well); validation enforces it.
- **`tso` is fixed and validated** — `validate()` fails unless `tso` equals the canonical
  `"TTTCTTATATGGG"`, because `parse_barcodes` extracts cDNA as `s[36:].lstrip("G")` and an
  arbitrary TSO would silently corrupt extraction. Truth-support dominance aggregates by
  `source_pair_id` (clonal copies across cells sum before dominance is chosen).
- **Read-count/OOM guard**: `validate(actual_n_cells=…)` estimates
  `n_cells·2·recovery·molecules·survival·depth` (**no** release factor — release moves, not adds)
  and refuses past a budget; `run()` calls it with the resolved cell count (works when `n_cells=None`).
- `run()` **fails on a non-empty output directory** (or cleans it) so stale FASTQs can't
  contaminate an experiment.
- All stage schemas (cells → molecules → reads → built) are declared in the plan, not just truth.

## 10. Reproducibility & keyed RNG

**v1 guarantee (honest scope):** *same seed + same input (order) + same execution layout →
identical content.* Randomness is drawn from **per-stage** named streams via
`rng_for(seed, stage) = SeedSequence(blake2b("{seed}|{stage}"))` (never `seed+offset`, never
Python `hash()`), so stages are independent and a run is bit-reproducible. It does **not** yet
guarantee invariance to row reordering or chunk boundaries — stages consume one sequential RNG
over the whole table.

**Deferred to Phase 5 (true sharding invariance):** entity-keyed randomness — derive each draw
from a stable identifier `(stage, cell_id|molecule_id, draw_slot)` (Philox counter or
blake2b-seeded per-entity), so results are independent of partition/chunk size. Note that putting
`chunk_id` in the seed does **not** achieve this (re-chunking changes chunk IDs). v1 does not
claim chunk-size invariance.

## 11. Hypothesis matrix (visible, phased)

| Phenomenon | Status | Phase |
|---|---|---|
| Overloaded shared-barcode droplet mates | core | 1 |
| Free-molecule ambient (barcode retained, well changes, pre-PCR) | core | 1 |
| Molecule survival before amplification | core | 1 |
| Index hopping (cross-well, post-amplification) | core | 1 |
| Inherited RT error vs independent seq error | core | 1 |
| `(final_well,barcode)` scorer, orthogonal axes, key_scores | core | 0B/2 |
| Chain dropout / broadened "resident chain absent" | core | 1/2 |
| Collision-induced mispair (same-barcode, same-well, asymmetric loss) | core test | 2/3 |
| Barcode reuse across droplets (`barcode_pool_size`) | optional mode, default off | 1 (mode) |
| UMI sequencing errors (inflate apparent distinct-UMI count → affects `min_cluster_umis`) | deferred, **tracked** | 3/4 |
| `barcode_swap` / PCR recombination | deferred, separate knob | later |
| PCR chimeras (H-L fusion, high-support) | deferred; **must be in final eval** | later |
| Genuine extra chains (dual-light, nonproductive, alt heavy) | ≥1 explicit scenario | 3 |
| Correlated abundance, NB/log-normal, jackpotting, empirical calibration | calibration | 4 |
| Well/cell heterogeneity | robustness | 4 |
| Alternative pairing **strategies** + PairPlex `pairing_policy` seam | **deferred** (user) | later |
| PairPlex hygiene fixes | **separate immediate task, own branch** | before Phase 4 |

## 12. Testing plan

**Deterministic mechanism fixtures (prove the mechanism; before any statistical test).** These
**must be genuinely controlled**, not stochastic full-generator runs "hoping" the seed produces
the condition: construct exact low-level tables (`cells`/`molecules`/`reads`) via private test
helpers that force the routing, then run only the downstream FASTQ → PairPlex → scorer path
free. Stochastic *frequency* behavior is covered by the separate single-factor tests.
0. **Clean golden invariant** — 1 cell/barcode, `barcode_pool_size=None`, `release_rate=0`,
   `recovery=survival=1`, no errors, `wells≥1` → ~100% `resident_correct`, **no** `ambiguous`/
   `unmatchable` pairs. (The one-cell-with-dropout negative control does not replace this.)
1. **Exact ambient mispair** — `wells≥2`; cells A,B share barcode X in different wells; force
   A-heavy present, A-light absent, one **free** B-light molecule routed to A's well; permissive
   PairPlex must emit A_H + B_L (`pairing_status=mispaired`). *(wells=1 would make this a
   collision, not ambient — invalid.)*
2. **One-cell negative control** — 1 cell/barcode, release+dropout on; **zero cross-source
   mispairs**, `ambient_coherent` outputs allowed.
3. **Same-well collision** — two same-barcode cells forced into one well, asymmetric chain loss →
   mispair at a `key_status=collision` key.
4. **Route composition** — a free molecule amplified in one well, one read forced to index-hop;
   assert (via `truth_reads`) `amplification_well ≠ final_well`, barcode+UMI unchanged.
5. **Joint ambiguity** — shared heavy across two source pairs + unique light resolves to one
   source (`pairing_status=correct`, `source_resolution` may be `unique`), not `ambiguous`.
6. **Missing output** — both resident chains `reference_pairable`, but a contaminant contig
   causes PairPlex to emit no pair → `key_scores` row `output_status=missing`.

**Per-stage unit tests** (keyed RNG, deterministic): droplet distribution + barcodes shared
within/distinct across (unless `barcode_pool_size` set, which must produce a verified reuse
collision); **well uniformity AND analytic same-barcode co-occupancy** `Σ_d C(k_d,2)/wells`
within tolerance;
molecule recovery/survival/UMIs; free/resident split fraction; free-molecule redistribution
retains barcode+UMI; index-hop cross-well fraction; inherited-RT-error family coherence; read
layout **round-trips through `pairplex.parse_barcodes`**; config validation/OOM guard.

**Statistical single-factor tests** (assert **mechanistic statistics**, not blanket "every knob
degrades" — effects can be non-monotonic, e.g. more depth can aid recovery *or* push a
contaminant over an absolute threshold): shared barcodes no-free → occupancy/collision rate;
free no-dropout → mostly yield loss, limited mispair; free + light dropout → mispairs emerge;
↑`min_cluster_fraction` → precision/yield tradeoff; ↑`min_cluster_umis` → helps only when ambient
UMI complexity low; index hopping alone → cross-well pattern.

**Reproducibility:** *same seed + same input (order) + same execution layout* → identical
decompressed FASTQ **and** truth tables. **No** chunk-size/row-order invariance claim in v1
(deferred to Phase 5 with entity-keyed draws). **Scale smoke test** at the v1 target (≤~50k cells).

## 13. Scale stance (v1: "A now, structured for B")

Reference simulator, in-memory, ~5k–50k cells; **no 1M/streaming claim for v1.** Stages are
**partitionable** by `final_well`, which makes the Phase 5 sharding structurally straightforward —
but Phase 5 also **requires** replacing per-stage sequential RNG with **entity-keyed draws**
(§10) to get chunk/order invariance, so it is not a pure drop-in swap of the current RNG.

## 14. Non-goals (Phase 1–2)

Sweep runner, alternative pairing strategies + `pairing_policy` seam, `barcode_swap`, PCR
chimeras/branch error, empirical calibration, paired-end realism, 1M streaming, and the PairPlex
hygiene fixes (separate branch). All tracked in §11, not dropped.
