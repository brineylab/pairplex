# SimPlex — synthetic data generator + scorer for PairPlex (design spec, v3)

**Date:** 2026-08-27 (v3 after design reviews 1 & 2)
**Branch:** `simplex`
**Status:** revised design; pending review before Phase 0–2 plan
**Author:** brainstormed with bnemoz

> **Version history.** v1 simulated the *conclusion* (read-level barcode-swap "ambient"). v2
> fixed the mechanism, truth, scorer, metrics, scale honesty. **v3** (this doc) fixes the
> compositional provenance/well schema, makes scoring joint + orthogonal + able to represent
> missing outputs, operationalizes Phase 0 honestly, and pins the public API, keyed RNG, locus
> contract, and single-factor mechanisms. Guiding principle unchanged: **simulate the wet-lab
> mechanism, not the hypothesis we want to confirm.**

## 1. Motivation & anti-circularity

PairPlex mispairs heavy/light chains at scale. The core pipeline is correct on clean data
(`INVESTIGATION_NOTES.md`); mispairing comes from contamination meeting permissive filters. To
choose thresholds we need synthetic "raw sequencing" data reproducing the **physical
mechanism** with **mechanism-faithful ground truth**, plus a scorer that measures the real
failure and the precision/yield tradeoff.

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
  precision–yield **Pareto** with CIs; never optimize+evaluate on the same seeds.
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
locus, umi, origin_barcode, final_barcode, resident_well, amplification_well,
is_free, is_index_hopped, is_barcode_swapped, n_reads
```

`final_well` is per-**read** (index hopping splits a molecule's reads across wells), so per-read
truth carries `final_well`; component aggregation counts reads by destination.

**Ground-truth outputs:**

- **`truth_components.parquet`** *(primary scorer input; small)* — one row per
  `(final_well, final_barcode, origin_cell_id, chain)`:
  `final_well, final_barcode, origin_cell_id, source_pair_id, chain, locus, sequence,
  is_resident_source, n_source_molecules, n_umis, n_reads, n_reads_resident, n_reads_free,
  n_reads_index_hopped`. (Separate resident/free/hopped read counts because one origin+chain can
  contribute both resident and free molecules to the same key — a scalar `route` cannot describe
  it.)
- **`truth_cells.parquet`** — one row per cell: `cell_id, source_pair_id, droplet_id, barcode,
  resident_well, chain{0,1}_id, chain{0,1}_seq, chain{0,1}_locus`, and per chain: `captured,
  survived, n_molecules, n_umis, n_reads_generated, n_reads_resident, n_reads_free_out,
  n_reads_index_hopped_out`.
- **`truth_barcodes.parquet`** — per `(final_well, final_barcode)`: resident source set,
  `n_resident_cells`, `is_collision` (≥2 resident cells), **per-locus** `dominant_heavy_source`
  / `dominant_light_source` (by observed reads/UMIs — dominance is per locus because the
  dominant heavy and light source can differ, the exact mispair condition), `is_ambient_only`.
- **`truth_reads.parquet`** *(optional; `write_read_truth=False`)* — per-read; chunked by well.
- **`simplex_config.json`** + **`run_manifest.json`** (resolved seed, versions, stage RNG keys).

## 6. Scorer specification (0B defines; Phase 2 implements)

Two outputs:

- **`pair_scores.parquet`** — one row per PairPlex-returned pair.
- **`key_scores.parquet`** — one row per truth `(final_well, barcode)` evaluation unit
  (**including keys PairPlex returned nothing for** — required for recall/yield).

**Keying.** Barcode = token before `_contig` in the pair's `sequence_id`; well = the `well`
column (tolerant of decorated `name`s like `{bc}_d{dd}_w{ww}` seen in real output — never assume
`name == barcode`).

**Sequence matching returns a SET of candidate `source_pair_id`s** (duplicates are guaranteed:
the reference input has ~40% shared heavies / ~50% shared lights). Matching is **locus-restricted**
(from truth locus, never PairPlex's own annotation — circular) and **restricted to sources
actually present at that key**, then resolved **jointly**: examine pairwise source assignments;
mark ambiguous only when **multiple biologically distinct pair assignments** remain (heavy
`{A,B}` ∩ light `{A}` → `A`, *not* ambiguous). Clean tests use exact/substring; noisy tests use
orientation-aware alignment / edit-distance with an explicit ambiguity rule.

**Orthogonal status axes (named labels are derived, not primary):**
```
pairing_status : correct | mispaired | unmatchable | ambiguous
origin_status  : resident | resident_plus_ambient | ambient
key_status     : singleton | collision | ambient_only
output_status  : unique | duplicate | missing        # missing only on key_scores
```
`key_scores` also carries `output_count`, and (best-effort, **conditional on PairPlex metadata /
unpaired output being available**) a `no_output` refinement:
`filtered_all | single_chain_only | extra_contigs_rejected | annotation_failure | unknown`.

**Threshold-independent observability levels** (fixed, never the threshold under test — avoids
biasing the denominator): `captured_both`, `survived_both`, `sequenced_both` (≥1 final read each
chain), `reference_pairable_both` (a frozen preregistered minimum, e.g. fixed min reads+UMIs, set
only to exclude physically unrecoverable cases). Report **recall against both `sequenced_both`
and `reference_pairable_both`.**

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
    cells_per_droplet_mean=5, cells_per_droplet_sd=2, barcode_reuse=False,
    recovery_rate=0.5, molecules_per_chain_mean=10,
    release_rate=0.02,                 # fraction of molecules released/free (ambient pool)
    molecule_survival_rate=0.8,        # molecule Bernoulli survival BEFORE amplification
    reads_per_molecule_mean=5,
    rt_sub_rate=0.0, rt_indel_rate=0.0,            # inherited by a UMI's read family
    sequencing_sub_rate=0.001, sequencing_indel_rate=0.0,  # independent per read
    index_hop_rate=0.001,
    barcode_length=16, umi_length=10, tso="TTTCTTATATGGG", chemistry="v2",
    output_mode="merged", read_length=300, rc_fraction=0.0, platform="illumina",
    variable_length=True, write_read_truth=False, seed=0,
)
simplex.score(pairplex_paired_parquet, truth_dir, *, pairplex_metadata=None) -> (pair_scores, key_scores)
```

- `leakage_rate` (v1) is **removed**; ambient is `release_rate` (redistribution). Barcode-changing
  events are the deferred `barcode_swap`.
- `ambient_only_barcodes` removed — emergent.
- **Config validation + read-count/OOM guard**: estimate total reads
  (`n_cells·2·recovery·molecules·survival·depth·(1+release drift)`) and refuse / warn past a
  budget.
- All stage schemas (cells → molecules → reads → built) are declared in the plan, not just truth.

## 10. Keyed RNG (reproducible under sharding)

Randomness derives from stable keys, **not** `seed + fixed_offset` (insufficient for
chunk-order invariance) and **never** Python `hash()` (process-randomized). Use
`numpy.random.SeedSequence` spawning (or Philox counter-based) seeded from a stable integer key
built from `(master_seed, stage_name, entity_id|well_id, chunk_id)` via a fixed hash
(`hashlib.blake2b` digest → int). Result is independent of chunk size and processing order.

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
| Barcode reuse across droplets | optional mode, default off | 1 (mode) |
| `barcode_swap` / PCR recombination | deferred, separate knob | later |
| PCR chimeras (H-L fusion, high-support) | deferred; **must be in final eval** | later |
| Genuine extra chains (dual-light, nonproductive, alt heavy) | ≥1 explicit scenario | 3 |
| Correlated abundance, NB/log-normal, jackpotting, empirical calibration | calibration | 4 |
| Well/cell heterogeneity | robustness | 4 |
| Alternative pairing **strategies** + PairPlex `pairing_policy` seam | **deferred** (user) | later |
| PairPlex hygiene fixes | **separate immediate task, own branch** | before Phase 4 |

## 12. Testing plan

**Deterministic mechanism fixtures (prove the mechanism; before any statistical test):**
1. **Exact ambient mispair** — cells A,B share barcode X, different wells, A-light absent, one
   B-light molecule routed to A's well; permissive PairPlex must emit A_H + B_L.
2. **One-cell negative control** — 1 cell/barcode, release+dropout on; **zero cross-source
   mispairs**, `ambient_coherent` outputs allowed.
3. **Same-well collision** — two same-barcode cells in one well, asymmetric chain loss →
   classified collision-induced mispair.
4. **Route composition** — a free molecule amplified in one well, one read index-hops;
   `amplification_well ≠ final_well`, barcode+UMI unchanged.
5. **Joint ambiguity** — shared heavy + unique light resolves to one source (not
   `sequence_ambiguous`).
6. **Missing output** — both resident chains `reference_pairable`, but a contaminant contig
   causes PairPlex to emit no pair → `key_scores` row `output_status=missing`.

**Per-stage unit tests** (keyed RNG, deterministic): droplet distribution + barcodes shared
within/distinct across (unless `barcode_reuse`); **well uniformity + analytic collision rate**;
molecule recovery/survival/UMIs; free/resident split fraction; free-molecule redistribution
retains barcode+UMI; index-hop cross-well fraction; inherited-RT-error family coherence; read
layout **round-trips through `pairplex.parse_barcodes`**; config validation/OOM guard.

**Statistical single-factor tests** (assert **mechanistic statistics**, not blanket "every knob
degrades" — effects can be non-monotonic, e.g. more depth can aid recovery *or* push a
contaminant over an absolute threshold): shared barcodes no-free → occupancy/collision rate;
free no-dropout → mostly yield loss, limited mispair; free + light dropout → mispairs emerge;
↑`min_cluster_fraction` → precision/yield tradeoff; ↑`min_cluster_umis` → helps only when ambient
UMI complexity low; index hopping alone → cross-well pattern.

**Reproducibility:** identical seed → identical content, independent of chunk size/order (keyed
RNG). **Scale smoke test** at the v1 target (≤~50k cells).

## 13. Scale stance (v1: "A now, structured for B")

Reference simulator, in-memory, ~5k–50k cells; **no 1M/streaming claim for v1.** Stages partition
by `final_well` and use keyed RNG so Phase 5 (shard, stream, partitioned truth, drop full-read
Python loops) is a swap, not a rewrite.

## 14. Non-goals (Phase 1–2)

Sweep runner, alternative pairing strategies + `pairing_policy` seam, `barcode_swap`, PCR
chimeras/branch error, empirical calibration, paired-end realism, 1M streaming, and the PairPlex
hygiene fixes (separate branch). All tracked in §11, not dropped.
