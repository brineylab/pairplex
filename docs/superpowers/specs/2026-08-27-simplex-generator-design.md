# SimPlex — synthetic data generator for PairPlex (design spec, v2)

**Date:** 2026-08-27 (v2 after design review)
**Branch:** `simplex`
**Status:** revised design; pending review before plan rewrite
**Author:** brainstormed with bnemoz

> **v2 note:** v1 simulated the *expected conclusion* (low-fraction ambient contigs injected
> at read level) rather than the *physical mechanism*. This revision fixes the ambient model,
> the ground truth, the scorer, the metrics, and the scale claims. Guiding principle:
> **simulate the wet-lab mechanism, not the hypothesis we want to confirm.**

## 1. Motivation

Large-scale PairPlex runs mispair heavy/light chains. The investigation
(`INVESTIGATION_NOTES.md`, gitignored) showed the core pipeline is correct on clean data and
that mispairing is driven by contamination interacting with permissive filters. To choose
production thresholds we need synthetic "raw sequencing" data that reproduces the **physical
mechanism** with **known, mechanism-faithful ground truth** and **tunable knobs**, plus a
scorer that measures the actual failure and the precision/yield tradeoff.

**Circularity risk to avoid:** if the simulator injects the exact defect we hypothesize (e.g.
low-fraction ambient contigs) and the sweep then "discovers" the matching filter is optimal,
we have learned nothing. Mitigation: **Phase 0 calibrates against real data first**, and
SimPlex must reproduce several real summary statistics before any threshold is trusted.

## 2. Scope, decomposition, and roadmap

Full vision = harness: generator → scorer → sweep runner. Built in phases; **thresholds are
not selected until Phase 4.**

- **Phase 0 — Real-data audit + scoring/metric definitions.** Characterize real distributions;
  write a failure taxonomy; define pairable denominators and precision/yield metrics; specify
  ambiguity/duplicate handling. Produces the scorer *spec* and calibration targets. **Blocks
  trusting any simulated threshold.**
- **Phase 1 — Mechanistically faithful minimal generator.** Cells → overloaded droplets/
  barcodes → whole-cell well assignment → per-chain molecules/UMIs → cell-associated vs free
  molecules → free molecules reassigned to a well **retaining barcode+UMI** → molecule-level
  amplification → merged reads → compact truth. Minimal error knobs.
- **Phase 2 — Compact truth + scorer.** `truth_components.parquet`; scorer keyed by
  `(well, barcode)` with the multi-label taxonomy; implemented before sophisticated seq effects.
- **Phase 3 — Golden + single-factor tests.** Each mechanism isolated (below).
- **Phase 4 — Calibration + robust sweeps.** Bracket real distributions; sweep across params,
  seeds, held-out sequences; report a precision–yield **Pareto frontier** with CIs; **never
  optimize and evaluate on the same seeds.**
- **Phase 5 — Scale redesign.** Shard by observed well; chunk-stable RNG; streaming FASTQ;
  partitioned truth; remove full-read Python loops.

Each phase is its own spec→plan→implementation cycle. This document covers the design for all
phases but the *implementation plan* will target Phase 0–2 first.

## 3. Packaging

Sibling top-level package `simplex/`, imported as `import simplex` → `simplex.run(...)`.
May import read-structure constants / whitelists from `pairplex`.

## 4. Corrected biological model (the core fix)

The wet lab (author-confirmed): cells fixed in bulk → 10X barcode added by **RT inside
overloaded GEMs, so many cells share one barcode** → **whole fixed cells** distributed into a
96-well plate → per-well Illumina index → sequenced, demuxed upstream to one FASTX per well.

Mechanistically faithful stage order:

1. **Cells → overloaded droplets.** Partition cells into droplets; cells-per-droplet ~
   Normal(mean, sd), clamped ≥1. Each droplet gets a 10X barcode. Overloading ⇒ **multiple
   cells share a barcode.** (Optional `barcode_reuse` mode lets distinct droplets collide on a
   barcode; off by default but *available* so the simulator can test it — not suppressed by
   construction.)
2. **RT molecules.** Per cell, per chain: draw captured/not (per-chain recovery) and a molecule
   count; each molecule = one UMI, carrying the cell's **droplet barcode**.
3. **Cell-associated vs free.** Each molecule is marked `resident` (stays with its cell) or
   `free`/released with probability `release_rate`. Free molecules are the ambient pool and
   **retain their barcode and UMI**.
4. **Whole-cell → well.** Each cell → one well uniformly (`resident_well`). Resident molecules
   inherit it as `observed_well`.
5. **Free molecules → well.** Each free molecule **independently** picks an `observed_well`
   (uniform for now), **keeping barcode + UMI**. This is the only place the "where" changes.
6. **Molecule-level amplification.** Within its `observed_well`, each surviving molecule expands
   into a read family sharing one UMI (depth ~ distribution; molecule survival applied *before*
   amplification — see §7). PCR lineage id retained for later.
7. **Sequencing errors** per read (independent); molecule-level RT error optionally applied at
   step 2 so it is **inherited** by the whole read family (§7).
8. **Index hopping** post-amplification: a fraction of reads move to a different `observed_well`
   (barcode retained), flagged `index_hop`.
9. *(Deferred)* **`barcode_swap`** (PCR recombination / index swap that actually changes the
   barcode) — a **separate** knob, later; explicitly **not** called "ambient."

**Why the stage/order matters:** contamination must enter as **whole molecules pre-PCR**, so an
ambient molecule produces a UMI-coherent read family. Read-level reassignment (v1) scatters PCR
siblings across barcodes and would give wrong conclusions about `min_cluster_reads` /
`min_cluster_umis`, which count reads and distinct UMIs. Emergent consequences of this model:
resident-cell contamination by a same-barcode droplet mate; ambient-only `(well, barcode)`
populations (no explicit knob needed); cross-well contamination via free molecules + hopping;
realistic UMI families; support values usable to assess the thresholds under study.

**Key correctness fact:** with `cells_per_droplet == 1` there are no droplet mates, so every
molecule under barcode X (resident or free) comes from the same cell → a coherent (correct)
pair. **Mispairs require `cells_per_droplet > 1` + dropout.** (v1's degradation test using
`cells_per_droplet_mean=1` was testing the wrong thing.)

## 5. Provenance & ground truth

**Molecule-level provenance** (the atomic record), minimally:

```
molecule_id, origin_cell_id, origin_droplet_id, source_pair_id, barcode, chain, umi,
resident_well, observed_well, route (resident|ambient|index_hop|barcode_swap),
parent_molecule_id, n_umis, n_reads
```

**Ground-truth outputs:**

- **`truth_components.parquet`** *(primary; small; the scorer's main input)* — one row per
  `(observed_well, barcode, origin_cell_id, chain)`:
  `observed_well, barcode, origin_cell_id, source_pair_id, chain, sequence, route,
  n_molecules, n_umis, n_reads, is_resident_source`.
  Far smaller than per-read truth, far more useful than `truth_cells`+`truth_barcodes` alone.
- **`truth_cells.parquet`** — one row per simulated cell with the **specified schema**:
  `cell_id, source_pair_id, droplet_id, barcode, resident_well, chain{0,1}_id, chain{0,1}_seq`,
  and per chain: `captured, n_molecules, n_umis, n_reads_generated, n_reads_resident,
  n_reads_ambient_out, n_reads_index_hopped_out`.
- **`truth_barcodes.parquet`** — per `(observed_well, barcode)`: resident source set,
  `n_resident_cells`, `is_collision` (≥2 resident cells), **per-locus** `dominant_heavy_source`
  and `dominant_light_source` (by observed reads/UMIs — dominance is per locus because the
  dominant heavy and light source can differ, which is exactly the mispair condition),
  `is_ambient_only` (reads present, no resident cell).
- **`truth_reads.parquet`** *(optional, `write_read_truth=False`)* — per-read provenance,
  chunked per well.
- **`simplex_config.json`** — knobs + resolved seed + versions.

## 6. Scorer specification (Phase 0 defines, Phase 2 implements)

**Keyed by `(observed_well, barcode)`.** For each PairPlex output pair, read its `(well,
barcode)` (well from the source file, barcode from the pair name), look up resident source(s)
from truth, and match each chain sequence to candidate origins.

**Sequence matching returns a SET of candidate `source_pair_id`s, never one cell.** Duplicate/
shared sequences (guaranteed by oversampling-with-replacement, clonal expansion, shared chains)
make single-cell mapping wrong. Use `source_pair_id` equivalence: repeated copies of one source
pair are sequence-equivalent; a sequence compatible with multiple distinct source pairs is
**ambiguous** and must not be silently scored. Clean tests use exact/substring matching; noisy
tests need orientation-aware alignment / edit-distance with an explicit ambiguity rule so a
residual consensus error is not miscounted as a false positive.

**Per-pair labels (aggregate into production definitions later):**
`resident_correct`, `resident_mispaired`, `resident_plus_ambient`, `ambient_coherent`,
`ambient_mispaired`, `collision_ambiguous`, `sequence_ambiguous`, `unmatchable`,
`duplicate_output`.

**Denominators / metrics (report all; "best threshold" is not a scalar):**
biological recovery (cells with both chains captured at all); technical observability (singleton
resident cells producing ≥1 parseable read/molecule for both chains); algorithmic recall
(correct pairs among technically observable singletons); pair precision (resident-correct among
returned pairs); mispair rate; rejection/yield loss (observable cells lost to extra contigs,
clustering splits, pairing logic); collision performance (behavior when two same-barcode cells
share a well). Sweeps produce a **precision–yield Pareto frontier**; a production default is
chosen for a target precision / max mispair rate.

## 7. Molecule/read-count and error model

**Support model (Phase 1 minimal, Phase 4 calibrated).** Minimal: per-chain recovery Bernoulli
→ molecule count per recovered chain → **molecule survival Bernoulli applied before
amplification** (fixes the v1 semantic bug where `seq_efficiency` thinned *reads* and corrupted
observed-UMI counts) → per-molecule read depth. Phase 4 adds, as needed: separate heavy/light
recovery; a **cell-level abundance factor** shared by both chains (correlated dropout/support);
chain-specific abundance; negative-binomial / log-normal counts; PCR jackpotting or an empirical
reads-per-UMI distribution; well-level depth variation; empirical support distributions loaded
from real metadata. These live in the **hypothesis matrix** (§10), not the Phase 1 build.

**Layered, inheritance-aware errors.** RT/molecule error (inherited by every read of a UMI);
PCR error (inherited by a PCR-family branch); sequencing error (independent per read, quality-
linked). Phase 1 must at least support **molecule-level inherited RT error**, otherwise
consensus removes nearly all independent errors and the `clustering_threshold` sweep is
unrealistically easy. Full PCR-branch simulation is deferred.

## 8. Read assembly, paired-end, and fastp validation

Merged read layout matches `pairplex.parse_barcodes`: `barcode(16)+umi(10)+TSO+cDNA`.
Paired mode must be **actually validated**, not assumed: model R1/R2 with realistic overlap,
adapter read-through, `rc_fraction` applied in paired mode too, and **long inserts that fail to
merge**. The paired golden test writes R1/R2 and calls `pairplex.run(..., merge_paired_reads=
True)` with a mix of mergeable and non-mergeable fragments, asserting actual fastp retention.
If paired realism slips, v1 may ship **merged-only** and drop the paired realism claims until
the validated path lands (Phase 3). Illumina filename tokens verified against
`abstar.pp.merge_fastqs`, not guessed.

## 9. Controlled sequence-similarity scenarios

Random sampling won't stress `clustering_threshold`. Provide curated input scenarios: unrelated
H/L; near-identical clonal variants; shared heavy + different lights; shared light + different
heavies; highly similar sequences from different same-barcode cells; truncated templates; exact
duplicate source pairs (clonal expansion). These both exercise the threshold near hard clonal
families and force correct sequence-ambiguity handling in the scorer.

## 10. Hypothesis matrix (kept visible, phased — not silently "solved")

| Phenomenon | Status | Phase |
|---|---|---|
| Overloaded shared-barcode droplet mates | core mechanism | 1 |
| Free-molecule ambient (barcode retained, well changes) | core mechanism | 1 |
| Index hopping (cross-well) | core | 1 |
| Molecule-level inherited RT error | core | 1 |
| `(well,barcode)` scorer + taxonomy + denominators | core | 0/2 |
| Chain dropout / recovery | core | 1 |
| Barcode reuse across droplets | optional mode, default off | 1 (mode) / 4 |
| `barcode_swap` / PCR recombination | deferred, separate knob | 4 |
| PCR chimeras (H-L fusion) | deferred; **must appear in final root-cause eval** | 4 |
| True biological extra chains (dual-light, nonproductive, alt heavy) | ≥1 explicit scenario | 3/4 |
| Correlated abundance, NB/log-normal, jackpotting, empirical calibration | calibration | 4 |
| Well/cell heterogeneity (uneven loading, quality, H/L asymmetry) | robustness | 4 |

Rationale for a few: dual-chain biology matters because a "drop every 1H+2L" or "pick dominant
light" strategy can erase genuine dual-chain cells — that tradeoff must be visible. PCR chimeras
produce **high-support** wrong chains that a cluster-fraction filter won't catch like low-level
ambient does, so they must be in the final evaluation even if deferred from the generator MVP.

## 11. Testing plan

- **Per-stage unit tests** (deterministic via keyed RNG): droplet/barcode assignment
  (cells-per-droplet distribution; barcodes shared within droplet; distinct across, unless
  `barcode_reuse`); **well uniformity + analytic collision rate** (not just range); molecule
  recovery/survival/UMI counts; free-vs-resident split fractions; free-molecule well
  redistribution retains barcode+UMI; index-hop cross-well fraction; inherited-RT-error family
  coherence; read layout **round-trips through `pairplex.parse_barcodes`**.
- **Golden + single-factor mechanistic tests** (assert **mechanistic statistics**, not blanket
  "every knob degrades" — some effects are non-monotonic, e.g. more depth can either aid
  recovery or push a contaminant over an absolute threshold):
  1. Clean singleton barcodes → ~100% `resident_correct`.
  2. Shared barcodes, no free molecules → collision rate matches expectation.
  3. Free molecules, no dropout → mostly yield loss via extra contigs, limited mispairing.
  4. Free molecules + light dropout → mispairs emerge.
  5. ↑`min_cluster_fraction` → expected precision/yield tradeoff.
  6. ↑`min_cluster_umis` → helps only when ambient UMI complexity is low.
  7. Index hopping alone → predicted cross-well pattern.
  8. Paired-end path actually passes through fastp (mergeable + non-mergeable).
- **Reproducibility:** identical seed → identical content (compare decompressed FASTQ + parquet;
  RNG keyed by `(seed, stage, well, chunk)` so results are independent of chunk size / order).
- **Scale smoke test** at the v1 target (≤~50k cells) within a time/memory budget.

## 12. Scale stance (v1: "A now, structured for B")

Reference simulator, in-memory, correctness-first, target ~5k–50k cells. **No 1M/streaming
claim for v1.** But stage interfaces are **shard-friendly**: functions partition cleanly by
`observed_well`, and all randomness derives from keys `(seed, stage, well, chunk)` so Phase 5
converts to per-well sharded, streaming, partitioned-truth execution as a swap, not a rewrite.

## 13. Non-goals (near-term)

Sweep runner (Phase 4 tooling), `barcode_swap`, PCR chimeras and full PCR-branch error, empirical
distribution calibration, and 1M-scale streaming are **not** in the Phase 1–2 build. They are
tracked in §10 and the roadmap, not dropped.
