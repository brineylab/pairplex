# SimPlex — synthetic data generator for PairPlex (design spec)

**Date:** 2026-08-27
**Branch:** `simplex`
**Status:** approved design; ready for implementation plan
**Author:** brainstormed with bnemoz

## 1. Motivation

Large-scale runs show PairPlex mispairing antibody heavy/light chains. Investigation
(see repo `INVESTIGATION_NOTES.md`, gitignored) established that the core pairing pipeline
is correct on clean data, and that mispairing is driven by **within-barcode / cross-well
contamination** (ambient RNA, leakage, collisions) passing PairPlex's near-disabled default
filters. We could only reproduce this by hand-injecting contamination.

We need a principled way to generate realistic "raw sequencing" data with **known ground
truth** and **tunable knobs**, so we can run PairPlex under many configurations and find the
thresholds/approach that maximize correct pairing and yield.

**SimPlex** (simulated PairPlex) is that tool.

## 2. Scope and decomposition

The full vision is a **harness**: generator → scorer → parameter-sweep runner. We build it
**step by step**. This spec covers **only the generator** (sub-project 1). The generator's
ground-truth outputs are explicitly designed to feed the future scorer without rework.

Deferred to later sub-projects (their own spec → plan cycles):
- **Scorer**: compares PairPlex output to ground truth → correct / mispaired / false-positive,
  yield, precision/recall.
- **Sweep runner**: loops PairPlex configs over generated data → comparison tables/plots.

## 3. Packaging

New **sibling top-level package** `simplex/` in this repo, alongside `pairplex/`. Imported as
`import simplex` → `simplex.run(...)`. Keeps the test tool cleanly separated from shipped
PairPlex code; gets its own CLI (`simplex ...`) later. It may import read-structure constants
from `pairplex` where useful (barcode/UMI layout, whitelist paths) to stay in lockstep with
what `pairplex.parse_barcodes` expects.

## 4. Architecture

**Staged vectorized pipeline.** A chain of pure functions, each taking and returning a polars
DataFrame (or writing a shard), using polars + numpy for vectorized generation, with FASTQ
written **streaming per well**. Fully reproducible from a single `seed`. Chosen over per-cell
Python generators (too slow at 1M cells) and a declarative config engine (over-abstracted).

Target scale: **100k–1M cells**, i.e. hundreds of millions of reads. Implications:
- Vectorize with numpy/polars; avoid per-read Python loops in hot paths.
- Stream output per well (never hold all reads in memory); chunk within a well if needed.
- `truth_reads` (per-read provenance) is huge, so it is **optional** (`write_read_truth=False`
  by default) and written chunked per well.

## 5. Simulation pipeline

The wet lab is modeled as an ordered pipeline. Each stage is an **independently testable
function** taking/returning a polars frame.

1. **`load_pairs(input_data, n_cells, seed)`** — read the input paired parquet
   (`sequence_id:0, sequence:0, sequence_id:1, sequence:1` required) → true-cell table
   `{cell_id, source_pair_id, chain0_id, chain0_seq, chain1_id, chain1_seq}`. Locus-agnostic:
   a row is "two sequences belonging to the same cell"; PairPlex re-annotates H/L downstream.
   `n_cells=None` uses all input rows; an integer subsamples (or oversamples with replacement
   if `n_cells` > input size).
2. **`assign_droplets_and_barcodes(cells, cells_per_droplet_mean, cells_per_droplet_sd,
   chemistry, seed)`** — group cells into droplets; cells-per-droplet ~ Normal(mean, sd),
   clamped ≥1 and rounded; each droplet draws a **unique** 10X barcode from the real whitelist
   for `chemistry`. Cells in the same droplet **share** a barcode → this is where overloading
   creates shared barcodes.
3. **`assign_wells(cells, wells, seed)`** — each whole cell → one of `wells` wells uniformly.
   Within-well barcode collisions emerge naturally when two same-droplet cells land in one well.
4. **`generate_molecules(cells, recovery_rate, molecules_per_chain_mean, seed)`** — per cell,
   per chain: capture with probability `recovery_rate` (models chain dropout). If captured,
   draw a molecule (UMI) count ~ (e.g. Poisson `molecules_per_chain_mean`); each molecule gets
   a random UMI of `umi_length`.
5. **`amplify_and_sequence(molecules, reads_per_molecule_mean, seq_efficiency, seed)`** — each
   molecule → reads; depth ~ `reads_per_molecule_mean`, thinned by `seq_efficiency`. Emits the
   read table `{read_id, well, barcode, umi, cdna, true_cell_id, chain, is_ambient=False,
   is_leakage=False, is_index_hopped=False}`.
6. **`inject_ambient(reads, ambient_rate, leakage_rate, seed)`** — **both** mechanisms:
   (a) **well soup**: an `ambient_rate` fraction of each well's reads are reassigned a random
   *wrong* barcode present in that well; (b) **cross-barcode leakage**: each cell leaks a
   `leakage_rate` fraction of its reads into other random barcodes. Both flagged in ground
   truth. Also produces **empty/ambient-only barcodes** as a natural consequence (barcodes that
   end up carrying only reassigned reads); optionally seeded explicitly via a small
   `ambient_only_barcodes` knob.
7. **`index_hop(reads, index_hop_rate, wells, seed)`** — a small `index_hop_rate` fraction of
   reads are reassigned to a **different well** (Illumina index misassignment), flagged
   `is_index_hopped`. Creates cross-well contamination.
8. **`apply_sequencing_errors(reads, sub_rate, indel_rate, errors_per_read, regions, seed)`** —
   vectorized substitutions at per-base `sub_rate` plus indels at `indel_rate`; optional
   per-read count override `errors_per_read`. Applied to configurable `regions`
   (barcode/UMI/cDNA). Records `n_errors` per read.
9. **`build_reads(reads, output_mode, read_length, rc_fraction, barcode_length, umi_length,
   tso, seed)`** — assemble the PairPlex read layout `[barcode][UMI][TSO][cDNA]`; `rc_fraction`
   flips reads to reverse complement. Variable cDNA/read length supported (5'/3' truncation).
   For `output_mode='paired'`, split each fragment into R1/R2 with realistic overlap so fastp
   can merge; fragments too long to overlap simply fail to merge (realistically lost).
10. **`write_output(...)`** — stream per-well FASTQ(.gz), Illumina-named in paired mode so
    `pairplex`/`abstar.pp.merge_fastqs` pairs them; write ground-truth files; write config.

## 6. Public API (knobs)

```python
simplex.run(
    input_data,                       # path (or list) to paired parquet
    output_directory,
    # sampling
    n_cells=None,                     # None = all input pairs; int subsamples/oversamples
    # overloading + shuffling
    wells=96,
    cells_per_droplet_mean=5,
    cells_per_droplet_sd=2,
    # capture / depth
    recovery_rate=0.5,                # per-chain capture prob (dropout)
    molecules_per_chain_mean=10,      # UMI diversity per captured chain
    reads_per_molecule_mean=5,        # amplification depth
    seq_efficiency=0.8,               # fraction of molecules yielding reads
    # contamination
    ambient_rate=0.02,                # well-soup fraction
    leakage_rate=0.01,                # cross-barcode leakage fraction
    ambient_only_barcodes=0,          # extra background-only barcodes to seed
    index_hop_rate=0.001,             # cross-well misassignment fraction
    # errors (layered)
    sub_rate=0.001,                   # per-base substitution rate
    indel_rate=0.0,                   # per-base indel rate
    errors_per_read=None,             # optional per-read Poisson count override
    error_regions=("cdna",),          # which regions to mutate: barcode|umi|cdna
    # read structure
    barcode_length=16, umi_length=10, tso="TTTCTTATATGGG",
    chemistry="v2",                   # picks whitelist + structure (v2/v3)
    output_mode="paired",             # "paired" | "merged"
    read_length=300, rc_fraction=0.0, platform="illumina",
    variable_length=True,             # enable 5'/3' truncation of cDNA
    # bookkeeping
    write_read_truth=False,           # emit per-read truth parquet (huge)
    seed=0,
)
```

Defaults are starting guesses to be tuned; they are not claims about real-run values.

## 7. Ground truth outputs

Designed for the future scorer, which labels each PairPlex pair (keyed by `well + barcode`).

- **`truth_cells.parquet`** — one row per simulated cell:
  `cell_id, source_pair_id, chain0_id, chain0_seq, chain1_id, chain1_seq, droplet_id, barcode,
  well, captured_chain0, captured_chain1, n_reads_chain0, n_reads_chain1`.
  Canonical definition of a correct pair.
- **`truth_barcodes.parquet`** — per `(well, barcode)`:
  `well, barcode, true_cell_ids (list), n_true_cells, is_collision, dominant_cell,
  is_ambient_only`. The scorer joins PairPlex pairs against this.
- **`truth_reads.parquet`** *(optional; `write_read_truth=True`)* — one row per read:
  `read_id, well, barcode, umi, true_cell_id, chain, is_ambient, is_leakage, is_index_hopped,
  n_errors`. Written chunked per well.
- **`simplex_config.json`** — all knob values + resolved seed + SimPlex/PairPlex versions.

**Scoring contract (future scorer, informs GT):** a PairPlex pair at `(well, barcode)` is
**correct** iff both its chains match (by sequence identity, not junction) the same true cell
in `truth_cells` for that `(well, barcode)`; **mispaired** if the two chains trace to different
true cells; **false-positive** if the barcode is `is_ambient_only`.

## 8. Output layout

```
output_directory/
  reads/                              # pass this to pairplex.run(sequences=...)
    well001_S1_L001_R1_001.fastq.gz
    well001_S1_L001_R2_001.fastq.gz   # paired mode (Illumina-named for merge_fastqs)
    ...                               # or well001.fastq.gz per well in merged mode
  truth/
    truth_cells.parquet
    truth_barcodes.parquet
    truth_reads.parquet               # optional
  simplex_config.json
```

## 9. Read structure & fastp compatibility

Must match what `pairplex.parse_barcodes` consumes: `barcode = s[:16]`, `umi = s[16:26]`,
`sequence = s[36:].lstrip("G")`. Merged reads are `[16bp barcode][10bp UMI]["TTTCTTATATGGG"]
[cDNA]`. Paired mode splits this fragment into R1 (barcode+UMI+TSO+5' cDNA) and R2 (RC of 3'
cDNA) sized by `read_length`, with enough overlap for fastp to merge.

**Open implementation detail (verify, don't guess):** the exact filename tokens
(`_S#_L00#_R1/R2_001`) and any lane/sample expectations of `abstar.pp.merge_fastqs` for the
`illumina` (and `element`) schema — confirm against its source during implementation and match
one well = one "sample".

## 10. Phenomena: included vs deferred

Included in first generator: overloading (shared barcodes), well shuffling + collisions,
chain dropout, UMIs, amplification/efficiency, **well-soup ambient + cross-barcode leakage**,
**index hopping (cross-well)**, **variable read/cDNA length**, **empty/ambient-only barcodes**,
layered sequencing errors (sub + indel + optional per-read count), read orientation
(`rc_fraction`), paired/merged output.

Deferred: **PCR chimeras** (template-switching heavy-light fusions).

## 11. Testing plan

- **Per-stage unit tests**, deterministic via seed:
  - droplet/barcode assignment: cells-per-droplet distribution ≈ target; barcodes unique per
    droplet and shared within droplet.
  - well assignment: uniform; collision rate ≈ analytic expectation.
  - molecule generation: `recovery_rate` dropout fraction and UMI counts honored.
  - ambient/leakage: injected fractions match knobs and are flagged in GT.
  - index hopping: cross-well fraction matches knob.
  - errors: realized rates ≈ `sub_rate`/`indel_rate`; region targeting respected.
  - **round-trip**: `build_reads` output parses back correctly through
    `pairplex.parse_barcodes` (barcode, UMI, cDNA recovered).
- **Golden integration test (key invariant):** a clean run (`ambient_rate=0, leakage_rate=0,
  index_hop_rate=0`, zero error rates, `cells_per_droplet=1`, `recovery_rate=1`,
  `variable_length=False`) fed to `pairplex.run` yields ~100% correct pairs, scored by
  **sequence matching (never junction_aa)**. Then increasing each knob degrades results in the
  predicted direction.
- **Reproducibility:** identical `seed` → identical output *content* (same reads/records and
  ground truth; compare decompressed FASTQ and parquet, since gzip embeds mtime).
- **Scale smoke test:** e.g. 50k cells within an agreed time/memory budget; confirms streaming
  writes and vectorized paths hold.

## 12. Non-goals (this sub-project)

- No scorer or sweep runner yet (separate sub-projects).
- No PCR chimera model.
- Not attempting to reproduce a specific real chemistry's exact error/UMI distributions; knobs
  are approximate and tunable.
