# PairPlex v2 — design notes

Design-level summary of changes motivated by a diagnosis of split-emulsion (UDA-seq style)
libraries. Study data and numbers are kept with the project, not in this repository.

## Observed failure modes (generic)
1. **Barcoded cDNA leaves fixed cells before the well split.** A droplet barcode is then not a
   cell within a well: a well holds part of each resident cell plus leaked chains of every
   droplet-mate sharing the barcode. Per-well pairing mixes cells of the same droplet.
2. **Ambient mRNA captured into other droplets before RT.** The same V(D)J appears under many
   barcodes with low support, spread uniformly over wells.
3. **Pipeline-internal errors**: no UMI error correction (inflated support), global-alignment
   clustering that merges distinct light chains sharing a V gene and splits single molecules
   across contigs, read-weighted consensus, and an exact 1H+1L rule that both mispairs and drops
   cells silently.

## Implemented
- `pairplex.pooled_pairing` / `pairplex pair-pooled`: pools a per-well run by droplet barcode,
  estimates leakage from the run, computes a home-well posterior per chain
  (`P(home=w) = r^n_w / (sum r^n_v + W - k)`, `r = (1-f+f/W)/(f/W)`), and pairs heavy and light
  chains sharing a home well with a confidence score (reciprocal best, margin reported).
- `available_cpus()`: worker count from `PAIRPLEX_MAX_WORKERS`, cgroup quota or CPU affinity.

## Planned (priority order)
1. Directional Hamming-1 UMI collapsing before any counting.
2. Per-UMI consensus before clustering; truncation-tolerant identity; junction-aware splitting.
3. Molecule-weighted consensus.
4. Scored, dominance-aware pairing everywhere; propagate support and confidence to outputs.
5. Ambient flagging across barcodes with a public-chain whitelist.
6. Read-level pooled pre-pass (from the local `pooled.py`) with per-well UMI tracking.
7. Per-stage drop counts and a QC report (leakage by chain, ambient fraction, UMI error rate,
   fragment fraction, confidence distribution).

Validation: every change is scored against SimPlex ground truth (`simplex.PRINCEPS_2025` profile
plus scenario variants, run by the validation harness kept with the diagnosis scripts).
