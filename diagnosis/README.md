# diagnosis/

Real-data diagnostics for PairPlex mispairing (UDA-seq style split-emulsion libraries).
Each script is standalone, single-process, and samples droplet barcodes by hash so it stays
within modest memory. Inputs are PairPlex per-well outputs (`parsed/VDJ_*.parquet`,
`metadata/VDJ_*.csv`). Results and the findings log live outside the repo
(`~/work/PairPlex/diagnosis/`).

| script | question |
|---|---|
| `noise_control.py` | identity between reads of the same molecule (noise floor) |
| `cluster_purity.py` | does PairPlex's greedy clustering merge distinct molecule species? (null-calibrated) |
| `umi_chimera.py` | do barcode+UMI families contain a second V(D)J (PCR chimeras)? |
| `umi_cross_well.py` | same molecule in several wells (post-pool swapping / index hopping) |
| `residency.py` | cross-well spread of a chain's molecules from per-well contig UMI counts (inflated by swaps) |
| `residency_dedup.py` | same, counting each molecule once at its top well → leakage estimate |
| `across_barcode.py` | chains under far more droplet barcodes than clonality allows (heavy-naive and cross-donor nulls) |
| `satellites.py` | are multi-barcode copies co-located with the parent cell's home well? (ambient pre-RT vs post-break re-barcoding) |
| `heavy_burden.py` | per-donor ambient burden from naive heavy chains (no comparator) |
| `burden_vs_viability.py` | burden vs sample viability across donors (Spearman, permutation p) |
| `homewell_pairing.py` | prototype: pool by barcode, home-well posterior, pair by shared home well; clonal-purity validation |
| `umi_integrity.py` | molecule splitting by greedy clustering + UMI error signature (Hamming-1 excess) |
| `duplicate_contigs.py` | same-CDR3 duplicate contigs per barcode-well, real vs SimPlex (realism check for the fragment model) |
| `simplex_drop_reasons.py` | ground-truth reasons per-well PairPlex drops simulated cells (collision / true extra chain / lost / spurious contig) |
| `summarize_leakage.py` | leakage by chain across reactions; light vs heavy; vs ambient burden |
| `figures.py` | static PNG figures for the deck from results/ |
| `funnel.py` | per-well read/molecule funnel through every PairPlex stage + barcode composition |

Requires `edlib`, `polars`, `pyspoa` (PairPlex deps).
