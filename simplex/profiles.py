"""Calibrated knob profiles for `simplex.run(**profile)`.

`PRINCEPS_2025` encodes what the real-data diagnosis measured on the 12-donor PairPlex princeps
study (UDA-seq style: overloaded 10x HT 5' lane, ~1M fixed cells per reaction, 96-well split,
pools of 24). Sources are in the repo's `diagnosis/` scripts; values are medians/ballparks, not
fits — sweep around them.

- release (pre-split leakage of barcoded cDNA out of fixed cells): medians over 23 reactions
  heavy 0.53, kappa 0.86, lambda 0.84 (light > heavy in every reaction, p < 1e-4); naive and memory
  reactions do not differ  [residency_dedup.py, summarize_leakage.py]
- ambient (pre-RT capture of free mRNA into other droplets): ~20-27% of naive chain occurrences
  are copies under other barcodes  [across_barcode.py, satellites.py]; ambient_rate ~0.3
- loading: ~1M cells over ~456k observed barcodes -> lambda ~2.2
- depth: 70% of passing contigs are single-UMI; reads/UMI median ~6-11
- light template ~8x heavy by qPCR (library level); light molecules per chain set 2x heavy
- per-read divergence of reads from one molecule ~2.3% (incl. PCR) -> sub 0.008, indel 0.0005
- index hopping / post-pool swapping: ~1.7% of reads leave the molecule's top well
- UMI errors: 20.7% of real UMIs are 1-read Hamming-1 neighbours of a >=5-read UMI (x3,520 the
  random rate) -> umi_sub_rate 0.004 per base per read  [umi_integrity.py]
- fragmented RNA: real heavy reads are mostly short 5'-truncated fragments (constant region, no V
  start; 75% of reads lost at annotation are heavy) -> fragment_rate_heavy 0.6, light 0.3  [funnel.py]

Scale knobs (`n_cells`, `wells`) are left to the caller: real wells hold ~10k cells, which is not
tractable; collisions depend on lambda and wells, not on cells per well.
"""
PRINCEPS_2025 = dict(
    cells_per_droplet_mean=2.2,
    recovery_rate=0.6,
    molecules_per_chain_mean=2.0,
    molecules_per_chain_mean_light=4.0,
    molecule_survival_rate=0.8,
    reads_per_molecule_mean=8.0,
    release_rate=0.8,
    release_rate_heavy=0.55,
    release_rate_light=0.85,
    ambient_rate=0.3,
    index_hop_rate=0.017,
    sequencing_sub_rate=0.008,
    sequencing_indel_rate=0.0005,
    truncation="fragment_5p",
    fragment_rate=0.3,
    fragment_rate_heavy=0.6,
    umi_sub_rate=0.004,
)

__all__ = ["PRINCEPS_2025"]
