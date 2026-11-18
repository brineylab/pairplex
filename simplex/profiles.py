"""Calibrated knob profiles for `simplex.run(**profile)`.

`PRINCEPS_2025` encodes what the real-data diagnosis measured on the 12-donor PairPlex princeps
study (UDA-seq style: overloaded 10x HT 5' lane, ~1M fixed cells per reaction, 96-well split,
pools of 24). Sources are the diagnosis scripts (not shipped with the package); values are medians/ballparks, not
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
- fragments are mostly C-region +/- J (lost heavy contigs median 108 nt, 79% without CDR3) and
  rarely become annotated contigs: same-CDR3 duplicate contigs in 1.5-3.1% of real barcode-wells
  with H and L, at full length -> fragment_keep (0.05, 0.35) of the V(D)J (v2b's 0.3-0.9 gave
  63% in the clean scenario, half-length)

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
    fragment_keep=(0.05, 0.35),
    umi_sub_rate=0.004,
)

__all__ = ["PRINCEPS_2025"]


# --- SimPlex v2d: profile family calibrated to all 23 princeps reactions (2026-11-17) --------------------------------
# Targets: diagnosis/simplex_v2d/stats/princeps_targets.csv + umi_inflation_real.csv (existing per-well PairPlex outputs,
# default settings). Each profile is checked against the 3 real reactions of its kind with the closest Ig UMIs per
# barcode (family_compare.py): Ig UMIs and wells per barcode, light:heavy UMIs, 1-UMI contig shares, 1H+1L and
# multi-chain barcode-well shares, reads and UMIs per contig by chain. Key v2d findings encoded here:
# - real cells carry ~3x more molecules than v2c (Ig UMIs per barcode median 26 memory / 30 naive; P10 9-17, P90 38);
# - heavy molecules are sequenced ~5x deeper than light (reads per UMI 17-30 vs 3.4-4.5), with heavy-tailed families;
#   their jackpots spawn the error UMIs that make heavy contigs look multi-molecule (barnyard: 42% of raw heavy UMIs
#   are errors), so heavy depth + umi_sub_rate, not extra molecules, reproduce the heavy contig depth;
# - leakage per kind from the 23-reaction medians (memory H .52 / L .85; naive H .57 / L .83).
# P10/P50/P90 = Ig UMIs per barcode quantiles of the reactions of that kind. Pipeline choices should be robust across
# the family, not tuned to one point. Step-3 knobs (junk, read structure, clones) are added in PRINCEPS_2026_FULL_*.
_V2D_BASE = dict(PRINCEPS_2025, reads_per_molecule_dispersion=2.5, umi_sub_rate=0.0045)
_V2D_MEM = dict(_V2D_BASE, reads_per_molecule_mean=20.0, reads_per_molecule_mean_light=2.0, release_rate_heavy=0.52, release_rate_light=0.85)
_V2D_NAI = dict(_V2D_BASE, reads_per_molecule_mean=30.0, reads_per_molecule_mean_light=2.5, release_rate_heavy=0.57, release_rate_light=0.83)
PRINCEPS_2026_MEMORY_P10 = dict(_V2D_MEM, molecules_per_chain_mean=4.5, molecules_per_chain_mean_light=13.0)
PRINCEPS_2026_MEMORY_P50 = dict(_V2D_MEM, molecules_per_chain_mean=11.0, molecules_per_chain_mean_light=31.0)
PRINCEPS_2026_MEMORY_P90 = dict(_V2D_MEM, molecules_per_chain_mean=15.0, molecules_per_chain_mean_light=44.0)
PRINCEPS_2026_NAIVE_P10 = dict(_V2D_NAI, molecules_per_chain_mean=8.3, molecules_per_chain_mean_light=19.0)
PRINCEPS_2026_NAIVE_P50 = dict(_V2D_NAI, molecules_per_chain_mean=13.7, molecules_per_chain_mean_light=31.0)
PRINCEPS_2026_NAIVE_P90 = dict(_V2D_NAI, molecules_per_chain_mean=17.0, molecules_per_chain_mean_light=38.0)
PRINCEPS_2026_FAMILY = {"memory_p10": PRINCEPS_2026_MEMORY_P10, "memory_p50": PRINCEPS_2026_MEMORY_P50, "memory_p90": PRINCEPS_2026_MEMORY_P90,
                        "naive_p10": PRINCEPS_2026_NAIVE_P10, "naive_p50": PRINCEPS_2026_NAIVE_P50, "naive_p90": PRINCEPS_2026_NAIVE_P90}
__all__ = ["PRINCEPS_2025", "PRINCEPS_2026_FAMILY"] + [f"PRINCEPS_2026_{k.upper()}" for k in PRINCEPS_2026_FAMILY]

# --- SimPlex v2d step 3: full mechanism set, princeps memory median (2026-11-17) ------------------------------------
# Adds to PRINCEPS_2026_MEMORY_P50: non-Ig library molecules (share of reads reaching productive Ig contigs 0.41 vs real
# memory median 0.47), constant region to the primer (measured extents; donor02 memory isotype mix, mostly IgA), barcode
# errors (1.7% of real reads need Hamming-1 correction), TSO body errors + indels and the measured G-run distribution
# (real merged reads: 60% exact body, ~20% indel-like, G-run after an exact body mostly 4-5), RT errors, and clonal
# sampling of real lineages. Use with input = jaffe_paired_lineages.parquet (Jaffe paired + donor/clone_id/is_memory,
# built by diagnosis/simplex_v2d/prep_jaffe_lineages.py) and input_filter={"donor": "d1", "is_memory": True}.
# Remaining gaps (see diagnosis/simplex_v2d/SIMPLEX_V2D.md): share of 1-read UMIs (0.63 vs 0.44-0.54) and pairing:
# simulated clonal purity per-well 0.62 / pooled 0.67 vs real 0.48-0.51 / 0.41-0.44 (real pooled < per-well).
PRINCEPS_2026_FULL_MEMORY_P50 = dict(
    PRINCEPS_2026_MEMORY_P50, reads_per_molecule_dispersion=1.5, junk_rate=0.5, constant_region=True,
    heavy_isotype_mix={"IGHA1": 0.6, "IGHG1": 0.15, "IGHM": 0.25}, barcode_sub_rate=0.0011, tso_sub_rate=0.028,
    tso_indel_rate=0.2, tso_g_run_probs=[0.068, 0.085, 0.048, 0.081, 0.237, 0.311, 0.055, 0.091, 0.024],
    rt_sub_rate=0.0001, sample_clone_alpha=1.5)
__all__ = __all__ + ["PRINCEPS_2026_FULL_MEMORY_P50"]
