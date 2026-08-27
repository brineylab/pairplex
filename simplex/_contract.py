"""Frozen constants shared across the SimPlex pipeline and scorer.

These values are fixed by contract with `pairplex.parse_barcodes` (read layout) or by
the frozen scoring spec (reference-pairable minimum), and must not be changed casually —
several other modules validate against them (see `config.SimplexConfig.validate`).
"""
REF_MIN_READS = 3     # frozen reference-pairable minimum (threshold-independent)
REF_MIN_UMIS = 1
BARCODE_LEN = 16      # fixed by pairplex.parse_barcodes (s[:16])
UMI_LEN = 10          # s[16:26]
TSO = "TTTCTTATATGGG" # fixed: parse_barcodes does s[36:].lstrip("G"); arbitrary TSO corrupts cDNA
