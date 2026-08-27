"""Test-only helpers for constructing deterministic mechanism fixtures.

Used by the mechanism tests (design doc S12) to build exact, hand-controlled
`cells`/`molecules`/`reads` tables that force a specific routing outcome (e.g. an ambient
mispair, an index hop), bypassing the stochastic generator stages so the downstream
FASTQ -> PairPlex -> scorer path can be exercised deterministically.
"""
import polars as pl
from .reads import build_merged
from .io import write_merged_fastq, write_truth
from .truth import build_truth_components, build_truth_cells, build_truth_barcodes

def emit(cells, chain_status, molecules, reads, out, *, write_read_truth=False):
    """Given hand-built `cells`/`chain_status`/`molecules`/`reads` tables, build truth
    tables and write merged FASTQ + truth parquets under `out`, mirroring what
    `run.run` would produce from a real generator run. Returns the `reads/` directory path.
    """
    comp=build_truth_components(cells,reads); tc=build_truth_cells(cells,chain_status,molecules,reads)
    write_merged_fastq(build_merged(reads,"TTTCTTATATGGG",0.0,False,0), out)
    write_truth(out, comp, tc, build_truth_barcodes(cells,tc,comp), reads if write_read_truth else None)
    return out/"reads"

def family(mid, cell, spid, chain, locus, seq, well, barcode, umi, is_free, is_hopped=False, hop_one_to=None):
    """Build a 4-read family (one UMI-sharing read family) for a hand-constructed molecule.

    All 4 reads share `molecule_id=mid`, the given `barcode`/`umi`/`cdna`, and
    `amplification_well=well`. If `hop_one_to` is given, exactly one of the 4 reads gets
    `final_well=hop_one_to` (and `is_index_hopped=True`) while the rest stay at `well` —
    used to construct the "route composition" fixture (one read forced to hop, the rest
    resident). Otherwise all reads' `is_index_hopped` is `is_hopped` and `final_well=well`.
    """
    fw=[well]*4
    if hop_one_to is not None: fw=[hop_one_to]+[well]*3   # route composition: one read hops
    hop=[is_hopped or (hop_one_to is not None and j==0) for j in range(4)]
    return pl.DataFrame({"read_id":[f"{mid}_{j}" for j in range(4)],"molecule_id":[mid]*4,"origin_cell_id":[cell]*4,
        "source_pair_id":[spid]*4,"chain":[chain]*4,"locus":[locus]*4,"umi":[umi]*4,"barcode":[barcode]*4,
        "amplification_well":[well]*4,"final_well":fw,"is_free":[is_free]*4,"is_index_hopped":hop,
        "cdna":[seq]*4,"n_seq_errors":[0]*4})
