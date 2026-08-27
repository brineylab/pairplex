import polars as pl
from .reads import build_merged
from .io import write_merged_fastq, write_truth
from .truth import build_truth_components, build_truth_cells, build_truth_barcodes

def emit(cells, chain_status, molecules, reads, out, *, write_read_truth=False):
    comp=build_truth_components(cells,reads); tc=build_truth_cells(cells,chain_status,molecules,reads)
    write_merged_fastq(build_merged(reads,"TTTCTTATATGGG",0.0,False,0), out)
    write_truth(out, comp, tc, build_truth_barcodes(cells,tc,comp), reads if write_read_truth else None)
    return out/"reads"

def family(mid, cell, spid, chain, locus, seq, well, barcode, umi, is_free, is_hopped=False, hop_one_to=None):
    fw=[well]*4
    if hop_one_to is not None: fw=[hop_one_to]+[well]*3   # route composition: one read hops
    hop=[is_hopped or (hop_one_to is not None and j==0) for j in range(4)]
    return pl.DataFrame({"read_id":[f"{mid}_{j}" for j in range(4)],"molecule_id":[mid]*4,"origin_cell_id":[cell]*4,
        "source_pair_id":[spid]*4,"chain":[chain]*4,"locus":[locus]*4,"umi":[umi]*4,"barcode":[barcode]*4,
        "amplification_well":[well]*4,"final_well":fw,"is_free":[is_free]*4,"is_index_hopped":hop,
        "cdna":[seq]*4,"n_seq_errors":[0]*4})
