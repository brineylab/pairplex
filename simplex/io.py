"""Output writers: merged per-well FASTQ files and ground-truth parquets.

Final I/O step of `run.run`, after `reads.build_merged` (reads) and `truth.py` (truth
tables) have produced their in-memory frames.
"""
import gzip
from pathlib import Path
def _tag(w): return f"well{int(w):03d}"
def write_merged_fastq(built, output_dir, compress=True):   # in-memory per-well writer (v1 scale)
    """Write `built` (the `build_merged` output: read_id/final_well/read_seq/qual) as one
    FASTQ(.gz) file per `final_well` under `output_dir/reads/well<NNN>.fastq[.gz]`.

    In-memory per-well writer sized for v1 scale (not streaming). Returns the list of
    written paths (empty list if `built` is empty).
    """
    rd=Path(output_dir)/"reads"; rd.mkdir(parents=True,exist_ok=True)
    ext="fastq.gz" if compress else "fastq"; op=(lambda p: gzip.open(p,"wt")) if compress else (lambda p: open(p,"w"))
    paths=[]
    if built.height==0: return paths
    for (well,),sub in built.group_by(["final_well"], maintain_order=True):
        p=rd/f"{_tag(well)}.{ext}"
        with op(p) as fh:
            fh.write("".join(f"@{i}\n{s}\n+\n{q}\n" for i,s,q in zip(sub["read_id"],sub["read_seq"],sub["qual"])))
        paths.append(p)
    return paths
def write_truth(output_dir, comp, cells, barcodes, reads=None):
    """Write the ground-truth tables under `output_dir/truth/`: `truth_components.parquet`,
    `truth_cells.parquet`, `truth_barcodes.parquet`, and — only if `reads` is given (i.e.
    `write_read_truth=True`) — `truth_reads.parquet` as a single parquet file (per-well
    chunking is deferred to Phase 5).
    """
    td=Path(output_dir)/"truth"; td.mkdir(parents=True,exist_ok=True)
    comp.write_parquet(td/"truth_components.parquet"); cells.write_parquet(td/"truth_cells.parquet")
    barcodes.write_parquet(td/"truth_barcodes.parquet")
    if reads is not None: reads.write_parquet(td/"truth_reads.parquet")   # single parquet in v1
