import gzip
from pathlib import Path
def _tag(w): return f"well{int(w):03d}"
def write_merged_fastq(built, output_dir, compress=True):   # in-memory per-well writer (v1 scale)
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
    td=Path(output_dir)/"truth"; td.mkdir(parents=True,exist_ok=True)
    comp.write_parquet(td/"truth_components.parquet"); cells.write_parquet(td/"truth_cells.parquet")
    barcodes.write_parquet(td/"truth_barcodes.parquet")
    if reads is not None: reads.write_parquet(td/"truth_reads.parquet")   # single parquet in v1
