import numpy as np, polars as pl
from ._rng import rng_for
from .barcodes import load_barcodes
def load_pairs(input_data, n_cells=None, seed=0):
    df=pl.read_parquet(input_data)
    req={"sequence_id:0":"chain0_id","sequence:0":"chain0_seq","sequence_id:1":"chain1_id","sequence:1":"chain1_seq"}
    miss=[k for k in req if k not in df.columns]
    if miss: raise ValueError(f"input missing {miss}")
    if "locus:0" not in df.columns or "locus:1" not in df.columns:
        raise ValueError("locus:0/1 required in Phase 1-2 (won't proceed with unknown loci)")
    out=df.select([pl.col(k).alias(v) for k,v in req.items()]+[
        (pl.col("name").cast(pl.Utf8) if "name" in df.columns else pl.int_range(pl.len()).cast(pl.Utf8)).alias("source_pair_id"),
        pl.col("locus:0").cast(pl.Utf8).alias("chain0_locus"), pl.col("locus:1").cast(pl.Utf8).alias("chain1_locus")])
    bad=out.group_by("source_pair_id").agg([pl.col(c).n_unique().alias(c) for c in
         ["chain0_seq","chain1_seq","chain0_locus","chain1_locus"]]).filter(
         (pl.col("chain0_seq")>1)|(pl.col("chain1_seq")>1)|(pl.col("chain0_locus")>1)|(pl.col("chain1_locus")>1))
    if bad.height: raise ValueError(f"{bad.height} source_pair_id(s) map to differing sequences/loci")
    if n_cells is not None:
        idx=rng_for(seed,"subsample").choice(out.height,size=n_cells,replace=n_cells>out.height); out=out[idx]
    return out.with_row_index("cell_id").select(
        ["cell_id","source_pair_id","chain0_id","chain0_seq","chain0_locus","chain1_id","chain1_seq","chain1_locus"])
