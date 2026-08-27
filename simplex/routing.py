import numpy as np, polars as pl
from ._rng import rng_for
_READS_SCHEMA={"read_id":pl.Utf8,"molecule_id":pl.Int64,"origin_cell_id":pl.Int64,"source_pair_id":pl.Utf8,
    "chain":pl.Int8,"locus":pl.Utf8,"umi":pl.Utf8,"barcode":pl.Utf8,"amplification_well":pl.Int64,
    "final_well":pl.Int64,"is_free":pl.Boolean,"is_index_hopped":pl.Boolean,"cdna":pl.Utf8,"n_seq_errors":pl.Int64}
def route_and_amplify(molecules,wells,molecule_survival_rate,reads_per_molecule_mean,index_hop_rate,seed):
    rng=rng_for(seed,"routing"); n=molecules.height
    free=molecules["is_free"].to_numpy() if n else np.array([],bool)
    amp=(np.where(free,rng.integers(0,wells,size=n),molecules["resident_well"].to_numpy()).astype(np.int64) if n else np.array([],np.int64))
    surv=rng.random(n)<molecule_survival_rate if n else np.array([],bool)
    mols=molecules.with_columns([pl.Series("amplification_well",amp),pl.Series("survived",surv)]) if n else molecules
    survd=mols.filter(pl.col("survived"))
    if survd.height==0:
        return mols, pl.DataFrame(schema=_READS_SCHEMA)
    depth=np.maximum(rng.poisson(reads_per_molecule_mean,survd.height),1).astype(np.int64)
    rep=np.repeat(np.arange(survd.height),depth); reads=survd[rep]; k=reads.height
    hop=rng.random(k)<index_hop_rate; off=rng.integers(1,max(2,wells),size=k); a=reads["amplification_well"].to_numpy()
    final=np.where(hop,(a+off)%wells,a).astype(np.int64)
    reads=reads.with_columns([pl.Series("read_id",[f"r{i}" for i in range(k)]),pl.Series("final_well",final),
        pl.Series("is_index_hopped",hop),pl.lit(0,pl.Int64).alias("n_seq_errors")]).select(list(_READS_SCHEMA.keys()))
    return mols, reads
