"""Stage 3 of the pipeline: well routing/amplification for molecules, then per-read index hopping.

Consumes the `molecules` frame from `molecules.py`, assigns each molecule an
`amplification_well` (resident molecules stay in their cell's well; free molecules pick a
random well, keeping barcode+UMI), applies molecule survival before expanding surviving
molecules into read families, and finally lets individual reads index-hop to a different
`final_well`. Feeds `reads.py` next in the cells -> molecules -> routing -> reads -> truth
-> scoring pipeline.
"""
import numpy as np, polars as pl
from ._rng import rng_for
_READS_SCHEMA={"read_id":pl.Utf8,"molecule_id":pl.Int64,"origin_cell_id":pl.Int64,"source_pair_id":pl.Utf8,
    "chain":pl.Int8,"locus":pl.Utf8,"umi":pl.Utf8,"barcode":pl.Utf8,"amplification_well":pl.Int64,
    "final_well":pl.Int64,"is_free":pl.Boolean,"is_index_hopped":pl.Boolean,"cdna":pl.Utf8,"n_seq_errors":pl.Int64}
def route_and_amplify(molecules,wells,molecule_survival_rate,reads_per_molecule_mean,index_hop_rate,seed):
    """Route molecules to an amplification well, apply survival, then amplify into reads with index hopping.

    `amplification_well` is `resident_well` for resident molecules and a uniform random
    well for free (`is_free`) molecules — free molecules retain barcode+UMI but move
    wells. Each molecule then survives independently with probability
    `molecule_survival_rate`, applied *before* amplification (so non-survivors never
    contribute reads — this is not post-hoc read thinning). Each surviving molecule
    expands into a read family of size `max(1, Poisson(reads_per_molecule_mean))`, all
    sharing that molecule's UMI/cDNA. Each read then independently index-hops with
    probability `index_hop_rate` to `final_well = (amplification_well + random_offset) % wells`
    (barcode/UMI unchanged); non-hopped reads have `final_well == amplification_well`.

    Empty-safe: if `molecules` is empty or no molecule survives, returns a typed empty
    `reads` frame matching `_READS_SCHEMA`.

    Returns `(molecules, reads)`: `molecules` is the input frame with `amplification_well`
    and `survived` filled in; `reads` is the per-read record used by `reads.py`/`truth.py`.
    """
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
