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
    "final_well":pl.Int64,"is_free":pl.Boolean,"is_index_hopped":pl.Boolean,"cdna":pl.Utf8,"n_seq_errors":pl.Int64,"is_ambient":pl.Boolean}
def route_and_amplify(molecules,wells,molecule_survival_rate,reads_per_molecule_mean,index_hop_rate,seed,umi_sub_rate=0.0,
                      reads_per_molecule_dispersion=0.0,reads_per_molecule_mean_light=None):
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

    `umi_sub_rate` (default 0): per-base substitution rate applied to each READ's UMI after
    amplification (sequencing/PCR error in the UMI). Error UMIs are distinct strings for the same
    molecule, so they inflate UMI counts exactly as uncorrected real data does (real reads: ~21% of
    UMIs are 1-read Hamming-1 neighbours of a deep UMI). `molecule_id` keeps the truth.

    `reads_per_molecule_dispersion` (default 0 = Poisson, unchanged draws): read-family size becomes
    negative binomial, `Poisson(Gamma(shape=1/d, scale=mean*d))`, variance `mean + d*mean^2` (PCR jackpots:
    real families are heavy-tailed). Drawn from the separate `"depth"` RNG stream, so the routing stream
    (wells, survival, hops) is identical to the Poisson model for the same seed.

    `reads_per_molecule_mean_light` (default None = same as heavy): read-family mean for IGK/IGL molecules
    (real heavy molecules are sequenced ~5x deeper than light; their jackpots spawn the error UMIs).

    Returns `(molecules, reads)`: `molecules` is the input frame with `amplification_well`
    and `survived` filled in; `reads` is the per-read record used by `reads.py`/`truth.py`.
    """
    if "is_ambient" not in molecules.columns:
        molecules=molecules.with_columns(pl.lit(False).alias("is_ambient"))
    rng=rng_for(seed,"routing"); n=molecules.height
    free=molecules["is_free"].to_numpy() if n else np.array([],bool)
    amp=(np.where(free,rng.integers(0,wells,size=n),molecules["resident_well"].to_numpy()).astype(np.int64) if n else np.array([],np.int64))
    surv=rng.random(n)<molecule_survival_rate if n else np.array([],bool)
    mols=molecules.with_columns([pl.Series("amplification_well",amp),pl.Series("survived",surv)]) if n else molecules
    survd=mols.filter(pl.col("survived"))
    if survd.height==0:
        return mols, pl.DataFrame(schema=_READS_SCHEMA)
    mean=reads_per_molecule_mean
    if reads_per_molecule_mean_light is not None and survd.height:
        mean=np.where(survd["locus"].is_in(["IGK","IGL","junk"]).to_numpy(),reads_per_molecule_mean_light,reads_per_molecule_mean)
    if reads_per_molecule_dispersion and reads_per_molecule_dispersion>0:
        drng=rng_for(seed,"depth"); d=reads_per_molecule_dispersion
        lam=drng.gamma(1.0/d,np.asarray(mean,dtype=float)*d,size=survd.height)
        depth=np.maximum(drng.poisson(lam),1).astype(np.int64)
        rng.poisson(reads_per_molecule_mean,survd.height)  # keep the routing stream aligned with the Poisson model
    else:
        depth=np.maximum(rng.poisson(mean,survd.height),1).astype(np.int64)
    rep=np.repeat(np.arange(survd.height),depth); reads=survd[rep]; k=reads.height
    hop=rng.random(k)<index_hop_rate; off=rng.integers(1,max(2,wells),size=k); a=reads["amplification_well"].to_numpy()
    final=np.where(hop,(a+off)%wells,a).astype(np.int64)
    if umi_sub_rate>0 and k:
        urng=rng_for(seed,"umierr")
        U=np.array([list(u) for u in reads["umi"].to_list()])
        hit=urng.random(U.shape)<umi_sub_rate
        if hit.any():
            alt=np.array(list("ACGT"))
            sub=alt[urng.integers(0,4,size=int(hit.sum()))]
            same=sub==U[hit]
            sub[same]=np.array([{"A":"C","C":"G","G":"T","T":"A"}.get(c,"A") for c in sub[same]])
            U[hit]=sub
            reads=reads.with_columns(pl.Series("umi",["".join(r) for r in U]))
    reads=reads.with_columns([pl.Series("read_id",[f"r{i}" for i in range(k)]),pl.Series("final_well",final),
        pl.Series("is_index_hopped",hop),pl.lit(0,pl.Int64).alias("n_seq_errors")]).select(list(_READS_SCHEMA.keys()))
    return mols, reads
