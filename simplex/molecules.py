import numpy as np, polars as pl
from ._dna import random_dna, mutate_strings
from ._rng import rng_for
_MOL_SCHEMA={"molecule_id":pl.Int64,"parent_molecule_id":pl.Int64,"origin_cell_id":pl.Int64,
    "origin_droplet_id":pl.Int64,"source_pair_id":pl.Utf8,"chain":pl.Int8,"locus":pl.Utf8,"umi":pl.Utf8,
    "barcode":pl.Utf8,"resident_well":pl.Int64,"amplification_well":pl.Int64,"is_free":pl.Boolean,
    "survived":pl.Boolean,"cdna":pl.Utf8}
def generate_molecules(cells,recovery_rate,molecules_per_chain_mean,release_rate,umi_length,rt_sub_rate,rt_indel_rate,seed):
    rng=rng_for(seed,"molecules"); n=cells.height; frames=[]; status=[]
    for chain in (0,1):
        captured=rng.random(n)<recovery_rate
        nmol=np.where(captured,np.maximum(rng.poisson(molecules_per_chain_mean,n),1),0).astype(np.int64)
        status.append(pl.DataFrame({"cell_id":cells["cell_id"],"chain":np.full(n,chain,np.int8),"captured":captured,"n_molecules":nmol}))
        rep=np.repeat(np.arange(n),nmol)
        if rep.size==0: continue
        sub=cells[rep]; k=rep.size; cdna=list(sub[f"chain{chain}_seq"])
        if rt_sub_rate>0 or rt_indel_rate>0:
            cdna,_=mutate_strings(cdna,rt_sub_rate,rt_indel_rate,rng_for(seed,f"rt{chain}"))
        bc=sub["barcode"].to_numpy().astype(str)
        frames.append(pl.DataFrame({"origin_cell_id":sub["cell_id"],"origin_droplet_id":sub["droplet_id"],
            "source_pair_id":sub["source_pair_id"],"chain":np.full(k,chain,np.int8),"locus":sub[f"chain{chain}_locus"],
            "umi":random_dna(rng,k,umi_length),"barcode":bc,"resident_well":sub["resident_well"],
            "is_free":rng.random(k)<release_rate,"cdna":cdna}))
    cs=pl.concat(status)
    if not frames:
        empty=pl.DataFrame(schema=_MOL_SCHEMA); return empty, cs
    m=pl.concat(frames).with_row_index("molecule_id").with_columns([
        pl.col("molecule_id").cast(pl.Int64),pl.col("molecule_id").cast(pl.Int64).alias("parent_molecule_id"),
        pl.lit(0).cast(pl.Int64).alias("amplification_well"), pl.lit(False).alias("survived")])
    return m.select(list(_MOL_SCHEMA.keys())), cs
