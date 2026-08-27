"""Stage 2 of the pipeline: RT capture of per-chain molecules from each cell, and the
resident-vs-free (ambient) split.

Consumes the `cells` frame from `cells.py` and produces the atomic `molecules` record
(one row per captured RT molecule, one UMI each) plus a `chain_status` summary, feeding
`routing.py` next in the cells -> molecules -> routing -> reads -> truth -> scoring pipeline.
"""
import numpy as np, polars as pl
from ._dna import random_dna, mutate_strings
from ._rng import rng_for
_MOL_SCHEMA={"molecule_id":pl.Int64,"parent_molecule_id":pl.Int64,"origin_cell_id":pl.Int64,
    "origin_droplet_id":pl.Int64,"source_pair_id":pl.Utf8,"chain":pl.Int8,"locus":pl.Utf8,"umi":pl.Utf8,
    "barcode":pl.Utf8,"resident_well":pl.Int64,"amplification_well":pl.Int64,"is_free":pl.Boolean,
    "survived":pl.Boolean,"cdna":pl.Utf8}
def generate_molecules(cells,recovery_rate,molecules_per_chain_mean,release_rate,umi_length,rt_sub_rate,rt_indel_rate,seed):
    """Simulate RT capture per cell/chain and split molecules into resident vs free (ambient).

    For each chain (0, 1) independently: each cell captures the chain with probability
    `recovery_rate` (`"molecules"` RNG stream); captured cells get
    `max(1, Poisson(molecules_per_chain_mean))` molecules. Each molecule gets its own UMI
    (`umi_length` long, so molecule-level `n_umis` is implicitly 1 and UMI collisions only
    surface later as `n_source_molecules > n_umis` on aggregation), inherits the cell's
    `barcode`/`resident_well`, and independently becomes `is_free` with probability
    `release_rate` (the ambient pool — free molecules keep barcode+UMI but are later
    redistributed to a different well by `routing.route_and_amplify`). If `rt_sub_rate`/
    `rt_indel_rate` are nonzero, cDNA is mutated once per molecule (RT error), so the
    error is inherited by every read in that molecule's family downstream.

    Empty-safe: if no molecules are generated at all, returns a typed empty `molecules`
    frame matching `_MOL_SCHEMA`.

    Returns `(molecules, chain_status)`: `molecules` is the per-molecule truth record
    (`molecule_id`, `parent_molecule_id` initialized equal to it, `amplification_well`
    initialized to 0 and `survived` to False — both set later by `routing.py`); `chain_status`
    is one row per `(cell_id, chain)` with `captured`/`n_molecules`, used by `truth.build_truth_cells`.
    """
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
