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
    "survived":pl.Boolean,"cdna":pl.Utf8,"is_ambient":pl.Boolean}
def generate_molecules(cells,recovery_rate,molecules_per_chain_mean,release_rate,umi_length,rt_sub_rate,rt_indel_rate,seed,
                       release_rate_heavy=None,release_rate_light=None,molecules_per_chain_mean_light=None,
                       ambient_rate=0.0):
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

    Chain-specific knobs (all default to the shared values): `release_rate_heavy` /
    `release_rate_light` override `release_rate` for IGH / IGK+IGL molecules (measured leakage
    differs by chain), and `molecules_per_chain_mean_light` overrides the molecule count for light
    chains (light-chain template is more abundant than heavy).

    Ambient axis (`ambient_rate`, default 0): free mRNA from lysed cells captured in OTHER droplets
    before RT. `Poisson(ambient_rate * n_cell_molecules)` ambient molecules are added; each copies the
    sequence of a uniformly chosen cell molecule (so chain mix follows template abundance), gets the
    barcode of a uniformly chosen droplet among those holding cells, a fresh UMI, its own RT error,
    and is free (`is_free=True`, routed to a random well) with `is_ambient=True`. Unlike release,
    ambient ADDS molecules and moves a chain to a different barcode.

    Returns `(molecules, chain_status)`: `molecules` is the per-molecule truth record
    (`molecule_id`, `parent_molecule_id` initialized equal to it, `amplification_well`
    initialized to 0 and `survived` to False — both set later by `routing.py`); `chain_status`
    is one row per `(cell_id, chain)` with `captured`/`n_molecules`, used by `truth.build_truth_cells`.
    """
    rng=rng_for(seed,"molecules"); n=cells.height; frames=[]; status=[]
    def _is_light(chain):
        return cells[f"chain{chain}_locus"].is_in(["IGK","IGL"]).to_numpy() if n else np.array([],bool)
    for chain in (0,1):
        light=_is_light(chain)
        mean=np.where(light, molecules_per_chain_mean if molecules_per_chain_mean_light is None else molecules_per_chain_mean_light,
                      molecules_per_chain_mean)
        captured=rng.random(n)<recovery_rate
        nmol=np.where(captured,np.maximum(rng.poisson(mean),1),0).astype(np.int64)
        status.append(pl.DataFrame({"cell_id":cells["cell_id"],"chain":np.full(n,chain,np.int8),"captured":captured,"n_molecules":nmol}))
        rep=np.repeat(np.arange(n),nmol)
        if rep.size==0: continue
        sub=cells[rep]; k=rep.size; cdna=list(sub[f"chain{chain}_seq"])
        if rt_sub_rate>0 or rt_indel_rate>0:
            cdna,_=mutate_strings(cdna,rt_sub_rate,rt_indel_rate,rng_for(seed,f"rt{chain}"))
        bc=sub["barcode"].to_numpy().astype(str)
        lk=light[rep]
        rh=release_rate if release_rate_heavy is None else release_rate_heavy
        rl=release_rate if release_rate_light is None else release_rate_light
        rel_k=np.where(lk,rl,rh)
        frames.append(pl.DataFrame({"origin_cell_id":sub["cell_id"],"origin_droplet_id":sub["droplet_id"],
            "source_pair_id":sub["source_pair_id"],"chain":np.full(k,chain,np.int8),"locus":sub[f"chain{chain}_locus"],
            "umi":random_dna(rng,k,umi_length),"barcode":bc,"resident_well":sub["resident_well"],
            "is_free":rng.random(k)<rel_k,"cdna":cdna}))
    cs=pl.concat(status)
    if not frames:
        empty=pl.DataFrame(schema=_MOL_SCHEMA); return empty, cs
    m=pl.concat(frames).with_columns(pl.lit(False).alias("is_ambient"))
    if ambient_rate>0 and m.height:
        arng=rng_for(seed,"ambient")
        na=int(arng.poisson(ambient_rate*m.height))
        if na:
            src=m[arng.integers(0,m.height,size=na)]
            droplet_bcs=cells.select("droplet_id","barcode").unique().sort("droplet_id")
            pick=droplet_bcs[arng.integers(0,droplet_bcs.height,size=na)]
            # ambient mRNA is a separate molecule: clean cell sequence + its own RT error
            seq_lookup={(int(c),ch):s for c,ch,s in pl.concat([cells.select("cell_id",pl.lit(ch).alias("ch"),pl.col(f"chain{ch}_seq").alias("s")) for ch in (0,1)]).iter_rows()}
            cdna=[seq_lookup[(int(c),int(ch))] for c,ch in zip(src["origin_cell_id"],src["chain"])]
            if rt_sub_rate>0 or rt_indel_rate>0:
                cdna,_=mutate_strings(cdna,rt_sub_rate,rt_indel_rate,rng_for(seed,"rt_ambient"))
            amb=src.with_columns([pl.Series("barcode",pick["barcode"].to_numpy().astype(str)),
                pl.Series("umi",random_dna(arng,na,umi_length)),pl.Series("cdna",cdna),
                pl.lit(True).alias("is_free"),pl.lit(True).alias("is_ambient")])
            m=pl.concat([m,amb.select(m.columns)])
    m=m.with_row_index("molecule_id").with_columns([
        pl.col("molecule_id").cast(pl.Int64),pl.col("molecule_id").cast(pl.Int64).alias("parent_molecule_id"),
        pl.lit(0).cast(pl.Int64).alias("amplification_well"), pl.lit(False).alias("survived")])
    return m.select(list(_MOL_SCHEMA.keys())), cs
