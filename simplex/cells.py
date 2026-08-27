"""Stage 1 of the pipeline: load source heavy/light pairs as cells, then assign them to
overloaded 10X droplets (shared barcodes) and to plate wells.

Sits at the front of the cells -> molecules -> routing -> reads -> truth -> scoring
pipeline: everything downstream (`molecules.py`, `routing.py`) consumes the `cells`
frame produced here.
"""
import numpy as np, polars as pl
from ._rng import rng_for
from .barcodes import load_barcodes
def load_pairs(input_data, n_cells=None, seed=0):
    """Load heavy/light source pairs from `input_data` (a paired parquet) into a `cells` frame.

    Validates that `locus:0`/`locus:1` are present (Phase 1-2 requires known loci) and
    that any repeated `source_pair_id` maps to a consistent sequence+locus (never treat
    two unrelated records with the same name as equivalent). If `n_cells` is given,
    subsamples (with replacement if `n_cells` exceeds the input size) using the
    `"subsample"` RNG stream keyed on `seed`. Returns one row per cell with a fresh
    `cell_id`, the two chain sequences/loci, and `source_pair_id` provenance.
    """
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
def assign_droplets_and_barcodes(cells, mean, overdispersion, chemistry, barcode_pool_size, seed):
    """Group cells into overloaded droplets (GEMs) and give each droplet a 10X barcode.

    Occupancy model (physically grounded): `mean` is the loading rate lambda = cells per
    GEM. Cells are randomly loaded into ``K = round(n_cells / mean)`` droplets, so droplet
    occupancy follows a **Poisson(lambda)** distribution — the random-encapsulation process,
    with a single interpretable knob and no clamp/round distortion. ``overdispersion`` (>= 0,
    default 0 = pure Poisson) makes per-droplet capture propensities vary via Dirichlet
    weights with concentration ``1/overdispersion`` (smaller alpha => more clumping),
    producing Negative-Binomial-like over-dispersed occupancy to model cell clumping /
    uneven GEMs. All cells in the same droplet share one `barcode` — the overloading
    mechanism that lets unrelated cells collide on a barcode.

    Barcodes come from the `chemistry` whitelist (`"barcodes"` RNG stream): one unique
    barcode per droplet if `barcode_pool_size` is `None`, otherwise sampled with replacement
    from a pool of that size (controlled cross-droplet reuse). Empty droplets (a natural
    Poisson consequence) simply own no cells. Adds `droplet_id` and `barcode` to `cells`.
    Uses the `"droplets"` RNG stream keyed on `seed`.
    """
    rng=rng_for(seed,"droplets"); n=cells.height
    K=max(1,int(round(n/mean)))
    if overdispersion and overdispersion>0:
        # Dirichlet capture propensities over droplets -> over-dispersed (NB-like) occupancy
        p=rng.dirichlet(np.full(K,1.0/overdispersion))
        droplet=rng.choice(K,size=n,p=p).astype(np.int64)
    else:
        # uniform random loading -> Poisson(lambda) occupancy
        droplet=rng.integers(0,K,size=n).astype(np.int64)
    brng=rng_for(seed,"barcodes")
    if barcode_pool_size:
        pool=np.array(load_barcodes(chemistry,min(barcode_pool_size,K),brng)); bc=pool[brng.integers(0,len(pool),size=K)]
    else:
        bc=np.array(load_barcodes(chemistry,K,brng))
    return cells.with_columns([pl.Series("droplet_id",droplet),pl.Series("barcode",bc[droplet])])
def assign_wells(cells,wells,seed):
    """Assign each whole cell a `resident_well` uniformly at random over `[0, wells)`.

    This models fixed whole cells being distributed into the plate before amplification;
    resident molecules of a cell later inherit this well as their `amplification_well`.
    Uses the `"wells"` RNG stream keyed on `seed`.
    """
    return cells.with_columns(pl.Series("resident_well",rng_for(seed,"wells").integers(0,wells,size=cells.height).astype(np.int64)))
