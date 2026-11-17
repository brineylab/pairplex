import numpy as np, polars as pl
from simplex.routing import route_and_amplify
def _mol(n=2000):
    rng=np.random.default_rng(0)
    return pl.DataFrame({"molecule_id":list(range(n)),"parent_molecule_id":list(range(n)),
        "origin_cell_id":rng.integers(0,500,n),"origin_droplet_id":rng.integers(0,300,n),
        "source_pair_id":[f"c{i%500}" for i in range(n)],"chain":rng.integers(0,2,n).astype(np.int8),
        "locus":["IGH"]*n,"umi":["AAAAAAAAAA"]*n,"barcode":["BC"]*n,
        "resident_well":rng.integers(0,4,n).astype(np.int64),"amplification_well":[0]*n,
        "survived":[False]*n,"is_free":rng.random(n)<0.2,"cdna":["ACGT"*50]*n})
def test_all_kept_with_survival():
    m,r=route_and_amplify(_mol(),4,0.5,3,0.0,0)
    assert m.height==2000 and "survived" in m.columns and r["molecule_id"].n_unique()==m.filter(pl.col("survived")).height
def test_free_keeps_bc_umi():
    _,r=route_and_amplify(_mol(),4,1.0,3,0.0,0); assert (r["barcode"]=="BC").all() and (r["umi"]=="AAAAAAAAAA").all()
def test_family_shares_umi_and_hop():
    _,r=route_and_amplify(_mol(),4,1.0,4,0.2,0)
    assert r.group_by("molecule_id").agg(pl.col("umi").n_unique().alias("u"))["u"].max()==1
    h=r.filter(pl.col("is_index_hopped")); assert (h["final_well"]!=h["amplification_well"]).all()
def test_zero_survival_empty():
    _,r=route_and_amplify(_mol(),4,0.0,3,0.0,0); assert r.height==0 and "read_id" in r.columns
def test_empty_molecules():
    m0=_mol(0) if False else _mol().head(0)
    _,r=route_and_amplify(m0,4,1.0,3,0.0,0); assert r.height==0
