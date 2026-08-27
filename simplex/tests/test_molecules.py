import polars as pl
from simplex.molecules import generate_molecules
def _cells(n=1000):
    return pl.DataFrame({"cell_id":list(range(n)),"source_pair_id":[f"c{i}" for i in range(n)],
        "chain0_id":[f"h{i}" for i in range(n)],"chain0_seq":["ACGT"*80]*n,"chain0_locus":["IGH"]*n,
        "chain1_id":[f"l{i}" for i in range(n)],"chain1_seq":["TTGG"*80]*n,"chain1_locus":["IGK"]*n,
        "droplet_id":list(range(n)),"barcode":["ACGTACGTACGTACGT"]*n,"resident_well":[0]*n})
def test_chain_status_all(tmp_path):
    m,cs=generate_molecules(_cells(500),0.5,5,0.0,10,0.0,0.0,0)
    assert cs.height==1000 and 0.4<cs["captured"].mean()<0.6 and (cs.filter(~pl.col("captured"))["n_molecules"]==0).all()
def test_zero_recovery_empty(tmp_path):
    m,cs=generate_molecules(_cells(50),0.0,5,0.0,10,0.0,0.0,0)
    assert m.height==0 and "molecule_id" in m.columns and cs.height==100
def test_release_and_rt(tmp_path):
    m,_=generate_molecules(_cells(2000),1.0,6,0.2,10,0.2,0.0,2)
    assert 0.15<m["is_free"].mean()<0.25 and (m.filter(pl.col("chain")==0)["cdna"]!="ACGT"*80).sum()>0
