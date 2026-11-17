import polars as pl, pytest
from simplex.cells import load_pairs
from simplex.barcodes import load_barcodes
from simplex._rng import rng_for
def _inp(tmp,n=8,locus=True,dupname=False):
    ids=[f"c{i}" for i in range(n)]
    if dupname: ids[1]=ids[0]                      # same name, different seqs -> must fail
    d={"sequence_id:0":[f"h{i}" for i in range(n)],"sequence:0":[f"ACGT{i}"*20 for i in range(n)],
       "sequence_id:1":[f"l{i}" for i in range(n)],"sequence:1":[f"TTGG{i}"*20 for i in range(n)],"name":ids}
    if locus: d["locus:0"]=["IGH"]*n; d["locus:1"]=["IGK"]*n
    p=tmp/"p.parquet"; pl.DataFrame(d).write_parquet(p); return p
def test_load(tmp_path):
    c=load_pairs(_inp(tmp_path)); assert c["chain0_locus"][0]=="IGH" and c["source_pair_id"][0]=="c0"
def test_locus_required(tmp_path):
    with pytest.raises(ValueError): load_pairs(_inp(tmp_path,locus=False))
def test_dup_name_inconsistent(tmp_path):
    with pytest.raises(ValueError): load_pairs(_inp(tmp_path,dupname=True))
def test_barcodes(tmp_path):
    b=load_barcodes("v2",300,rng_for(0,"bc")); assert len(set(b))==300 and all(len(x)==16 for x in b)
