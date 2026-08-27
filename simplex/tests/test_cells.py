import math, polars as pl
from simplex.cells import load_pairs, assign_droplets_and_barcodes, assign_wells
def _c(tmp,n=600):
    d={"sequence_id:0":[f"h{i}" for i in range(n)],"sequence:0":["A"*300]*n,"sequence_id:1":[f"l{i}" for i in range(n)],
       "sequence:1":["T"*300]*n,"name":[f"c{i}" for i in range(n)],"locus:0":["IGH"]*n,"locus:1":["IGK"]*n}
    p=tmp/"p.parquet"; pl.DataFrame(d).write_parquet(p); return load_pairs(p)
def test_unique_bc_per_droplet(tmp_path):
    c=assign_droplets_and_barcodes(_c(tmp_path),5,1,"v2",None,0)
    assert c.group_by("droplet_id").agg(pl.col("barcode").n_unique().alias("nb"))["nb"].max()==1
    assert c["barcode"].n_unique()==c["droplet_id"].n_unique() < c.height
def test_pool_reuse_collides(tmp_path):
    c=assign_droplets_and_barcodes(_c(tmp_path),5,1,"v2",20,0)
    assert c["barcode"].n_unique()<=20 < c["droplet_id"].n_unique()
def test_wells_uniform(tmp_path):
    c=assign_wells(_c(tmp_path,4000),8,0); k=c.group_by("resident_well").len()["len"].to_list()
    assert min(k)>4000/8*0.7 and max(k)<4000/8*1.3
def test_analytic_cooccupancy(tmp_path):
    wells=8; c=assign_wells(assign_droplets_and_barcodes(_c(tmp_path,2000),5,1,"v2",None,0),wells,0)
    exp=sum(math.comb(k,2) for k in c.group_by("droplet_id").len()["len"].to_list())/wells
    obs=c.group_by(["resident_well","barcode"]).len().filter(pl.col("len")>=2).select((pl.col("len")*(pl.col("len")-1)//2).sum()).item() or 0
    assert 0.6*exp<=obs<=1.6*exp
