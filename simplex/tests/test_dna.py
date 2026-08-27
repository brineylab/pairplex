import numpy as np, polars as pl
from simplex._dna import random_dna, revcomp_str, revcomp_expr, mutate_strings
def test_random_dna(): o=random_dna(np.random.default_rng(0),5,10); assert len(o)==5 and set("".join(o))<=set("ACGT")
def test_revcomp():
    assert revcomp_str("AAACCTGGN")=="NCCAGGTTT"
    assert pl.DataFrame({"s":["ACGT"]}).select(revcomp_expr("s"))["s"][0]=="ACGT"
def test_mutate():
    out,ne=mutate_strings(["ACGT"*50]*400,0.05,0.0,np.random.default_rng(0)); assert 4<ne.mean()<16
    o2,n2=mutate_strings(["ACGT"],0.0,0.0,np.random.default_rng(0)); assert o2==["ACGT"] and n2.sum()==0
    assert mutate_strings([],0.05,0.0,np.random.default_rng(0))[0]==[]        # empty ok
