import os, abstar
from abutils.io import parse_fastx
import polars as pl
from .matching import seq_match
_td=os.path.dirname(abstar.__file__)+"/test_data"
_h={s.id:s.sequence for s in parse_fastx(_td+"/test_hiv_bnab_hcs.fasta")}
_l={s.id:s.sequence for s in parse_fastx(_td+"/test_hiv_bnab_lcs.fasta")}
_names=[n for n in _h if n in _l]
def _pick_distinct():
    for i in range(len(_names)):
        for j in range(i+1,len(_names)):
            a,b=_names[i],_names[j]
            if not seq_match(_h[a],_h[b]) and not seq_match(_l[a],_l[b]): return a,b
    raise RuntimeError("no two distinguishable antibody pairs under the scorer threshold")
_A,_B=_pick_distinct()
HEAVY_A,LIGHT_A=_h[_A],_l[_A]; HEAVY_B,LIGHT_B=_h[_B],_l[_B]
assert not seq_match(HEAVY_A,HEAVY_B) and not seq_match(LIGHT_A,LIGHT_B)   # guaranteed distinct
def many_pairs_parquet(tmp, n=60):
    if n>len(_names): raise ValueError(f"requested {n} pairs but only {len(_names)} available")
    names=_names[:n]
    df=pl.DataFrame({"sequence_id:0":names,"sequence:0":[_h[x] for x in names],"locus:0":["IGH"]*len(names),
        "sequence_id:1":names,"sequence:1":[_l[x] for x in names],"locus:1":["IGK"]*len(names),"name":names})
    p=tmp/"pairs.parquet"; df.write_parquet(p); return p
