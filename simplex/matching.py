import edlib
from typing import NamedTuple

class OrientationResult(NamedTuple):
    pairing_status: str
    source_resolution: str
    resolved_source: object
    valid_assignments: set   # set[(heavy_source, light_source)] consistent with the observation

def seq_match(a,b,max_frac=0.06,min_len=50):
    if not a or not b: return False
    short,long=(a,b) if len(a)<=len(b) else (b,a)
    if len(short)<min_len: return False
    r=edlib.align(short,long,mode="HW",task="distance")
    return 0<=r["editDistance"]<=max_frac*len(short)
def candidates(seq,locus,key_entry,max_frac=0.06,min_len=50):
    if not seq or key_entry is None: return set()
    hits=set()
    for full,sources in key_entry.get(locus,{}).items():
        if seq==full or seq_match(seq,full,max_frac,min_len): hits|=sources
    return hits
def resolve(h,l):
    if not h or not l:
        return OrientationResult("unmatchable","none",None,set())
    inter=h&l
    if not inter:                                   # non-empty sets, empty intersection ⇒ mispaired
        return OrientationResult("mispaired","none",None,{(hh,ll) for hh in h for ll in l})
    va={(s,s) for s in inter}                        # valid CORRECT explanations are same-source only
    if len(inter)==1:
        return OrientationResult("correct","unique",next(iter(inter)),va)
    return OrientationResult("correct","ambiguous",None,va)
