# SimPlex Phase 0–2 Implementation Plan (v4 — implementation-ready)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use `- [ ]` checkboxes. This plan is **fully self-contained** — every task inlines its own code AND tests; do NOT consult prior plan versions.
>
> **v4 fixes (final plan review):** Tasks 5/7/10 fully inlined (code+tests, no v2 references); scorer resolution returns `valid_assignments` and classifies origin from them, orientations compared by assignment set (not label), `sequence_id:1` column fixed; zero-read truth path uses a complete `_COMP_AGG_SCHEMA`; observability/reference-pairability use total home-key support (`n_reads`/`n_umis`); Task 12 recall = singleton reference-pairable keys with a unique resident-correct output; `_testseqs` picks provably-distinct pairs and `many_pairs_parquet(n)` is exact-or-raise; TSO fixed+validated; dominance aggregates by `source_pair_id`; `truth_reads` single parquet.

**Goal:** Mechanistically-faithful SimPlex generator + compact truth + `(well,barcode)`-keyed scorer, so PairPlex can be driven on synthetic data with known truth to measure precision/yield.

**Architecture:** polars+numpy staged pipeline. **Phase 0B first** (matcher + scorer contract on hand-crafted truth) to freeze the truth schema. Then the generator: all molecules retained with a `survived` flag; free molecules redistribute across wells pre-amplification keeping barcode+UMI; read families inherit RT error; index hopping is per-read. Spec: `docs/superpowers/specs/2026-08-27-simplex-generator-design-v5.md` (frozen).

**Tech Stack:** Python 3.10+, polars 1.39, numpy 2.x, edlib, pytest.

## Global Constraints

- Sibling package `simplex/`. Merged layout `barcode(16)+umi(10)+TSO+cDNA`; **fixed** for Phase 1–2 (`barcode_length==16`, `umi_length==10` enforced). `output_mode="merged"` only.
- **Reproducibility (v1):** same seed + same input order + same layout → identical content. Per-stage RNG `rng_for(seed, stage)`. No chunk/order-invariance claim.
- **Shared contract constants** (`simplex/_contract.py`): `REF_MIN_READS=3`, `REF_MIN_UMIS=1`, `BARCODE_LEN=16`, `UMI_LEN=10`. The scorer and truth builder both import these — never hardcode twice.
- Scorer: bounded edit-distance match (edlib), **orientation-agnostic** (never trust PairPlex's locus for which output chain is heavy), set-valued, key-local, joint over **all** wells; `pairing_status` and `source_resolution` are separate axes; empty intersection of non-empty sets ⇒ `mispaired`; both orientations viable & incompatible ⇒ `ambiguous`.
- Truth preserves `captured`/`survived`/`n_molecules`; occupancy from `cells`; observability computed **per resident cell at its home key** before aggregation.
- Every stage returns a typed **empty frame** for valid zero cases (0 captured / 0 survived / 0 reads / 0 outputs). Commit after each task.

## Canonical schemas

```
cells:     cell_id:i64, source_pair_id:str, chain0_id/seq/locus:str, chain1_id/seq/locus:str
  +droplet: droplet_id:i64, barcode:str    +well: resident_well:i64
chain_status: cell_id:i64, chain:i8, captured:bool, n_molecules:i64        # ALL cell×chain
molecules: molecule_id:i64, parent_molecule_id:i64, origin_cell_id:i64, origin_droplet_id:i64,
           source_pair_id:str, chain:i8, locus:str, umi:str, barcode:str, resident_well:i64,
           amplification_well:i64, is_free:bool, survived:bool, cdna:str      # ALL molecules kept
reads:     read_id:str, molecule_id:i64, origin_cell_id:i64, source_pair_id:str, chain:i8, locus:str,
           umi:str, barcode:str, amplification_well:i64, final_well:i64, is_free:bool,
           is_index_hopped:bool, cdna:str, n_seq_errors:i64            # survivors only, per-read hop
built:     read_id:str, final_well:i64, read_seq:str, qual:str
truth_components: (final_well, barcode, origin_cell_id, chain) + source_pair_id, locus, sequence,
           is_resident_source, n_source_molecules, n_umis, n_reads, n_reads_resident,
           n_reads_free, n_reads_index_hopped
truth_cells: cells + per-chain captured, survived, n_molecules, n_umis, n_reads_generated,
           n_reads_resident, n_reads_free_out, n_reads_index_hopped_out
truth_barcodes: (well, barcode) + resident_source_ids, n_resident_cells, is_collision, is_ambient_only,
           n_captured_both/n_survived_both/n_sequenced_both/n_reference_pairable_resident_cells,
           dominant_{heavy,light}_source_by_{reads,umis}, {heavy,light}_dominance_is_tied
pair_scores: pair_id, source_file, well, barcode, sequence_id:0, sequence_id:1, pairing_status,
           source_resolution, origin_status, key_status, output_status, resolved_source
key_scores: well, barcode, key_status, output_count, output_status, n_resident_cells,
           captured_both, survived_both, sequenced_both, reference_pairable_both, no_output_reason
```

---

### Task 1: Scaffold — contract constants, keyed RNG, DNA helpers, config

**Files:** Create `simplex/__init__.py`, `simplex/_contract.py`, `simplex/_rng.py`, `simplex/_dna.py`, `simplex/config.py`, `simplex/tests/__init__.py`, `simplex/tests/test_rng.py`, `simplex/tests/test_dna.py`, `simplex/tests/test_config.py`. Modify `pyproject.toml`.

- [ ] **Step 1: failing tests**

`simplex/tests/test_rng.py`:
```python
from simplex._rng import rng_for
def test_same_stage(): assert list(rng_for(0,"m").integers(0,10**6,50))==list(rng_for(0,"m").integers(0,10**6,50))
def test_diff_stage(): assert list(rng_for(0,"m").integers(0,10**6,50))!=list(rng_for(0,"n").integers(0,10**6,50))
```
`simplex/tests/test_dna.py`:
```python
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
```
`simplex/tests/test_config.py`:
```python
import pytest
from simplex.config import SimplexConfig
def C(**k): return SimplexConfig(input_data="x", output_directory="o", **k)
def test_defaults(tmp_path): c=C(); assert c.output_mode=="merged"; c.to_json(tmp_path/"c.json")
def test_reject_paired():
    with pytest.raises(ValueError): C(output_mode="paired").validate()
def test_reject_bad_rate():
    with pytest.raises(ValueError): C(release_rate=1.5).validate()
def test_index_hop_one_well():
    with pytest.raises(ValueError): C(wells=1, index_hop_rate=0.01).validate()
def test_reject_fixed_structure_change():
    with pytest.raises(ValueError): C(barcode_length=12).validate()
    with pytest.raises(ValueError): C(umi_length=12).validate()
    with pytest.raises(ValueError): C(tso="GGGGGGGGGGGGG").validate()
def test_reject_nonpositive():
    for k in ["wells","cells_per_droplet_mean","molecules_per_chain_mean","reads_per_molecule_mean"]:
        with pytest.raises(ValueError): C(**{k:0}).validate()
def test_oom():
    with pytest.raises(ValueError): C(reads_per_molecule_mean=50, molecules_per_chain_mean=50).validate(actual_n_cells=10_000_000, max_reads=5_000_000_000)
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement**

`simplex/_contract.py`:
```python
REF_MIN_READS = 3     # frozen reference-pairable minimum (threshold-independent)
REF_MIN_UMIS = 1
BARCODE_LEN = 16      # fixed by pairplex.parse_barcodes (s[:16])
UMI_LEN = 10          # s[16:26]
TSO = "TTTCTTATATGGG" # fixed: parse_barcodes does s[36:].lstrip("G"); arbitrary TSO corrupts cDNA
```
`simplex/_rng.py`:
```python
import hashlib
import numpy as np
def rng_for(seed, stage):
    ent = int.from_bytes(hashlib.blake2b(f"{seed}|{stage}".encode(), digest_size=16).digest(), "big")
    return np.random.default_rng(np.random.SeedSequence(ent))
```
`simplex/_dna.py`:
```python
import numpy as np, polars as pl
_A=np.array([65,67,71,84],np.uint8); _C=bytes.maketrans(b"ACGTN",b"TGCAN"); _B=np.array(list("ACGT"))
def random_dna(rng,k,length):
    if k==0: return np.array([],object)
    return _A[rng.integers(0,4,size=(k,length),dtype=np.uint8)].view(f"S{length}").reshape(k).astype(str)
def revcomp_str(s): return s.translate(_C)[::-1]
def revcomp_expr(col): return pl.col(col).str.reverse().str.replace_many(["A","C","G","T"],["T","G","C","A"])
def mutate_strings(seqs, sub_rate, indel_rate, rng):
    out,cnt=[],np.zeros(len(seqs),np.int64)
    for i,s in enumerate(seqs):
        ch,n=list(s),0
        if sub_rate>0:
            for p in np.nonzero(rng.random(len(ch))<sub_rate)[0]:
                a=rng.choice(_B)
                while a==ch[p]: a=rng.choice(_B)
                ch[p]=str(a); n+=1
        if indel_rate>0:
            r=[]
            for c in ch:
                u=rng.random()
                if u<indel_rate/2: n+=1; continue
                r.append(c)
                if u>1-indel_rate/2: r.append(str(rng.choice(_B))); n+=1
            ch=r
        out.append("".join(ch)); cnt[i]=n
    return out,cnt
```
`simplex/config.py`:
```python
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from ._contract import BARCODE_LEN, UMI_LEN, TSO as _TSO

@dataclass
class SimplexConfig:
    input_data: str; output_directory: str
    n_cells: int | None = None; wells: int = 96
    cells_per_droplet_mean: float = 5.0; cells_per_droplet_sd: float = 2.0
    barcode_pool_size: int | None = None
    recovery_rate: float = 0.5; molecules_per_chain_mean: float = 10.0
    release_rate: float = 0.02; molecule_survival_rate: float = 0.8; reads_per_molecule_mean: float = 5.0
    rt_sub_rate: float = 0.0; rt_indel_rate: float = 0.0
    sequencing_sub_rate: float = 0.001; sequencing_indel_rate: float = 0.0
    index_hop_rate: float = 0.001
    barcode_length: int = BARCODE_LEN; umi_length: int = UMI_LEN; tso: str = "TTTCTTATATGGG"; chemistry: str = "v2"
    output_mode: str = "merged"; rc_fraction: float = 0.0
    variable_length: bool = True; write_read_truth: bool = False; seed: int = 0

    _RATES=("recovery_rate","release_rate","molecule_survival_rate","rt_sub_rate","rt_indel_rate",
            "sequencing_sub_rate","sequencing_indel_rate","index_hop_rate","rc_fraction")
    _POS=("wells","cells_per_droplet_mean","molecules_per_chain_mean","reads_per_molecule_mean")

    def to_dict(self): return asdict(self)
    def to_json(self,p): Path(p).write_text(json.dumps(self.to_dict(),indent=2))
    def estimated_reads(self,n):
        return int(n*2*self.recovery_rate*self.molecules_per_chain_mean*self.molecule_survival_rate*self.reads_per_molecule_mean)
    def validate(self, actual_n_cells=None, max_reads=3_000_000_000):
        for r in self._RATES:
            v=getattr(self,r)
            if not (0.0<=v<=1.0): raise ValueError(f"{r}={v} not in [0,1]")
        for r in self._POS:
            if getattr(self,r)<=0: raise ValueError(f"{r} must be > 0")
        if self.cells_per_droplet_sd<0: raise ValueError("cells_per_droplet_sd must be >= 0")
        if self.barcode_pool_size is not None and self.barcode_pool_size<=0: raise ValueError("barcode_pool_size must be > 0 or None")
        if self.n_cells is not None and self.n_cells<=0: raise ValueError("n_cells must be > 0 or None")
        if self.output_mode!="merged": raise ValueError("Phase 1-2: output_mode='merged' only")
        if self.barcode_length!=BARCODE_LEN or self.umi_length!=UMI_LEN:
            raise ValueError(f"Phase 1-2 fixes barcode_length={BARCODE_LEN}, umi_length={UMI_LEN}")
        if self.tso!=_TSO: raise ValueError(f"Phase 1-2 fixes tso={_TSO!r} (parser assumes it)")
        if self.wells==1 and self.index_hop_rate!=0: raise ValueError("index_hop_rate must be 0 when wells==1")
        n=actual_n_cells if actual_n_cells is not None else self.n_cells
        if n and self.estimated_reads(n)>max_reads: raise ValueError(f"est reads {self.estimated_reads(n)}>budget {max_reads}")
        return self
```
`simplex/__init__.py`:
```python
from .config import SimplexConfig
__all__=["SimplexConfig","run","score"]
def run(*a,**k):
    from .run import run as r; return r(*a,**k)
def score(*a,**k):
    from .scoring import score as s; return s(*a,**k)
```
`pyproject.toml`: include `simplex*`. `read_length`/`platform` are intentionally **omitted** from the Phase 1–2 API (merged-only; reserved for Phase 3).

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/ pyproject.toml && git commit -m "feat(simplex): scaffold — contract constants, keyed RNG, DNA, config validation"`

---

### Task 2 (0B): bounded matcher + joint resolver

**Files:** Create `simplex/matching.py`, `simplex/tests/test_matching.py`.

**Produces:** `seq_match(a,b,max_frac=0.06,min_len=50)`, `candidates(seq,locus,key_entry,...)`, `resolve(h_cands,l_cands) -> (pairing_status, source_resolution, resolved|None)`.

- [ ] **Step 1: failing tests**
```python
from simplex.matching import resolve, seq_match
def test_disjoint_singletons():
    r=resolve({"A"},{"B"}); assert r[:3]==("mispaired","none",None) and r.valid_assignments=={("A","B")}
def test_disjoint_one_ambiguous():
    r=resolve({"A","B"},{"C"}); assert r[:3]==("mispaired","none",None) and r.valid_assignments=={("A","C"),("B","C")}
def test_unique():
    r=resolve({"A","B"},{"A"}); assert r[:3]==("correct","unique","A") and r.valid_assignments=={("A","A")}
def test_nonunique():
    r=resolve({"A","B"},{"A","B"}); assert r[:3]==("correct","ambiguous",None) and r.valid_assignments=={("A","A"),("B","B")}
def test_empty():
    r=resolve(set(),{"A"}); assert r[:3]==("unmatchable","none",None) and r.valid_assignments==set()
def test_seq_match():
    a="ACGT"*30; b=a[:60]+"T"+a[61:]
    assert seq_match(a,b) and not seq_match(a,"TTTT"*30) and not seq_match("ACG","ACG")  # too short
```

- [ ] **Step 2: run → FAIL.**   - [ ] **Step 3: implement**
```python
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
```

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/matching.py simplex/tests/test_matching.py && git commit -m "feat(simplex): bounded matcher + joint resolver (0B)"`

---

### Task 3 (0B): scorer — orientation-agnostic, joint, key-level, origin enumeration

**Files:** Create `simplex/scoring.py`, `simplex/tests/test_scoring.py`.

**Produces:** `score(pairplex_output, truth_dir, *, pairplex_metadata=None) -> (pair_scores, key_scores)`. `pairplex_output` = dir (globs `**/*_paired.parquet`) | parquet | list; reads all jointly.

- [ ] **Step 1: failing tests**
```python
import polars as pl
from simplex.scoring import score
def _truth(tmp):
    td=tmp/"truth"; td.mkdir()
    pl.DataFrame({"final_well":[0,0],"barcode":["X","X"],"origin_cell_id":[0,0],"source_pair_id":["A","A"],
        "chain":[0,1],"locus":["IGH","IGK"],"sequence":["H_A"*20,"L_A"*20],"is_resident_source":[True,True],
        "n_source_molecules":[3,3],"n_umis":[3,3],"n_reads":[9,9],"n_reads_resident":[9,9],
        "n_reads_free":[0,0],"n_reads_index_hopped":[0,0]}).write_parquet(td/"truth_components.parquet")
    pl.DataFrame({"well":[0],"barcode":["X"],"n_resident_cells":[1],"is_collision":[False],"is_ambient_only":[False],
        "n_sequenced_both_resident_cells":[1],"n_reference_pairable_resident_cells":[1]}).write_parquet(td/"truth_barcodes.parquet")
    return td
def _pp(tmp, s0="H_A"*20, s1="L_A"*20, bc="X", loc0="IGH", loc1="IGK"):
    p=tmp/"annotated"; p.mkdir(exist_ok=True)
    pl.DataFrame({"name":[f"{bc}_d0_w0"],"well":["0"],"sequence_id:0":[f"{bc}_contig-0"],"sequence:0":[s0],
        "locus:0":[loc0],"sequence_id:1":[f"{bc}_contig-1"],"sequence:1":[s1],"locus:1":[loc1]}).write_parquet(p/"w_paired.parquet")
    return tmp
def test_correct(tmp_path):
    ps,_=score(_pp(tmp_path),_truth(tmp_path)); r=ps.to_dicts()[0]
    assert r["pairing_status"]=="correct" and r["origin_status"]=="resident"
def test_orientation_agnostic(tmp_path):
    # swap the output columns AND lie about loci; truth-based orientation must still resolve
    ps,_=score(_pp(tmp_path, s0="L_A"*20, s1="H_A"*20, loc0="IGK", loc1="IGH"), _truth(tmp_path))
    assert ps.to_dicts()[0]["pairing_status"]=="correct"
def test_missing_key(tmp_path):
    ps,ks=score(_pp(tmp_path, bc="Z"), _truth(tmp_path))
    assert ps.to_dicts()[0]["key_status"]=="unknown"
    assert ks.filter((pl.col("well")==0)&(pl.col("barcode")=="X")).to_dicts()[0]["output_status"]=="missing"
```

- [ ] **Step 2: run → FAIL.**   - [ ] **Step 3: implement**
```python
import re
from pathlib import Path
import polars as pl
from .matching import candidates, resolve
_LIGHT=("IGK","IGL")

def _files(x):
    if isinstance(x,(list,tuple)): return [Path(p) for p in x]
    x=Path(x)
    return sorted(x.glob("**/*_paired.parquet")) if x.is_dir() else [x]

def _bc(sid): return re.split(r"_contig",sid)[0] if sid else sid

def _index(comp):
    idx={}
    for r in comp.iter_rows(named=True):
        e=idx.setdefault((int(r["final_well"]),r["barcode"]),{}).setdefault(r["locus"],{})
        e.setdefault(r["sequence"],set()).add(r["source_pair_id"])
    return idx

def _lights(seq, entry):
    out=set()
    for L in _LIGHT: out|=candidates(seq,L,entry)
    return out

def _classify_origin(valid_assignments, resident):
    cats=set()
    for h,l in valid_assignments:
        hr,lr=h in resident,l in resident
        cats.add("resident" if hr and lr else "ambient" if not hr and not lr else "resident_plus_ambient")
    return cats.pop() if len(cats)==1 else ("ambiguous" if cats else "unknown")

def score(pairplex_output, truth_dir, *, pairplex_metadata=None):
    truth_dir=Path(truth_dir)
    comp=pl.read_parquet(truth_dir/"truth_components.parquet")
    tbar=pl.read_parquet(truth_dir/"truth_barcodes.parquet")
    idx=_index(comp)
    resident_at={}
    for r in comp.filter(pl.col("is_resident_source")).iter_rows(named=True):
        resident_at.setdefault((int(r["final_well"]),r["barcode"]),set()).add(r["source_pair_id"])
    kstat={(int(r["well"]),r["barcode"]):("collision" if r["is_collision"] else "ambient_only" if r["is_ambient_only"] else "singleton") for r in tbar.iter_rows(named=True)}

    files=_files(pairplex_output)
    rows,seen=[],{}
    for f in files:
        df=pl.read_parquet(f)
        for r in df.to_dicts():
            well=int(r["well"]); bc=_bc(r.get("sequence_id:0") or r.get("name","")); key=(well,bc); entry=idx.get(key)
            s0,s1=r.get("sequence:0"),r.get("sequence:1")
            res_here=resident_at.get(key,set())
            # try BOTH orientations against TRUTH loci (never trust PairPlex's annotation)
            results=[]
            for hseq,lseq in ((s0,s1),(s1,s0)):                  # (heavy,light) candidate orientation
                h=candidates(hseq,"IGH",entry); l=_lights(lseq,entry)
                if h and l: results.append(resolve(h,l))
            if not results:
                pstat,sres,resolved,origin=("unmatchable","none",None,"unknown")
            elif len(results)==1 or results[0].valid_assignments==results[1].valid_assignments:
                r0=results[0]; pstat,sres,resolved=r0.pairing_status,r0.source_resolution,r0.resolved_source
                origin=_classify_origin(r0.valid_assignments,res_here)
            else:                                                # two orientations, incompatible interpretations
                pstat,sres,resolved,origin=("ambiguous","none",None,"ambiguous")
            seen[key]=seen.get(key,0)+1
            rows.append({"pair_id":f"{f.stem}:{r.get('sequence_id:0')}","source_file":str(f),
                "well":well,"barcode":bc,"sequence_id:0":r.get("sequence_id:0"),"sequence_id:1":r.get("sequence_id:1"),
                "pairing_status":pstat,"source_resolution":sres,"origin_status":origin,
                "key_status":kstat.get(key,"unknown"),"output_status":"unique","resolved_source":resolved})
    for pr in rows:
        if seen[(pr["well"],pr["barcode"])]>1: pr["output_status"]="duplicate"
    pair_scores=pl.DataFrame(rows) if rows else pl.DataFrame(schema={c:pl.Utf8 for c in
        ["pair_id","source_file","barcode","sequence_id:0","sequence_id:1","pairing_status","source_resolution","origin_status","key_status","output_status","resolved_source"]}|{"well":pl.Int64})

    key_rows=[]
    for r in tbar.iter_rows(named=True):
        well,bc=int(r["well"]),r["barcode"]; oc=seen.get((well,bc),0)
        key_rows.append({"well":well,"barcode":bc,
            "key_status":("collision" if r["is_collision"] else "ambient_only" if r["is_ambient_only"] else "singleton"),
            "output_count":oc,"output_status":("missing" if oc==0 else "unique" if oc==1 else "duplicate"),
            "n_resident_cells":r.get("n_resident_cells",0),
            "captured_both":r.get("n_captured_both_resident_cells",0)>0,
            "survived_both":r.get("n_survived_both_resident_cells",0)>0,
            "sequenced_both":r.get("n_sequenced_both_resident_cells",0)>0,
            "reference_pairable_both":r.get("n_reference_pairable_resident_cells",0)>0,
            "no_output_reason":None if oc>0 else "unknown"})
    return pair_scores, pl.DataFrame(key_rows)
```
> `no_output_reason` refines beyond `unknown` only when `pairplex_metadata` is supplied.

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/scoring.py simplex/tests/test_scoring.py && git commit -m "feat(simplex): scorer — orientation-agnostic, joint, key-level, origin enumeration"`

---

### Task 4: load_pairs (locus + consistency) + barcodes

**Files:** Create `simplex/barcodes.py`, `simplex/cells.py`, `simplex/tests/test_load.py`.

- [ ] **Step 1: failing tests**
```python
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
```

- [ ] **Step 2: run → FAIL.**   - [ ] **Step 3: implement**

`simplex/barcodes.py`:
```python
import gzip
from pathlib import Path
from pairplex.utils import get_whitelist_path
def load_barcodes(chemistry,n,rng):
    p=Path(get_whitelist_path(chemistry.lower()))
    op=gzip.open if str(p).endswith(".gz") else open
    with op(p,"rt") as f: wl=[l.strip() for l in f if l.strip()]
    if n>len(wl): raise ValueError(f"need {n}, whitelist has {len(wl)}")
    return [wl[i] for i in rng.choice(len(wl),size=n,replace=False)]
```
`simplex/cells.py`:
```python
import numpy as np, polars as pl
from ._rng import rng_for
from .barcodes import load_barcodes
def load_pairs(input_data, n_cells=None, seed=0):
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
```

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/barcodes.py simplex/cells.py simplex/tests/test_load.py && git commit -m "feat(simplex): load_pairs (locus + consistency) + barcode loader"`

---

### Task 5: droplets (`barcode_pool_size`) + wells + analytic collision

**Files:** Modify `simplex/cells.py`; Create `simplex/tests/test_cells.py`.

**Produces:** `assign_droplets_and_barcodes(cells,mean,sd,chemistry,barcode_pool_size,seed)`; `assign_wells(cells,wells,seed)`. `barcode_pool_size` None → unique per droplet, int → sample droplet barcodes from a pool of that size.

- [ ] **Step 1: failing tests** — `simplex/tests/test_cells.py`:
```python
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
```
- [ ] **Step 2: run → FAIL.**   - [ ] **Step 3: implement** *(append to `simplex/cells.py`):*
```python
def assign_droplets_and_barcodes(cells, mean, sd, chemistry, barcode_pool_size, seed):
    rng=rng_for(seed,"droplets"); n=cells.height; order=rng.permutation(n); droplet=np.empty(n,np.int64); i=d=0
    while i<n:
        for _ in range(max(1,int(round(rng.normal(mean,sd))))):
            if i>=n: break
            droplet[order[i]]=d; i+=1
        d+=1
    brng=rng_for(seed,"barcodes")
    if barcode_pool_size:
        pool=np.array(load_barcodes(chemistry,min(barcode_pool_size,d),brng)); bc=pool[brng.integers(0,len(pool),size=d)]
    else:
        bc=np.array(load_barcodes(chemistry,d,brng))
    return cells.with_columns([pl.Series("droplet_id",droplet),pl.Series("barcode",bc[droplet])])
def assign_wells(cells,wells,seed):
    return cells.with_columns(pl.Series("resident_well",rng_for(seed,"wells").integers(0,wells,size=cells.height).astype(np.int64)))
```
- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/cells.py simplex/tests/test_cells.py && git commit -m "feat(simplex): droplets (barcode_pool_size) + wells + analytic collision"`

---

### Task 6: molecules — chain_status, free split, RT error, **empty-safe**

**Files:** Create `simplex/molecules.py`, `simplex/tests/test_molecules.py`.

**Produces:** `generate_molecules(...) -> (molecules_df, chain_status_df)`. Returns a **typed empty molecules frame** when nothing is captured.

- [ ] **Step 1: failing tests** (add zero-recovery case)
```python
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
```

- [ ] **Step 2: run → FAIL.**   - [ ] **Step 3: implement**
```python
import numpy as np, polars as pl
from ._dna import random_dna, mutate_strings
from ._rng import rng_for
_MOL_SCHEMA={"molecule_id":pl.Int64,"parent_molecule_id":pl.Int64,"origin_cell_id":pl.Int64,
    "origin_droplet_id":pl.Int64,"source_pair_id":pl.Utf8,"chain":pl.Int8,"locus":pl.Utf8,"umi":pl.Utf8,
    "barcode":pl.Utf8,"resident_well":pl.Int64,"amplification_well":pl.Int64,"is_free":pl.Boolean,
    "survived":pl.Boolean,"cdna":pl.Utf8}
def generate_molecules(cells,recovery_rate,molecules_per_chain_mean,release_rate,umi_length,rt_sub_rate,rt_indel_rate,seed):
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
```
> `amplification_well`/`survived` are placeholders set by routing; included so the schema is stable even for the empty frame.

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/molecules.py simplex/tests/test_molecules.py && git commit -m "feat(simplex): molecules + chain_status, free split, RT error, empty-safe"`

---

### Task 7: routing — survival flag kept, redistribution, amplification, per-read hop, **empty-safe**

**Files:** Create `simplex/routing.py`, `simplex/tests/test_routing.py`.

**Produces:** `route_and_amplify(molecules,wells,molecule_survival_rate,reads_per_molecule_mean,index_hop_rate,seed) -> (molecules_with_survival, reads)`. Keeps all molecules (sets `amplification_well`/`survived`); expands only survivors; returns typed empty `reads` when nothing survives.

- [ ] **Step 1: failing tests** — `simplex/tests/test_routing.py`:
```python
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
```
- [ ] **Step 2: run → FAIL.**   - [ ] **Step 3: implement** *(`simplex/routing.py`):*
```python
import numpy as np, polars as pl
from ._rng import rng_for
_READS_SCHEMA={"read_id":pl.Utf8,"molecule_id":pl.Int64,"origin_cell_id":pl.Int64,"source_pair_id":pl.Utf8,
    "chain":pl.Int8,"locus":pl.Utf8,"umi":pl.Utf8,"barcode":pl.Utf8,"amplification_well":pl.Int64,
    "final_well":pl.Int64,"is_free":pl.Boolean,"is_index_hopped":pl.Boolean,"cdna":pl.Utf8,"n_seq_errors":pl.Int64}
def route_and_amplify(molecules,wells,molecule_survival_rate,reads_per_molecule_mean,index_hop_rate,seed):
    rng=rng_for(seed,"routing"); n=molecules.height
    free=molecules["is_free"].to_numpy() if n else np.array([],bool)
    amp=(np.where(free,rng.integers(0,wells,size=n),molecules["resident_well"].to_numpy()).astype(np.int64) if n else np.array([],np.int64))
    surv=rng.random(n)<molecule_survival_rate if n else np.array([],bool)
    mols=molecules.with_columns([pl.Series("amplification_well",amp),pl.Series("survived",surv)]) if n else molecules
    survd=mols.filter(pl.col("survived"))
    if survd.height==0:
        return mols, pl.DataFrame(schema=_READS_SCHEMA)
    depth=np.maximum(rng.poisson(reads_per_molecule_mean,survd.height),1).astype(np.int64)
    rep=np.repeat(np.arange(survd.height),depth); reads=survd[rep]; k=reads.height
    hop=rng.random(k)<index_hop_rate; off=rng.integers(1,max(2,wells),size=k); a=reads["amplification_well"].to_numpy()
    final=np.where(hop,(a+off)%wells,a).astype(np.int64)
    reads=reads.with_columns([pl.Series("read_id",[f"r{i}" for i in range(k)]),pl.Series("final_well",final),
        pl.Series("is_index_hopped",hop),pl.lit(0,pl.Int64).alias("n_seq_errors")]).select(list(_READS_SCHEMA.keys()))
    return mols, reads
```
- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/routing.py simplex/tests/test_routing.py && git commit -m "feat(simplex): routing — survival kept, redistribution, amplification, per-read hop, empty-safe"`

---

### Task 8: sequencing errors + merged reads (**fully inlined**, round-trip)

**Files:** Create `simplex/reads.py`, `simplex/tests/test_reads.py`.

**Produces:** `apply_sequencing_errors(reads,sub_rate,indel_rate,seed)`; `build_merged(reads,tso,rc_fraction,variable_length,seed)`.

- [ ] **Step 1: failing tests**
```python
import polars as pl
from simplex.reads import apply_sequencing_errors, build_merged
from pairplex.utils import parse_barcodes
def _reads(n=3):
    return pl.DataFrame({"read_id":[f"r{i}" for i in range(n)],"final_well":[0]*n,
        "barcode":["ACGTACGTACGTACGT"]*n,"umi":["AAAAAAAAAA"]*n,"cdna":["GATTACAGGT"*20]*n,"n_seq_errors":[0]*n})
def test_seq_error(): 
    r=apply_sequencing_errors(_reads(500),0.05,0.0,0); assert r["n_seq_errors"].sum()>0
def test_zero_error(): 
    r=apply_sequencing_errors(_reads(),0.0,0.0,0); assert r["cdna"][0]=="GATTACAGGT"*20 and r["n_seq_errors"].sum()==0
def test_merged_round_trip(tmp_path):
    b=build_merged(_reads(),"TTTCTTATATGGG",0.0,False,0); s=b["read_seq"][0]
    assert s[:16]=="ACGTACGTACGTACGT" and s[16:26]=="AAAAAAAAAA"
    assert s[36:].lstrip("G")==("GATTACAGGT"*20).lstrip("G") and len(b["qual"][0])==len(s)
    # actual parse: write a fastq and confirm pairplex.parse_barcodes recovers the barcode
    fq=tmp_path/"r.fastq"; fq.write_text("".join(f"@{i}\n{seq}\n+\n{q}\n" for i,seq,q in zip(b["read_id"],b["read_seq"],b["qual"])))
    out=parse_barcodes(str(fq), str(tmp_path), whitelist_path=None, check_rc=True)
    if out is not None:
        assert pl.read_parquet(out)["barcode"][0]=="ACGTACGTACGTACGT"
def test_rc(tmp_path):
    from simplex._dna import revcomp_str
    b=build_merged(_reads(),"TTTCTTATATGGG",1.0,False,0); assert revcomp_str(b["read_seq"][0])[:16]=="ACGTACGTACGTACGT"
def test_variable_length():
    b=build_merged(_reads(),"TTTCTTATATGGG",0.0,True,0); assert all(len(s)>36 for s in b["read_seq"])
```
> The `parse_barcodes` assertion is guarded by `if out is not None` because the default whitelist may not contain this literal barcode; the layout assertions above are the hard guarantee. To make the parse assertion unconditional, the fixture in Task 11 uses a **real whitelist barcode** (`load_barcodes`).

- [ ] **Step 2: run → FAIL.**   - [ ] **Step 3: implement**
```python
import numpy as np, polars as pl
from ._dna import mutate_strings, revcomp_expr
from ._rng import rng_for
def apply_sequencing_errors(reads,sub_rate,indel_rate,seed):
    if reads.height==0 or (sub_rate==0 and indel_rate==0): return reads
    cdna,ne=mutate_strings(list(reads["cdna"]),sub_rate,indel_rate,rng_for(seed,"seqerr"))
    return reads.with_columns([pl.Series("cdna",cdna),(pl.col("n_seq_errors")+pl.Series(ne)).alias("n_seq_errors")])
def build_merged(reads,tso,rc_fraction,variable_length,seed):
    if reads.height==0:
        return pl.DataFrame(schema={"read_id":pl.Utf8,"final_well":pl.Int64,"read_seq":pl.Utf8,"qual":pl.Utf8})
    r=reads
    if variable_length:
        rng=rng_for(seed,"trunc"); lens=r["cdna"].str.len_chars().to_numpy()
        t5=rng.integers(0,np.maximum(1,lens//10)).astype(np.int64)
        nl=np.maximum(1,lens-t5-rng.integers(0,np.maximum(1,lens//10))).astype(np.int64)
        r=r.with_columns(pl.col("cdna").str.slice(pl.Series(t5),pl.Series(nl)).alias("cdna"))
    r=r.with_columns(pl.concat_str([pl.col("barcode"),pl.col("umi"),pl.lit(tso),pl.col("cdna")]).alias("_frag"))
    rc=pl.Series(rng_for(seed,"rc").random(r.height)<rc_fraction)
    r=r.with_columns(rc.alias("_rc")).with_columns(
        pl.when(pl.col("_rc")).then(revcomp_expr("_frag")).otherwise(pl.col("_frag")).alias("read_seq"))
    r=r.with_columns(pl.col("read_seq").str.replace_all(".","I").alias("qual"))
    return r.select(["read_id","final_well","read_seq","qual"])
```

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/reads.py simplex/tests/test_reads.py && git commit -m "feat(simplex): sequencing errors + merged reads (inlined, parse round-trip)"`

---

### Task 9: truth — components, cells (capture/survival), barcodes (per-cell counts + ties)

**Files:** Create `simplex/truth.py`, `simplex/tests/test_truth.py`.

**Produces:** `build_truth_components(cells,reads)`; `build_truth_cells(cells,chain_status,molecules,reads)`; `build_truth_barcodes(cells,truth_cells,components)` (four resident-cell counts computed **per cell at its home key**, dominance with tie flags).

- [ ] **Step 1: failing tests**
```python
import polars as pl
from simplex.truth import build_truth_components, build_truth_cells, build_truth_barcodes
def _cells():
    return pl.DataFrame({"cell_id":[0,1],"source_pair_id":["A","B"],
        "chain0_id":["hA","hB"],"chain0_seq":["HA","HB"],"chain0_locus":["IGH","IGH"],
        "chain1_id":["lA","lB"],"chain1_seq":["LA","LB"],"chain1_locus":["IGK","IGK"],
        "droplet_id":[0,0],"barcode":["X","X"],"resident_well":[0,0]})   # A,B collide on X@well0
def _status():
    return pl.DataFrame({"cell_id":[0,0,1,1],"chain":[0,1,0,1],"captured":[True,True,True,False],"n_molecules":[2,2,1,0]})
def _mols():
    return pl.DataFrame({"molecule_id":[0,1,2],"origin_cell_id":[0,0,1],"chain":[0,1,0],"survived":[True,True,True]})
def _reads():   # only cell0 produced reads; cell1 read-less but physically resident
    return pl.DataFrame({"read_id":["r0","r1"],"molecule_id":[0,1],"origin_cell_id":[0,0],"source_pair_id":["A","A"],
        "chain":[0,1],"locus":["IGH","IGK"],"barcode":["X","X"],"final_well":[0,0],
        "is_free":[False,False],"is_index_hopped":[False,False],"umi":["u0","u1"]})
def test_occupancy_from_cells():
    comp=build_truth_components(_cells(),_reads())
    tc=build_truth_cells(_cells(),_status(),_mols(),_reads())
    tb=build_truth_barcodes(_cells(),tc,comp)
    x=tb.filter((pl.col("well")==0)&(pl.col("barcode")=="X")).to_dicts()[0]
    assert x["n_resident_cells"]==2 and x["is_collision"] is True          # both counted incl read-less B
    assert x["n_sequenced_both_resident_cells"]==1                          # only A sequenced both chains
def test_cells_capture():
    tc=build_truth_cells(_cells(),_status(),_mols(),_reads())
    assert tc.filter(pl.col("cell_id")==1).to_dicts()[0]["captured_1"] is False
```

- [ ] **Step 2: run → FAIL.**   - [ ] **Step 3: implement**
```python
import polars as pl
from ._contract import REF_MIN_READS, REF_MIN_UMIS

# full aggregate schema so a completely read-less run still yields a valid (empty) component table
_COMP_AGG_SCHEMA={"final_well":pl.Int64,"barcode":pl.Utf8,"origin_cell_id":pl.Int64,"chain":pl.Int8,
    "source_pair_id":pl.Utf8,"locus":pl.Utf8,"n_reads":pl.Int64,"n_reads_resident":pl.Int64,
    "n_reads_free":pl.Int64,"n_reads_index_hopped":pl.Int64,"n_umis":pl.Int64,"n_source_molecules":pl.Int64}

def _cc_seq(cells):
    parts=[]
    for ch in (0,1):
        parts.append(cells.select([pl.col("cell_id").alias("origin_cell_id"),pl.lit(ch).cast(pl.Int8).alias("chain"),
            pl.col(f"chain{ch}_seq").alias("sequence"),pl.col(f"chain{ch}_locus").alias("locus"),
            pl.col("resident_well"),pl.col("barcode").alias("home_barcode")]))
    return pl.concat(parts)

def build_truth_components(cells,reads):
    cs=_cc_seq(cells)
    if reads.height==0:
        agg=pl.DataFrame(schema=_COMP_AGG_SCHEMA)
    else:
        agg=reads.group_by(["final_well","barcode","origin_cell_id","chain"]).agg([
            pl.col("source_pair_id").first(),pl.col("locus").first(),pl.len().alias("n_reads"),
            (~pl.col("is_free")&~pl.col("is_index_hopped")).sum().alias("n_reads_resident"),
            pl.col("is_free").sum().alias("n_reads_free"),pl.col("is_index_hopped").sum().alias("n_reads_index_hopped"),
            pl.col("umi").n_unique().alias("n_umis"),
            (pl.col("molecule_id").n_unique() if "molecule_id" in reads.columns else pl.col("umi").n_unique()).alias("n_source_molecules")])
    comp=agg.join(cs.select(["origin_cell_id","chain","sequence","resident_well","home_barcode"]),on=["origin_cell_id","chain"],how="left")
    return comp.with_columns(((pl.col("resident_well")==pl.col("final_well"))&(pl.col("home_barcode")==pl.col("barcode"))).alias("is_resident_source")).drop(["resident_well","home_barcode"])

def build_truth_cells(cells,chain_status,molecules,reads):
    surv=(molecules.filter(pl.col("survived")).group_by(["origin_cell_id","chain"]).len().rename({"origin_cell_id":"cell_id","len":"sn"}))
    if reads.height:
        rc=reads.group_by(["origin_cell_id","chain"]).agg([pl.len().alias("n_reads_generated"),
            (~pl.col("is_free")).sum().alias("n_reads_resident"),pl.col("is_free").sum().alias("n_reads_free_out"),
            pl.col("is_index_hopped").sum().alias("n_reads_index_hopped_out"),pl.col("umi").n_unique().alias("n_umis")]).rename({"origin_cell_id":"cell_id"})
    else:
        rc=pl.DataFrame(schema={"cell_id":pl.Int64,"chain":pl.Int8,"n_reads_generated":pl.Int64,"n_reads_resident":pl.Int64,"n_reads_free_out":pl.Int64,"n_reads_index_hopped_out":pl.Int64,"n_umis":pl.Int64})
    st=(chain_status.join(surv,on=["cell_id","chain"],how="left").with_columns((pl.col("sn").fill_null(0)>0).alias("survived"))
        .join(rc,on=["cell_id","chain"],how="left").fill_null(0))
    wide=st.pivot(index="cell_id",on="chain",values=["captured","survived","n_molecules","n_umis",
        "n_reads_generated","n_reads_resident","n_reads_free_out","n_reads_index_hopped_out"])
    return cells.join(wide,on="cell_id",how="left")

def build_truth_barcodes(cells,truth_cells,components):
    physical=cells.select([pl.col("resident_well").alias("well"),pl.col("barcode"),pl.col("cell_id"),pl.col("source_pair_id")])
    occ=physical.group_by(["well","barcode"]).agg([pl.col("source_pair_id").unique().alias("resident_source_ids"),
        pl.col("cell_id").n_unique().alias("n_resident_cells")])
    # capture/survival per resident cell at home key (join truth_cells onto physical)
    tc=truth_cells.select(["cell_id","captured_0","captured_1","survived_0","survived_1"])
    cap=physical.join(tc,on="cell_id",how="left").with_columns([
        (pl.col("captured_0")&pl.col("captured_1")).alias("cap_both"),
        (pl.col("captured_0")&pl.col("captured_1")&pl.col("survived_0")&pl.col("survived_1")).alias("surv_both")])
    capk=cap.group_by(["well","barcode"]).agg([pl.col("cap_both").sum().alias("n_captured_both_resident_cells"),
        pl.col("surv_both").sum().alias("n_survived_both_resident_cells")])
    # sequenced/reference per resident cell at home key from components (resident-source rows are AT home)
    res=components.filter(pl.col("is_resident_source"))
    # observability uses TOTAL observable support at the home key (n_reads/n_umis), not only
    # cell-associated reads: a free molecule of the same cell that returns home is legit support.
    per=res.group_by([pl.col("final_well").alias("well"),"barcode","origin_cell_id"]).agg([
        (pl.col("chain").n_unique()==2).alias("seq_both"),
        ((pl.col("n_reads").min()>=REF_MIN_READS)&(pl.col("n_umis").min()>=REF_MIN_UMIS)&(pl.col("chain").n_unique()==2)).alias("ref_both")])
    seqk=per.group_by(["well","barcode"]).agg([pl.col("seq_both").sum().alias("n_sequenced_both_resident_cells"),
        pl.col("ref_both").sum().alias("n_reference_pairable_resident_cells")])
    observed=components.select([pl.col("final_well").alias("well"),pl.col("barcode")]).unique()
    keys=pl.concat([occ.select(["well","barcode"]),observed]).unique()
    tb=keys.join(occ,on=["well","barcode"],how="left").join(capk,on=["well","barcode"],how="left").join(seqk,on=["well","barcode"],how="left")
    # per-locus dominance with tie detection, by reads and umis
    def dom(loci,by,name):
        col=f"dominant_{name}_source_by_{by.replace('n_','')}"; tie=f"{name}_dominance_is_tied_{by}"
        # aggregate support by source_pair_id FIRST (clonal copies across cells sum, not split)
        f=(components.filter(pl.col("locus").is_in(loci))
             .group_by([pl.col("final_well").alias("well"),"barcode","source_pair_id"]).agg(pl.col(by).sum().alias("supp")))
        g=(f.group_by(["well","barcode"]).agg([
             pl.col("source_pair_id").sort_by("supp",descending=True).alias("srcs"),
             pl.col("supp").sort(descending=True).alias("vals")]))
        return g.with_columns([pl.col("srcs").list.first().alias(col),
            ((pl.col("vals").list.len()>1)&(pl.col("vals").list.get(0)==pl.col("vals").list.get(1))).fill_null(False).alias(tie)]) \
            .select(["well","barcode",col,tie])
    for loci,name in [(["IGH"],"heavy"),(["IGK","IGL"],"light")]:
        for by in ("n_reads","n_umis"):
            tb=tb.join(dom(loci,by,name),on=["well","barcode"],how="left")
    return tb.with_columns([pl.col("n_resident_cells").fill_null(0),
        pl.col("n_captured_both_resident_cells").fill_null(0),pl.col("n_survived_both_resident_cells").fill_null(0),
        pl.col("n_sequenced_both_resident_cells").fill_null(0),pl.col("n_reference_pairable_resident_cells").fill_null(0),
        (pl.col("n_resident_cells").fill_null(0)>=2).alias("is_collision"),
        (pl.col("n_resident_cells").fill_null(0)==0).alias("is_ambient_only")])
```
> The two `*_dominance_is_tied_*` columns per locus keep read- and UMI-tie flags separate; consolidate to one `heavy_dominance_is_tied`/`light_dominance_is_tied` if only the read-based tie is needed downstream.

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/truth.py simplex/tests/test_truth.py && git commit -m "feat(simplex): truth — capture/survival, occupancy from cells, per-cell counts, dominance ties"`

---

### Task 10: IO writers + run() (guards, full manifest, zero-run safe)

**Files:** Create `simplex/io.py`, `simplex/run.py`, `simplex/tests/test_run.py`.

**Produces:** `io.write_merged_fastq(built, output_dir, compress=True)` (in-memory per-well writer); `io.write_truth(...)`; `run.run(input_data, output_directory, **knobs) -> Path`.

- [ ] **Step 1: failing tests** — `simplex/tests/test_run.py`:
```python
import gzip, json
from pathlib import Path
import polars as pl, pytest
from simplex.run import run
def _inp(tmp,n=60):
    d={"sequence_id:0":[f"h{i}" for i in range(n)],"sequence:0":["GATTACA"*30]*n,
       "sequence_id:1":[f"l{i}" for i in range(n)],"sequence:1":["CCGGTA"*30]*n,
       "name":[f"c{i}" for i in range(n)],"locus:0":["IGH"]*n,"locus:1":["IGK"]*n}
    p=tmp/"in.parquet"; pl.DataFrame(d).write_parquet(p); return p
def test_outputs_and_manifest(tmp_path):
    out=tmp_path/"o"; run(input_data=_inp(tmp_path),output_directory=out,wells=4,cells_per_droplet_mean=1,cells_per_droplet_sd=0,variable_length=False,seed=0)
    assert list((out/"reads").glob("*.fastq.gz"))
    for f in ["truth_components","truth_cells","truth_barcodes"]: assert (out/"truth"/f"{f}.parquet").exists()
    man=json.loads((out/"run_manifest.json").read_text())
    assert "input_fingerprint" in man and man["counts"]["reads"]>0 and "polars" in man
def test_refuses_nonempty(tmp_path):
    out=tmp_path/"o"; run(input_data=_inp(tmp_path),output_directory=out,wells=4,seed=0)
    with pytest.raises(FileExistsError): run(input_data=_inp(tmp_path),output_directory=out,wells=4,seed=0)
def test_reproducible(tmp_path):
    def content(d): return sorted(gzip.open(p,"rt").read() for p in Path(d).glob("*.fastq.gz"))
    a=run(input_data=_inp(tmp_path),output_directory=tmp_path/"a",wells=4,seed=5)
    b=run(input_data=_inp(tmp_path),output_directory=tmp_path/"b",wells=4,seed=5)
    assert content(a)==content(b)
def test_zero_recovery_run(tmp_path):  # full-pipeline zero-read case must not raise
    out=tmp_path/"z"; run(input_data=_inp(tmp_path),output_directory=out,wells=4,recovery_rate=0.0,seed=0)
    assert (out/"truth"/"truth_components.parquet").exists()
    assert pl.read_parquet(out/"truth"/"truth_components.parquet").height==0
def test_zero_survival_run(tmp_path):
    out=tmp_path/"z2"; run(input_data=_inp(tmp_path),output_directory=out,wells=4,molecule_survival_rate=0.0,seed=0)
    assert (out/"truth"/"truth_barcodes.parquet").exists()
```
- [ ] **Step 2: run → FAIL.**   - [ ] **Step 3: implement**

`simplex/io.py`:
```python
import gzip
from pathlib import Path
def _tag(w): return f"well{int(w):03d}"
def write_merged_fastq(built, output_dir, compress=True):   # in-memory per-well writer (v1 scale)
    rd=Path(output_dir)/"reads"; rd.mkdir(parents=True,exist_ok=True)
    ext="fastq.gz" if compress else "fastq"; op=(lambda p: gzip.open(p,"wt")) if compress else (lambda p: open(p,"w"))
    paths=[]
    if built.height==0: return paths
    for (well,),sub in built.group_by(["final_well"], maintain_order=True):
        p=rd/f"{_tag(well)}.{ext}"
        with op(p) as fh:
            fh.write("".join(f"@{i}\n{s}\n+\n{q}\n" for i,s,q in zip(sub["read_id"],sub["read_seq"],sub["qual"])))
        paths.append(p)
    return paths
def write_truth(output_dir, comp, cells, barcodes, reads=None):
    td=Path(output_dir)/"truth"; td.mkdir(parents=True,exist_ok=True)
    comp.write_parquet(td/"truth_components.parquet"); cells.write_parquet(td/"truth_cells.parquet")
    barcodes.write_parquet(td/"truth_barcodes.parquet")
    if reads is not None: reads.write_parquet(td/"truth_reads.parquet")   # single parquet in v1
```
`simplex/run.py`:
```python
import sys, hashlib, json
from pathlib import Path
import numpy, polars
from .cells import load_pairs, assign_droplets_and_barcodes, assign_wells
from .config import SimplexConfig
from .molecules import generate_molecules
from .routing import route_and_amplify
from .reads import apply_sequencing_errors, build_merged
from .truth import build_truth_components, build_truth_cells, build_truth_barcodes
from .io import write_merged_fastq, write_truth
try: from .version import __version__ as _SV
except Exception: _SV="0.0.0"
try: import pairplex; _PPV=getattr(pairplex,"__version__","unknown")
except Exception: _PPV="unknown"

def run(input_data, output_directory, **knobs):
    cfg=SimplexConfig(input_data=str(input_data), output_directory=str(output_directory), **knobs)
    out=Path(output_directory)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output dir {out} not empty; refusing to overwrite an experiment")
    out.mkdir(parents=True, exist_ok=True)
    cells=load_pairs(cfg.input_data, cfg.n_cells, cfg.seed); cfg.validate(actual_n_cells=cells.height)
    cells=assign_droplets_and_barcodes(cells, cfg.cells_per_droplet_mean, cfg.cells_per_droplet_sd, cfg.chemistry, cfg.barcode_pool_size, cfg.seed)
    cells=assign_wells(cells, cfg.wells, cfg.seed)
    mols, chain_status=generate_molecules(cells, cfg.recovery_rate, cfg.molecules_per_chain_mean, cfg.release_rate, cfg.umi_length, cfg.rt_sub_rate, cfg.rt_indel_rate, cfg.seed)
    mols, reads=route_and_amplify(mols, cfg.wells, cfg.molecule_survival_rate, cfg.reads_per_molecule_mean, cfg.index_hop_rate, cfg.seed)
    reads=apply_sequencing_errors(reads, cfg.sequencing_sub_rate, cfg.sequencing_indel_rate, cfg.seed)
    comp=build_truth_components(cells, reads)
    tcells=build_truth_cells(cells, chain_status, mols, reads)
    tbar=build_truth_barcodes(cells, tcells, comp)
    built=build_merged(reads, cfg.tso, cfg.rc_fraction, cfg.variable_length, cfg.seed)
    reads_paths=write_merged_fastq(built, out)
    write_truth(out, comp, tcells, tbar, reads if cfg.write_read_truth else None)
    cfg.to_json(out/"simplex_config.json")
    manifest={"simplex_version":_SV,"pairplex_version":_PPV,"python":sys.version.split()[0],
        "polars":polars.__version__,"numpy":numpy.__version__,"seed":cfg.seed,
        "input_fingerprint":hashlib.blake2b(Path(cfg.input_data).read_bytes(),digest_size=16).hexdigest(),
        "config_hash":hashlib.blake2b(json.dumps(cfg.to_dict(),sort_keys=True).encode(),digest_size=16).hexdigest(),
        "rng_scheme":"per-stage blake2b (v1: order-dependent, not chunk-invariant)",
        "counts":{"cells":cells.height,"molecules":mols.height,"reads":reads.height,"components":comp.height,"keys":tbar.height},
        "outputs":[p.name for p in reads_paths]+["truth/truth_components.parquet","truth/truth_cells.parquet","truth/truth_barcodes.parquet"]}
    (out/"run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return out/"reads"
```
- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/io.py simplex/run.py simplex/tests/test_run.py && git commit -m "feat(simplex): writers + run() with guards, full manifest, zero-run safe"`

---

### Task 11: controlled deterministic fixtures (all seven, valid input)

**Files:** Create `simplex/_fixtures.py`, `simplex/_testseqs.py`, `simplex/tests/test_mechanism.py`.

`_testseqs.py` loads four **distinct** real abstar bnAb H/L sequences (`HEAVY_A, LIGHT_A, HEAVY_B, LIGHT_B`) and provides `many_pairs_parquet(tmp, n)`. `_fixtures.emit(cells, chain_status, molecules, reads, out, *, write_read_truth=False)` builds+writes merged FASTQ and all truth (passing `truth_cells` into `build_truth_barcodes`). **Barcodes come from `load_barcodes("v2",…)`; UMIs are 10-mers.**

- [ ] **Step 1: write all seven** (compact but complete). Skeleton + the two hardest fully:
```python
# _fixtures.py
import polars as pl
from .reads import build_merged
from .io import write_merged_fastq, write_truth
from .truth import build_truth_components, build_truth_cells, build_truth_barcodes
def emit(cells, chain_status, molecules, reads, out, *, write_read_truth=False):
    comp=build_truth_components(cells,reads); tc=build_truth_cells(cells,chain_status,molecules,reads)
    write_merged_fastq(build_merged(reads,"TTTCTTATATGGG",0.0,False,0), out)
    write_truth(out, comp, tc, build_truth_barcodes(cells,tc,comp), reads if write_read_truth else None)
    return out/"reads"

# helper to make a 4-read resident/free family with a REAL barcode + 10-mer UMI
def family(mid, cell, spid, chain, locus, seq, well, barcode, umi, is_free, is_hopped=False, hop_one_to=None):
    fw=[well]*4
    if hop_one_to is not None: fw=[hop_one_to]+[well]*3   # route composition: one read hops
    hop=[is_hopped or (hop_one_to is not None and j==0) for j in range(4)]
    return pl.DataFrame({"read_id":[f"{mid}_{j}" for j in range(4)],"molecule_id":[mid]*4,"origin_cell_id":[cell]*4,
        "source_pair_id":[spid]*4,"chain":[chain]*4,"locus":[locus]*4,"umi":[umi]*4,"barcode":[barcode]*4,
        "amplification_well":[well]*4,"final_well":fw,"is_free":[is_free]*4,"is_index_hopped":hop,
        "cdna":[seq]*4,"n_seq_errors":[0]*4})
```
```python
# test_mechanism.py (exact ambient mispair; wells>=2; forced routing; consistent truth)
import pairplex, polars as pl
from simplex.barcodes import load_barcodes
from simplex._rng import rng_for
from simplex._fixtures import emit, family
from simplex._testseqs import HEAVY_A, LIGHT_A, HEAVY_B, LIGHT_B
from simplex.scoring import score
BC=load_barcodes("v2",1,rng_for(0,"fx"))[0]
def test_exact_ambient_mispair(tmp_path):
    cells=pl.DataFrame({"cell_id":[0,1],"source_pair_id":["A","B"],
        "chain0_id":["hA","hB"],"chain0_seq":[HEAVY_A,HEAVY_B],"chain0_locus":["IGH","IGH"],
        "chain1_id":["lA","lB"],"chain1_seq":[LIGHT_A,LIGHT_B],"chain1_locus":["IGK","IGK"],
        "droplet_id":[0,0],"barcode":[BC,BC],"resident_well":[0,1]})
    chain_status=pl.DataFrame({"cell_id":[0,0,1,1],"chain":[0,1,0,1],
        "captured":[True,False,False,True],"n_molecules":[1,0,0,1]})   # A: heavy only; B: light only
    molecules=pl.DataFrame({"molecule_id":[0,1],"origin_cell_id":[0,1],"chain":[0,1],"survived":[True,True]})
    reads=pl.concat([family(0,0,"A",0,"IGH",HEAVY_A,0,BC,"AAAAAAAAAA",False),      # A heavy resident @ well0
                     family(1,1,"B",1,"IGK",LIGHT_B,0,BC,"CCCCCCCCCC",True)])       # B light FREE -> well0
    rd=emit(cells,chain_status,molecules,reads,tmp_path/"sim")
    ppo=tmp_path/"pp"; pairplex.run(sequences=str(rd),output_directory=str(ppo),min_cluster_reads=1,min_cluster_umis=1,quiet=True)
    ps,_=score(ppo,(tmp_path/"sim"/"truth"))
    assert (ps["pairing_status"]=="mispaired").sum()>=1
```
Implement the other **six** (the ambient case above is fully shown; these six complete the seven) with the same `family`/`emit` pattern and consistent `chain_status`/`molecules`:
- **clean golden**: N cells, 1/barcode, all captured+survived, no free/errors → `pairing_status` all `correct`, no `ambiguous`/`unmatchable`.
- **one-cell negative control**: 1 cell/barcode, some free molecules + one chain dropped → `mispaired`==0 (`ambient_coherent` allowed).
- **same-well collision**: A,B same barcode, both `resident_well=0`, A-light & B-heavy absent → mispair at `key_status=="collision"`.
- **route composition**: one molecule, `hop_one_to=1`, `write_read_truth=True`; assert from `truth/truth_reads.parquet` that a read has `final_well!=amplification_well` with unchanged barcode+UMI.
- **joint ambiguity**: two source pairs share `HEAVY_A`, distinct lights; the correct light's pair resolves `pairing_status=="correct"`.
- **missing output**: resident A pair present + an extra contaminant heavy contig at the key so PairPlex rejects → `key_scores` `output_status=="missing"` for A's key.

`_testseqs.py`:
```python
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
```

- [ ] **Step 2–4:** run; iterate if a fixture exposes a real bug.   - [ ] **Step 5: commit** `git add simplex/_fixtures.py simplex/_testseqs.py simplex/tests/test_mechanism.py && git commit -m "test(simplex): seven controlled deterministic fixtures (valid input)"`

---

### Task 12: single-factor tests — correct recall, regime-specific directions

**Files:** Create `simplex/tests/test_single_factor.py`. Recall = **reference-pairable resident cells with a unique resident-correct output** (join `key_scores` × resident-correct `pair_scores`). Assert **regime-specific** directions with a **nontrivial-effect** guard.

- [ ] **Step 1: write tests**
```python
import pairplex, polars as pl
from simplex.run import run
from simplex.scoring import score
from simplex._testseqs import many_pairs_parquet

def metrics(ppo, truth):
    ps,ks=score(ppo,truth)
    mis=int((ps["pairing_status"]=="mispaired").sum())
    # recall over SINGLETON reference-pairable keys with a UNIQUE resident-correct output
    correct_keys={(r["well"],r["barcode"]) for r in ps.filter(
        (pl.col("pairing_status")=="correct")&(pl.col("origin_status")=="resident")
        &(pl.col("output_status")=="unique")&(pl.col("key_status")=="singleton")).to_dicts()}
    refpair=ks.filter(pl.col("reference_pairable_both")&(pl.col("key_status")=="singleton"))
    recall=sum(1 for r in refpair.to_dicts() if (r["well"],r["barcode"]) in correct_keys)/max(1,refpair.height)
    return mis, recall   # collision-key recovery is a SEPARATE metric (per-cell), not this key-level recall

def test_ambient_extra_contig_regime(tmp_path):
    # ambient adds a low-support extra chain: a fraction filter should reduce mispairs AND may raise recall
    inp=many_pairs_parquet(tmp_path,60); out=tmp_path/"sim"
    rd=run(input_data=inp,output_directory=out,wells=2,cells_per_droplet_mean=2,cells_per_droplet_sd=0,
           recovery_rate=0.9,release_rate=0.2,molecule_survival_rate=1.0,index_hop_rate=0.0,
           sequencing_sub_rate=0.0,variable_length=False,seed=1)
    lo=tmp_path/"lo"; pairplex.run(sequences=str(rd),output_directory=str(lo),min_cluster_reads=3,min_cluster_umis=1,min_cluster_fraction=0.0,quiet=True)
    hi=tmp_path/"hi"; pairplex.run(sequences=str(rd),output_directory=str(hi),min_cluster_reads=3,min_cluster_umis=1,min_cluster_fraction=0.3,quiet=True)
    mis_lo,rec_lo=metrics(lo,out/"truth"); mis_hi,rec_hi=metrics(hi,out/"truth")
    assert (mis_lo,rec_lo)!=(mis_hi,rec_hi)      # nontrivial effect
    assert mis_hi <= mis_lo                        # filtering ambient extra contigs reduces mispairs
```
Add a **weak-real-chain regime** test (low `reads_per_molecule_mean` for the real chain) asserting `rec_hi <= rec_lo`, and a **broad-sweep** test that only *reports* precision/mispair/recall/yield across a small grid without asserting a universal direction.

- [ ] **Step 2–4:** run; iterate.   - [ ] **Step 5: commit** `git add simplex/tests/test_single_factor.py && git commit -m "test(simplex): regime-specific single-factor tests with correct recall"`

---

### Task 13 (optional, Phase 0A): real-data marginal audit (normalized contract)

**Files:** Create `simplex/audit.py`, `simplex/tests/test_audit.py`.

**Produces:** `audit.normalize_metadata(raw) -> pl.DataFrame` with required fields `well, barcode, locus, reads, umis, cluster_fraction, pass_filters` (parsing PairPlex `metadata` `name`/filename where needed); `audit.audit_metadata(normalized_glob_or_df, report_path) -> pl.DataFrame` — marginal quantiles + per-`(well,barcode)` contig-count profile (1H+1L / 1H+2L / 2H+1L frequencies), **no calibration gate**, report header states the no-labeled-truth limitation.

- [ ] **Step 1–5:** TDD against a synthetic normalized fixture; assert marginal rows + caveat line. Commit `feat(simplex): optional Phase 0A marginal audit (normalized contract)`.

---

## Self-Review

**Six mandatory review items → resolved:** (1) Task 8 fully inlined incl. `parse_barcodes` round-trip; (2) `truth_barcodes` four counts implemented **per resident cell at home key** + shared `_contract` constants; (3) scorer orientation-agnostic (tries both, never trusts PairPlex loci; `ambiguous` now reachable) + mispaired origin via assignment enumeration + `pair_id`/`source_file`/`sequence_id`s on `pair_scores`; (4) fixtures use real whitelist barcodes + 10-mer UMIs + consistent `chain_status`/`molecules`, all seven spelled out, `emit(write_read_truth=…)`; (5) Task 12 recall = reference-pairable-resident with unique resident-correct output, regime-specific directions + nontrivial-effect guard; (6) reproducibility/scale wording corrected in spec v4.

**Contract/robustness → resolved:** molecule record drops per-read `is_index_hopped` (spec v4) and standardizes `barcode`; config validates positives/fixed-structure/one-well-hop/OOM-with-actual-cells; empty frames returned for zero-capture/zero-survival/zero-reads/zero-output (typed) + tests; repeated `source_pair_id` loci validated; dominance ties flagged; run manifest complete; Phase 0A normalized input contract.

**Placeholder scan:** no `...` in executable steps; Task 5/7/10 reuse is inlined with code (self-contained, no archived-plan dependency). **Type consistency:** `_MOL_SCHEMA`/`_READS_SCHEMA` pin frames; `chain_status`/`survived`/`molecule_id` flow T6→T7→T9; scorer axes stable; `score()` reads dir/list. **Watch:** polars `pivot`/`list` expressions on 1.39; edlib present; `pairplex.run` merged-mode throughout (no fastp dependency).
