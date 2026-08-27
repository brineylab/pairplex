# SimPlex Phase 0–2 Implementation Plan (v2, post plan-review)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the mechanistically-faithful SimPlex generator + compact truth + a `(well,barcode)`-keyed scorer, so we can drive PairPlex on synthetic data with known truth and measure precision/yield.

**Architecture:** Staged polars+numpy pipeline. **Phase 0B first**: freeze the scorer contract + bounded matcher against hand-crafted truth (this forces the truth schema, incl. `captured`/`survived`/`n_molecules`). Then the generator: molecules carry barcode+UMI; *all* molecules (incl. non-survivors) are retained with a `survived` flag; free molecules redistribute across wells pre-amplification; read families inherit RT error; index hopping moves reads post-amplification. See spec v3: `docs/superpowers/specs/2026-08-27-simplex-generator-design.md`.

**Tech Stack:** Python 3.10+, polars 1.39, numpy 2.x, edlib, pytest. Reuses `pairplex` whitelists + read structure.

## Global Constraints

- Sibling package `simplex/`; `import simplex`. Merged layout `barcode(16)+umi(10)+TSO+cDNA`; round-trips through `pairplex.parse_barcodes`. **`output_mode="merged"` only in Phase 1–2** (reject `"paired"`).
- **Reproducibility (v1 honest scope):** same seed + same input order + same layout → identical content. `rng_for(seed, stage)` via `SeedSequence(blake2b)`; per-stage streams, sequential within a stage. No chunk-size-invariance claim (Phase 5).
- **Locus required**: `load_pairs` fails if input lacks `locus:0/1`; repeated `source_pair_id` must describe identical sequences+loci.
- Scorer matches by **sequence** (bounded edit-distance via edlib), never `junction_aa`; candidate matches are **sets** of `source_pair_id`, **locus-restricted**, **key-local**; **any two non-empty candidate sets with empty intersection ⇒ mispaired**; `pairing_status` and `source_resolution` are separate axes; `score()` reads **all** wells jointly.
- Truth preserves `captured`/`survived`/`n_molecules`; barcode occupancy comes from **`cells`**, not observed reads.
- v1 scale ≤~50k cells, in-memory per-well writes. Commit after each task; tests in `simplex/tests/`.

## Canonical schemas

```
cells:     cell_id:i64, source_pair_id:str, chain0_id/seq/locus:str, chain1_id/seq/locus:str
  +droplet: droplet_id:i64, barcode:str    +well: resident_well:i64
chain_status: cell_id:i64, chain:i8, captured:bool, n_molecules:i64
molecules: molecule_id:i64, parent_molecule_id:i64, origin_cell_id:i64, origin_droplet_id:i64,
           source_pair_id:str, chain:i8, locus:str, umi:str, origin_barcode:str, final_barcode:str,
           resident_well:i64, amplification_well:i64, is_free:bool, survived:bool, cdna:str
reads:     read_id:str, molecule_id:i64, origin_cell_id:i64, source_pair_id:str, chain:i8, locus:str,
           umi:str, barcode:str, amplification_well:i64, final_well:i64, is_free:bool,
           is_index_hopped:bool, is_barcode_swapped:bool, cdna:str, n_seq_errors:i64
built(merged): read_id:str, final_well:i64, read_seq:str, qual:str
truth_components: (final_well, barcode, origin_cell_id, chain) + source_pair_id, locus, sequence,
           is_resident_source, n_source_molecules, n_umis, n_reads, n_reads_resident,
           n_reads_free, n_reads_index_hopped
truth_cells: cells + per-chain captured, survived, n_molecules, n_umis, n_reads_generated,
           n_reads_resident, n_reads_free_out, n_reads_index_hopped_out
truth_barcodes: (well, barcode) [union of physical (resident_well,barcode) & observed
           (final_well,barcode)] + resident_source_ids, n_resident_cells, is_collision,
           is_ambient_only, n_captured_both/n_survived_both/n_sequenced_both/
           n_reference_pairable_resident_cells, dominant_{heavy,light}_source_by_{reads,umis}
pair_scores: well, barcode, pairing_status, source_resolution, origin_status, key_status,
           output_status, resolved_source
key_scores: well, barcode, key_status, output_count, output_status, n_resident_cells,
           captured_both, survived_both, sequenced_both, reference_pairable_both, no_output_reason
```

---

### Task 1: Scaffold — keyed RNG, DNA helpers, config (guards)

**Files:** Create `simplex/__init__.py`, `simplex/_rng.py`, `simplex/_dna.py`, `simplex/config.py`, `simplex/tests/__init__.py`, `simplex/tests/test_rng.py`, `simplex/tests/test_dna.py`, `simplex/tests/test_config.py`. Modify `pyproject.toml` (packages incl `simplex*`).

**Produces:** `_rng.rng_for(seed, stage)`, `_dna.random_dna/revcomp_expr/revcomp_str/mutate_strings`, `config.SimplexConfig` with `.validate(actual_n_cells=None)`, `.to_json`, `.estimated_reads`.

- [ ] **Step 1: failing tests**

`simplex/tests/test_rng.py`:
```python
from simplex._rng import rng_for
def test_same_stage_same_stream():
    assert list(rng_for(0,"m").integers(0,1_000_000,50)) == list(rng_for(0,"m").integers(0,1_000_000,50))
def test_diff_stage_diff():
    assert list(rng_for(0,"m").integers(0,10**6,50)) != list(rng_for(0,"n").integers(0,10**6,50))
```
`simplex/tests/test_dna.py`:
```python
import numpy as np, polars as pl
from simplex._dna import random_dna, revcomp_str, revcomp_expr, mutate_strings
def test_random_dna():
    o = random_dna(np.random.default_rng(0),5,10); assert len(o)==5 and set("".join(o))<=set("ACGT")
def test_revcomp():
    assert revcomp_str("AAACCTGGN")=="NCCAGGTTT"
    assert pl.DataFrame({"s":["ACGT"]}).select(revcomp_expr("s"))["s"][0]=="ACGT"
def test_mutate():
    out,ne = mutate_strings(["ACGT"*50]*400,0.05,0.0,np.random.default_rng(0)); assert 4 < ne.mean() < 16
    o2,n2 = mutate_strings(["ACGT"],0.0,0.0,np.random.default_rng(0)); assert o2==["ACGT"] and n2.sum()==0
```
`simplex/tests/test_config.py`:
```python
import pytest
from simplex.config import SimplexConfig
def test_defaults(tmp_path):
    c = SimplexConfig(input_data="x",output_directory="o"); assert c.output_mode=="merged"
    c.to_json(tmp_path/"c.json"); assert (tmp_path/"c.json").exists()
def test_reject_paired():
    with pytest.raises(ValueError): SimplexConfig(input_data="x",output_directory="o",output_mode="paired").validate()
def test_reject_bad_rate():
    with pytest.raises(ValueError): SimplexConfig(input_data="x",output_directory="o",release_rate=1.5).validate()
def test_index_hop_one_well():
    with pytest.raises(ValueError): SimplexConfig(input_data="x",output_directory="o",wells=1,index_hop_rate=0.01).validate()
def test_oom_actual_cells():
    c = SimplexConfig(input_data="x",output_directory="o",reads_per_molecule_mean=50,molecules_per_chain_mean=50)
    with pytest.raises(ValueError): c.validate(actual_n_cells=10_000_000, max_reads=5_000_000_000)
```

- [ ] **Step 2: run → FAIL** (`python -m pytest simplex/tests/test_rng.py simplex/tests/test_dna.py simplex/tests/test_config.py -q`)

- [ ] **Step 3: implement**

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
_ASCII = np.array([65,67,71,84], np.uint8); _COMP = bytes.maketrans(b"ACGTN",b"TGCAN"); _B = np.array(list("ACGT"))
def random_dna(rng,k,length):
    if k==0: return np.array([],object)
    return _ASCII[rng.integers(0,4,size=(k,length),dtype=np.uint8)].view(f"S{length}").reshape(k).astype(str)
def revcomp_str(s): return s.translate(_COMP)[::-1]
def revcomp_expr(col): return pl.col(col).str.reverse().str.replace_many(["A","C","G","T"],["T","G","C","A"])
def mutate_strings(seqs, sub_rate, indel_rate, rng):
    out, cnt = [], np.zeros(len(seqs), np.int64)
    for i,s in enumerate(seqs):
        ch,n = list(s),0
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
    return out, cnt
```
`simplex/config.py`:
```python
import json
from dataclasses import asdict, dataclass
from pathlib import Path

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
    barcode_length: int = 16; umi_length: int = 10; tso: str = "TTTCTTATATGGG"; chemistry: str = "v2"
    output_mode: str = "merged"; read_length: int = 300; rc_fraction: float = 0.0; platform: str = "illumina"
    variable_length: bool = True; write_read_truth: bool = False; seed: int = 0

    _RATES = ("recovery_rate","release_rate","molecule_survival_rate","rt_sub_rate","rt_indel_rate",
              "sequencing_sub_rate","sequencing_indel_rate","index_hop_rate","rc_fraction")

    def to_dict(self): return asdict(self)
    def to_json(self, p): Path(p).write_text(json.dumps(self.to_dict(), indent=2))
    def estimated_reads(self, n):
        return int(n*2*self.recovery_rate*self.molecules_per_chain_mean*self.molecule_survival_rate*self.reads_per_molecule_mean)
    def validate(self, actual_n_cells=None, max_reads=3_000_000_000):
        for r in self._RATES:
            v=getattr(self,r)
            if not (0.0<=v<=1.0): raise ValueError(f"{r}={v} not in [0,1]")
        if self.output_mode != "merged": raise ValueError("Phase 1-2 supports output_mode='merged' only")
        if self.wells == 1 and self.index_hop_rate != 0: raise ValueError("index_hop_rate must be 0 when wells==1")
        n = actual_n_cells if actual_n_cells is not None else self.n_cells
        if n and self.estimated_reads(n) > max_reads:
            raise ValueError(f"est reads {self.estimated_reads(n)} > budget {max_reads}")
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
`pyproject.toml`: include `simplex*`.

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/ pyproject.toml && git commit -m "feat(simplex): scaffold — keyed RNG, DNA helpers, config guards"`

---

### Task 2 (Phase 0B): bounded matcher + joint resolver

**Files:** Create `simplex/matching.py`, `simplex/tests/test_matching.py`.

**Produces:** `matching.seq_match(a, b, max_frac=0.06, min_len=50) -> bool` (edlib infix); `matching.candidates(seq, locus, key_entry, **kw) -> set[str]`; `matching.resolve(h_cands, l_cands) -> (pairing_status, source_resolution, resolved_source|None)`.

- [ ] **Step 1: failing tests**
```python
from simplex.matching import resolve, seq_match
def test_disjoint_singletons_mispaired():
    assert resolve({"A"}, {"B"}) == ("mispaired", "none", None)
def test_disjoint_one_ambiguous_still_mispaired():
    assert resolve({"A","B"}, {"C"}) == ("mispaired", "none", None)   # empty intersection ⇒ mispaired
def test_unique_intersection_correct():
    assert resolve({"A","B"}, {"A"}) == ("correct", "unique", "A")
def test_nonunique_intersection_correct_ambiguous_source():
    assert resolve({"A","B"}, {"A","B"}) == ("correct", "ambiguous", None)
def test_empty_unmatchable():
    assert resolve(set(), {"A"}) == ("unmatchable", "none", None)
def test_seq_match_tolerates_one_sub():
    a = "ACGT"*30; b = a[:60] + "T" + a[61:]   # 1 substitution
    assert seq_match(a, b) and not seq_match(a, "TTTT"*30)
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement**

`simplex/matching.py`:
```python
import edlib

def seq_match(a, b, max_frac=0.06, min_len=50):
    if not a or not b: return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < min_len: return False
    r = edlib.align(short, long, mode="HW", task="distance")  # infix: short within long
    return 0 <= r["editDistance"] <= max_frac * len(short)

def candidates(seq, locus, key_entry, max_frac=0.06, min_len=50):
    if not seq or key_entry is None: return set()
    hits = set()
    for full, sources in key_entry.get(locus, {}).items():
        if seq == full or seq_match(seq, full, max_frac, min_len):
            hits |= sources
    return hits

def resolve(h_cands, l_cands):
    if not h_cands or not l_cands:
        return ("unmatchable", "none", None)
    inter = h_cands & l_cands
    if not inter:
        return ("mispaired", "none", None)     # non-empty sets, empty intersection ⇒ mispaired
    if len(inter) == 1:
        return ("correct", "unique", next(iter(inter)))
    return ("correct", "ambiguous", None)
```

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/matching.py simplex/tests/test_matching.py && git commit -m "feat(simplex): bounded matcher + joint pair resolver (Phase 0B)"`

---

### Task 3 (Phase 0B): scorer contract on hand-crafted truth

**Files:** Create `simplex/scoring.py`, `simplex/tests/test_scoring.py`. Uses hand-crafted truth (freezes the truth schema the generator must produce).

**Produces:** `scoring.score(pairplex_output, truth_dir, *, pairplex_metadata=None) -> (pair_scores, key_scores)`. `pairplex_output` = a directory (globs `**/*_paired.parquet`), a single parquet, or a list. Reads **all** jointly.

- [ ] **Step 1: failing tests**
```python
import polars as pl
from simplex.scoring import score

def _truth(tmp_path):
    td = tmp_path/"truth"; td.mkdir()
    pl.DataFrame({  # component key (well,barcode,cell,chain)
        "final_well":[0,0],"barcode":["X","X"],"origin_cell_id":[0,0],"source_pair_id":["A","A"],
        "chain":[0,1],"locus":["IGH","IGK"],"sequence":["H_A"*20,"L_A"*20],"is_resident_source":[True,True],
        "n_source_molecules":[3,3],"n_umis":[3,3],"n_reads":[9,9],
        "n_reads_resident":[9,9],"n_reads_free":[0,0],"n_reads_index_hopped":[0,0]}).write_parquet(td/"truth_components.parquet")
    pl.DataFrame({"cell_id":[0],"source_pair_id":["A"],"resident_well":[0],"barcode":["X"],
        "captured_0":[True],"captured_1":[True],"survived_0":[True],"survived_1":[True],
        "n_reads_generated_0":[9],"n_reads_generated_1":[9],"n_umis_0":[3],"n_umis_1":[3]}).write_parquet(td/"truth_cells.parquet")
    pl.DataFrame({"well":[0],"barcode":["X"],"n_resident_cells":[1],"is_collision":[False],
        "is_ambient_only":[False],"n_sequenced_both_resident_cells":[1],
        "n_reference_pairable_resident_cells":[1]}).write_parquet(td/"truth_barcodes.parquet")
    return td

def _pp(tmp_path, s0="H_A"*20, s1="L_A"*20, bc="X"):
    p = tmp_path/"annotated"; p.mkdir(exist_ok=True)
    f = p/"well000_paired.parquet"
    pl.DataFrame({"name":[f"{bc}_d0_w0"],"well":["0"],"sequence_id:0":[f"{bc}_contig-0_d0_w0"],
        "sequence:0":[s0],"locus:0":["IGH"],"sequence_id:1":[f"{bc}_contig-1_d0_w0"],
        "sequence:1":[s1],"locus:1":["IGK"]}).write_parquet(f)
    return p.parent   # pass the run dir; score globs **/ *_paired.parquet

def test_correct_resident(tmp_path):
    ps,_ = score(_pp(tmp_path), _truth(tmp_path))
    r = ps.to_dicts()[0]
    assert r["pairing_status"]=="correct" and r["origin_status"]=="resident" and r["well"]==0 and r["barcode"]=="X"

def test_wrong_barcode_is_key_unknown_and_missing(tmp_path):
    ppdir = _pp(tmp_path, bc="Z")   # emitted under Z (not in truth)
    ps, ks = score(ppdir, _truth(tmp_path))
    assert ps.to_dicts()[0]["key_status"]=="unknown"
    xrow = ks.filter((pl.col("well")==0)&(pl.col("barcode")=="X")).to_dicts()[0]
    assert xrow["output_status"]=="missing"
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement**

`simplex/scoring.py`:
```python
import re
from pathlib import Path
import polars as pl
from .matching import candidates, resolve

_REF_MIN_READS, _REF_MIN_UMIS = 3, 1
_LIGHT = ("IGK", "IGL")

def _paired_files(x):
    if isinstance(x, (list, tuple)): return [Path(p) for p in x]
    x = Path(x)
    if x.is_dir(): return sorted(x.glob("**/*_paired.parquet"))
    return [x]

def _barcode_from_id(sid):
    return re.split(r"_contig", sid)[0] if sid else sid

def _key_index(comp):
    idx = {}
    for r in comp.iter_rows(named=True):
        e = idx.setdefault((int(r["final_well"]), r["barcode"]), {}).setdefault(r["locus"], {})
        e.setdefault(r["sequence"], set()).add(r["source_pair_id"])
    return idx

def score(pairplex_output, truth_dir, *, pairplex_metadata=None):
    truth_dir = Path(truth_dir)
    comp = pl.read_parquet(truth_dir/"truth_components.parquet")
    tbar = pl.read_parquet(truth_dir/"truth_barcodes.parquet")
    idx = _key_index(comp)
    resident_at = {(int(r["final_well"]), r["barcode"]): set() for r in comp.iter_rows(named=True)}
    for r in comp.filter(pl.col("is_resident_source")).iter_rows(named=True):
        resident_at.setdefault((int(r["final_well"]), r["barcode"]), set()).add(r["source_pair_id"])
    key_status = {(int(r["well"]), r["barcode"]):
                  ("collision" if r["is_collision"] else "ambient_only" if r["is_ambient_only"] else "singleton")
                  for r in tbar.iter_rows(named=True)}

    df = pl.concat([pl.read_parquet(f) for f in _paired_files(pairplex_output)]) if _paired_files(pairplex_output) else pl.DataFrame()
    rows, seen = [], {}
    for r in (df.to_dicts() if df.height else []):
        well = int(r["well"]); bc = _barcode_from_id(r.get("sequence_id:0") or r.get("name",""))
        key = (well, bc); entry = idx.get(key)
        loc0 = r.get("locus:0")
        h_seq = r.get("sequence:0") if loc0 == "IGH" else r.get("sequence:1")
        l_seq = r.get("sequence:1") if loc0 == "IGH" else r.get("sequence:0")
        h_c = candidates(h_seq, "IGH", entry)
        l_c = set().union(*[candidates(l_seq, L, entry) for L in _LIGHT]) if entry else set()
        pstat, sres, resolved = resolve(h_c, l_c)
        # origin from resolved source(s)
        res_here = resident_at.get(key, set())
        cand_union = h_c | l_c
        if pstat == "correct" and resolved is not None:
            origin = "resident" if resolved in res_here else "ambient"
        elif pstat == "correct":  # ambiguous source
            origins = {"resident" if s in res_here else "ambient" for s in (h_c & l_c)}
            origin = origins.pop() if len(origins) == 1 else "ambiguous"
        elif pstat == "mispaired":
            h_res = any(s in res_here for s in h_c); l_res = any(s in res_here for s in l_c)
            origin = "resident_plus_ambient" if (h_res != l_res) else ("resident" if h_res and l_res else "ambient")
        else:
            origin = "unknown"
        seen[key] = seen.get(key, 0) + 1
        rows.append({"well":well,"barcode":bc,"pairing_status":pstat,"source_resolution":sres,
                     "origin_status":origin,"key_status":key_status.get(key,"unknown"),
                     "output_status":"unique","resolved_source":resolved})
    for pr in rows:
        if seen[(pr["well"],pr["barcode"])] > 1: pr["output_status"] = "duplicate"
    pair_scores = pl.DataFrame(rows) if rows else pl.DataFrame(schema={
        "well":pl.Int64,"barcode":pl.Utf8,"pairing_status":pl.Utf8,"source_resolution":pl.Utf8,
        "origin_status":pl.Utf8,"key_status":pl.Utf8,"output_status":pl.Utf8,"resolved_source":pl.Utf8})

    key_rows = []
    for r in tbar.iter_rows(named=True):
        well, bc = int(r["well"]), r["barcode"]; oc = seen.get((well, bc), 0)
        key_rows.append({"well":well,"barcode":bc,
            "key_status":("collision" if r["is_collision"] else "ambient_only" if r["is_ambient_only"] else "singleton"),
            "output_count":oc,
            "output_status":("missing" if oc==0 else "unique" if oc==1 else "duplicate"),
            "n_resident_cells": r.get("n_resident_cells", 0),
            "captured_both": r.get("n_captured_both_resident_cells", 0) > 0,
            "survived_both": r.get("n_survived_both_resident_cells", 0) > 0,
            "sequenced_both": r.get("n_sequenced_both_resident_cells", 0) > 0,
            "reference_pairable_both": r.get("n_reference_pairable_resident_cells", 0) > 0,
            "no_output_reason": None if oc>0 else "unknown"})
    return pair_scores, pl.DataFrame(key_rows)
```
> `no_output_reason` refines beyond `unknown` only when `pairplex_metadata` is supplied (best-effort; wire later).

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/scoring.py simplex/tests/test_scoring.py && git commit -m "feat(simplex): scorer contract — joint, orthogonal axes, key_scores (Phase 0B)"`

---

### Task 4: load_pairs (locus required) + barcode loader

**Files:** Create `simplex/barcodes.py`, `simplex/cells.py`, `simplex/tests/test_load.py`.

**Produces:** `barcodes.load_barcodes(chemistry, n, rng)`; `cells.load_pairs(input_data, n_cells=None, seed=0)` (raises if `locus:0/1` absent; validates repeated `source_pair_id` consistency).

- [ ] **Step 1: failing tests**
```python
import polars as pl, pytest
from simplex.cells import load_pairs
from simplex.barcodes import load_barcodes
from simplex._rng import rng_for
def _inp(tmp_path, n=8, locus=True):
    d={"sequence_id:0":[f"h{i}" for i in range(n)],"sequence:0":["ACGT"*90]*n,
       "sequence_id:1":[f"l{i}" for i in range(n)],"sequence:1":["TTGG"*80]*n,"name":[f"c{i}" for i in range(n)]}
    if locus: d["locus:0"]=["IGH"]*n; d["locus:1"]=["IGK"]*n
    p=tmp_path/"p.parquet"; pl.DataFrame(d).write_parquet(p); return p
def test_load(tmp_path):
    c=load_pairs(_inp(tmp_path)); assert c["chain0_locus"][0]=="IGH" and c["source_pair_id"][0]=="c0"
def test_locus_required(tmp_path):
    with pytest.raises(ValueError): load_pairs(_inp(tmp_path, locus=False))
def test_barcodes(tmp_path):
    b=load_barcodes("v2",300,rng_for(0,"bc")); assert len(set(b))==300 and all(len(x)==16 for x in b)
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement**

`simplex/barcodes.py`:
```python
import gzip
from pathlib import Path
from pairplex.utils import get_whitelist_path
def load_barcodes(chemistry, n, rng):
    p = Path(get_whitelist_path(chemistry.lower()))
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f: wl=[l.strip() for l in f if l.strip()]
    if n>len(wl): raise ValueError(f"need {n}, whitelist has {len(wl)}")
    return [wl[i] for i in rng.choice(len(wl), size=n, replace=False)]
```
`simplex/cells.py`:
```python
import numpy as np, polars as pl
from ._rng import rng_for
from .barcodes import load_barcodes

def load_pairs(input_data, n_cells=None, seed=0):
    df = pl.read_parquet(input_data)
    req = {"sequence_id:0":"chain0_id","sequence:0":"chain0_seq","sequence_id:1":"chain1_id","sequence:1":"chain1_seq"}
    miss=[k for k in req if k not in df.columns]
    if miss: raise ValueError(f"input missing {miss}")
    if "locus:0" not in df.columns or "locus:1" not in df.columns:
        raise ValueError("locus:0/1 required (Phase 1-2 will not proceed with unknown loci)")
    out = df.select([pl.col(k).alias(v) for k,v in req.items()] + [
        (pl.col("name").cast(pl.Utf8) if "name" in df.columns else pl.int_range(pl.len()).cast(pl.Utf8)).alias("source_pair_id"),
        pl.col("locus:0").cast(pl.Utf8).alias("chain0_locus"), pl.col("locus:1").cast(pl.Utf8).alias("chain1_locus")])
    # repeated source_pair_id must be consistent
    dup = out.group_by("source_pair_id").agg([pl.col("chain0_seq").n_unique().alias("a"),
        pl.col("chain1_seq").n_unique().alias("b")]).filter((pl.col("a")>1)|(pl.col("b")>1))
    if dup.height: raise ValueError(f"{dup.height} source_pair_id(s) map to differing sequences")
    if n_cells is not None:
        idx = rng_for(seed,"subsample").choice(out.height, size=n_cells, replace=n_cells>out.height); out=out[idx]
    return out.with_row_index("cell_id").select(
        ["cell_id","source_pair_id","chain0_id","chain0_seq","chain0_locus","chain1_id","chain1_seq","chain1_locus"])
```

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/barcodes.py simplex/cells.py simplex/tests/test_load.py && git commit -m "feat(simplex): locus-required load_pairs + barcode loader"`

---

### Task 5: droplets (`barcode_pool_size`) + wells + analytic collision test

**Files:** Modify `simplex/cells.py`; Create `simplex/tests/test_cells.py`.

**Produces:** `cells.assign_droplets_and_barcodes(cells, mean, sd, chemistry, barcode_pool_size, seed)`; `cells.assign_wells(cells, wells, seed)`.

- [ ] **Step 1: failing test** (incl analytic same-barcode co-occupancy)
```python
import math, polars as pl
from simplex.cells import load_pairs, assign_droplets_and_barcodes, assign_wells
def _c(tmp_path,n=600):
    d={"sequence_id:0":[f"h{i}" for i in range(n)],"sequence:0":["A"*300]*n,"sequence_id:1":[f"l{i}" for i in range(n)],
       "sequence:1":["T"*300]*n,"name":[f"c{i}" for i in range(n)],"locus:0":["IGH"]*n,"locus:1":["IGK"]*n}
    p=tmp_path/"p.parquet"; pl.DataFrame(d).write_parquet(p); return load_pairs(p)
def test_unique_barcode_per_droplet(tmp_path):
    c=assign_droplets_and_barcodes(_c(tmp_path),5,1,"v2",None,0)
    assert c.group_by("droplet_id").agg(pl.col("barcode").n_unique().alias("nb"))["nb"].max()==1
    assert c["barcode"].n_unique()==c["droplet_id"].n_unique() < c.height
def test_pool_reuse_collides(tmp_path):
    c=assign_droplets_and_barcodes(_c(tmp_path),5,1,"v2",20,0)  # tiny pool ⇒ reuse
    assert c["barcode"].n_unique() <= 20 < c["droplet_id"].n_unique()
def test_analytic_same_barcode_cooccupancy(tmp_path):
    wells=8
    c=assign_wells(assign_droplets_and_barcodes(_c(tmp_path,2000),5,1,"v2",None,0),wells,0)
    # expected co-occupant pairs ≈ sum_d C(k_d,2)/wells
    exp = sum(math.comb(k,2) for k in c.group_by("droplet_id").len()["len"].to_list())/wells
    obs = c.group_by(["resident_well","barcode"]).len().filter(pl.col("len")>=2) \
           .select((pl.col("len")*(pl.col("len")-1)//2).sum()).item() or 0
    assert 0.6*exp <= obs <= 1.6*exp
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement** (append to `simplex/cells.py`)
```python
def assign_droplets_and_barcodes(cells, mean, sd, chemistry, barcode_pool_size, seed):
    rng = rng_for(seed,"droplets"); n=cells.height; order=rng.permutation(n)
    droplet=np.empty(n,np.int64); i=d=0
    while i<n:
        for _ in range(max(1,int(round(rng.normal(mean,sd))))):
            if i>=n: break
            droplet[order[i]]=d; i+=1
        d+=1
    n_droplets=d; brng=rng_for(seed,"barcodes")
    if barcode_pool_size:
        pool=np.array(load_barcodes(chemistry, min(barcode_pool_size, n_droplets), brng))
        bc_of=pool[brng.integers(0,len(pool),size=n_droplets)]
    else:
        bc_of=np.array(load_barcodes(chemistry, n_droplets, brng))
    return cells.with_columns([pl.Series("droplet_id",droplet), pl.Series("barcode",bc_of[droplet])])

def assign_wells(cells, wells, seed):
    return cells.with_columns(pl.Series("resident_well",
        rng_for(seed,"wells").integers(0,wells,size=cells.height).astype(np.int64)))
```

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/cells.py simplex/tests/test_cells.py && git commit -m "feat(simplex): droplets (barcode_pool_size) + wells + analytic collision test"`

---

### Task 6: molecules — recovery, UMIs, resident/free, inherited RT error, **chain_status**

**Files:** Create `simplex/molecules.py`, `simplex/tests/test_molecules.py`.

**Produces:** `molecules.generate_molecules(cells, recovery_rate, molecules_per_chain_mean, release_rate, umi_length, rt_sub_rate, rt_indel_rate, seed) -> (molecules_df, chain_status_df)`. `chain_status` has `(cell_id, chain, captured, n_molecules)` for **all** cell×chain (captured=False rows too), so truth can report capture.

- [ ] **Step 1: failing test**
```python
import polars as pl
from simplex.molecules import generate_molecules
def _cells(n=1000):
    return pl.DataFrame({"cell_id":list(range(n)),"source_pair_id":[f"c{i}" for i in range(n)],
        "chain0_id":[f"h{i}" for i in range(n)],"chain0_seq":["ACGT"*80]*n,"chain0_locus":["IGH"]*n,
        "chain1_id":[f"l{i}" for i in range(n)],"chain1_seq":["TTGG"*80]*n,"chain1_locus":["IGK"]*n,
        "droplet_id":list(range(n)),"barcode":["ACGTACGTACGTACGT"]*n,"resident_well":[0]*n})
def test_chain_status_covers_all(tmp_path):
    m, cs = generate_molecules(_cells(500), 0.5, 5, 0.0, 10, 0.0, 0.0, 0)
    assert cs.height == 500*2                       # every cell×chain present, incl uncaptured
    assert 0.4 < cs["captured"].mean() < 0.6
    assert (cs.filter(~pl.col("captured"))["n_molecules"] == 0).all()
def test_release_and_rt(tmp_path):
    m,_ = generate_molecules(_cells(2000),1.0,6,0.2,10,0.2,0.0,2)
    assert 0.15 < m["is_free"].mean() < 0.25
    assert (m.filter(pl.col("chain")==0)["cdna"] != "ACGT"*80).sum() > 0
    assert (m["origin_barcode"]==m["final_barcode"]).all() and m["umi"].str.len_chars().max()==10
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement**
```python
import numpy as np, polars as pl
from ._dna import random_dna, mutate_strings
from ._rng import rng_for
def generate_molecules(cells, recovery_rate, molecules_per_chain_mean, release_rate,
                       umi_length, rt_sub_rate, rt_indel_rate, seed):
    rng = rng_for(seed,"molecules"); n=cells.height; frames=[]; status=[]
    for chain in (0,1):
        captured = rng.random(n) < recovery_rate
        nmol = np.where(captured, np.maximum(rng.poisson(molecules_per_chain_mean,n),1), 0).astype(np.int64)
        status.append(pl.DataFrame({"cell_id":cells["cell_id"], "chain":np.full(n,chain,np.int8),
                                    "captured":captured, "n_molecules":nmol}))
        rep=np.repeat(np.arange(n),nmol)
        if rep.size==0: continue
        sub=cells[rep]; k=rep.size; cdna=list(sub[f"chain{chain}_seq"])
        if rt_sub_rate>0 or rt_indel_rate>0:
            cdna,_=mutate_strings(cdna, rt_sub_rate, rt_indel_rate, rng_for(seed,f"rt{chain}"))
        bc=sub["barcode"].to_numpy().astype(str)
        frames.append(pl.DataFrame({"origin_cell_id":sub["cell_id"],"origin_droplet_id":sub["droplet_id"],
            "source_pair_id":sub["source_pair_id"],"chain":np.full(k,chain,np.int8),"locus":sub[f"chain{chain}_locus"],
            "umi":random_dna(rng,k,umi_length),"origin_barcode":bc,"final_barcode":bc,
            "resident_well":sub["resident_well"],"is_free":rng.random(k)<release_rate,"cdna":cdna}))
    mols=pl.concat(frames).with_row_index("molecule_id").with_columns([
        pl.col("molecule_id").cast(pl.Int64), pl.col("molecule_id").cast(pl.Int64).alias("parent_molecule_id")])
    return mols, pl.concat(status)
```

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/molecules.py simplex/tests/test_molecules.py && git commit -m "feat(simplex): molecules + chain_status (capture preserved), free split, RT error"`

---

### Task 7: routing — survival flag kept, redistribution, amplification, index hop

**Files:** Create `simplex/routing.py`, `simplex/tests/test_routing.py`.

**Produces:** `routing.route_and_amplify(molecules, wells, molecule_survival_rate, reads_per_molecule_mean, index_hop_rate, seed) -> (molecules_with_survival, reads)`. Keeps **all** molecules (adds `amplification_well`, `survived`); expands only survivors into `reads` (retains `molecule_id`).

- [ ] **Step 1: failing test**
```python
import numpy as np, polars as pl
from simplex.routing import route_and_amplify
def _mol(n=2000):
    rng=np.random.default_rng(0)
    return pl.DataFrame({"molecule_id":list(range(n)),"parent_molecule_id":list(range(n)),
        "origin_cell_id":rng.integers(0,500,n),"origin_droplet_id":rng.integers(0,300,n),
        "source_pair_id":[f"c{i%500}" for i in range(n)],"chain":rng.integers(0,2,n).astype(np.int8),
        "locus":["IGH"]*n,"umi":["AAAAAAAAAA"]*n,"origin_barcode":["BC"]*n,"final_barcode":["BC"]*n,
        "resident_well":rng.integers(0,4,n).astype(np.int64),"is_free":rng.random(n)<0.2,"cdna":["ACGT"*50]*n})
def test_all_molecules_kept_with_survival(tmp_path):
    mols, reads = route_and_amplify(_mol(),4,0.5,3,0.0,0)
    assert mols.height==2000 and "survived" in mols.columns
    assert reads["molecule_id"].n_unique() == mols.filter(pl.col("survived")).height
def test_free_keeps_barcode_umi(tmp_path):
    _, reads = route_and_amplify(_mol(),4,1.0,3,0.0,0)
    assert (reads["barcode"]=="BC").all() and (reads["umi"]=="AAAAAAAAAA").all()
def test_family_shares_umi_and_index_hop(tmp_path):
    _, reads = route_and_amplify(_mol(),4,1.0,4,0.2,0)
    assert reads.group_by("molecule_id").agg(pl.col("umi").n_unique().alias("u"))["u"].max()==1
    h=reads.filter(pl.col("is_index_hopped")); assert (h["final_well"]!=h["amplification_well"]).all()
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement**
```python
import numpy as np, polars as pl
from ._rng import rng_for
def route_and_amplify(molecules, wells, molecule_survival_rate, reads_per_molecule_mean, index_hop_rate, seed):
    rng=rng_for(seed,"routing"); n=molecules.height
    free=molecules["is_free"].to_numpy()
    amp=np.where(free, rng.integers(0,wells,size=n), molecules["resident_well"].to_numpy()).astype(np.int64)
    survived=rng.random(n) < molecule_survival_rate
    mols=molecules.with_columns([pl.Series("amplification_well",amp), pl.Series("survived",survived)])
    surv=mols.filter(pl.col("survived"))
    depth=np.maximum(rng.poisson(reads_per_molecule_mean, surv.height),1).astype(np.int64)
    rep=np.repeat(np.arange(surv.height), depth); reads=surv[rep]; k=reads.height
    hop=rng.random(k) < index_hop_rate
    off=rng.integers(1,max(2,wells),size=k); a=reads["amplification_well"].to_numpy()
    final=np.where(hop, (a+off)%wells, a).astype(np.int64)
    reads=reads.with_columns([pl.Series("read_id",[f"r{i}" for i in range(k)]),
        pl.col("final_barcode").alias("barcode"), pl.Series("final_well",final),
        pl.Series("is_index_hopped",hop), pl.lit(False).alias("is_barcode_swapped"),
        pl.lit(0,pl.Int64).alias("n_seq_errors")]).select(
        ["read_id","molecule_id","origin_cell_id","source_pair_id","chain","locus","umi","barcode",
         "amplification_well","final_well","is_free","is_index_hopped","is_barcode_swapped","cdna","n_seq_errors"])
    return mols, reads
```

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/routing.py simplex/tests/test_routing.py && git commit -m "feat(simplex): routing — survival kept, redistribution, amplification, index hop"`

---

### Task 8: sequencing errors + merged reads (round-trip)

**Files:** Create `simplex/reads.py`, `simplex/tests/test_reads.py`.

**Produces:** `reads.apply_sequencing_errors(reads, sub_rate, indel_rate, seed)`; `reads.build_merged(reads, tso, rc_fraction, variable_length, seed)`.

- [ ] **Step 1–4:** (same as the prior plan's Task 6 — unchanged; test asserts `s[:16]==barcode`, `s[16:26]==umi`, `s[36:].lstrip("G")` recovers cDNA, quality length; rc_fraction parses via revcomp). Implementation identical to the archived plan's `simplex/reads.py`.

- [ ] **Step 5: commit** `git add simplex/reads.py simplex/tests/test_reads.py && git commit -m "feat(simplex): sequencing errors + merged read assembly"`

*(Full code: see archived plan Task 6; behavior unchanged. Reproduce `apply_sequencing_errors` via `_dna.mutate_strings` and `build_merged` via `pl.concat_str` + `revcomp_expr` + `str.replace_all(".", "I")`.)*

---

### Task 9: truth — components, cells (capture/survival), barcodes (occupancy from cells)

**Files:** Create `simplex/truth.py`, `simplex/tests/test_truth.py`.

**Produces:** `truth.build_truth_components(cells, reads)`; `truth.build_truth_cells(cells, chain_status, molecules, reads)`; `truth.build_truth_barcodes(cells, molecules, reads, components)`.

- [ ] **Step 1: failing test** (key points: cells built from chain_status/molecules so `captured`/`survived`/`n_molecules` exist; barcodes occupancy from `cells` so a read-less resident still counts)
```python
import polars as pl
from simplex.truth import build_truth_components, build_truth_cells, build_truth_barcodes
def _cells():
    return pl.DataFrame({"cell_id":[0,1],"source_pair_id":["A","B"],
        "chain0_id":["hA","hB"],"chain0_seq":["HA","HB"],"chain0_locus":["IGH","IGH"],
        "chain1_id":["lA","lB"],"chain1_seq":["LA","LB"],"chain1_locus":["IGK","IGK"],
        "droplet_id":[0,0],"barcode":["X","X"],"resident_well":[0,0]})   # A,B collide on X in well 0
def _status():
    return pl.DataFrame({"cell_id":[0,0,1,1],"chain":[0,1,0,1],
        "captured":[True,True,True,False],"n_molecules":[2,2,1,0]})
def _mols():
    return pl.DataFrame({"molecule_id":[0,1,2],"origin_cell_id":[0,0,1],"chain":[0,1,0],
        "survived":[True,True,True]})
def _reads():  # cell1 produced NO reads (all lost) but is physically resident at X
    return pl.DataFrame({"read_id":["r0","r1"],"molecule_id":[0,1],"origin_cell_id":[0,0],
        "source_pair_id":["A","A"],"chain":[0,1],"locus":["IGH","IGK"],"barcode":["X","X"],
        "final_well":[0,0],"is_free":[False,False],"is_index_hopped":[False,False],"umi":["u0","u1"]})
def test_barcodes_occupancy_from_cells():
    comp=build_truth_components(_cells(),_reads())
    tb=build_truth_barcodes(_cells(),_mols(),_reads(),comp)
    x=tb.filter((pl.col("well")==0)&(pl.col("barcode")=="X")).to_dicts()[0]
    assert x["n_resident_cells"]==2 and x["is_collision"] is True   # both A and B counted, even read-less B
def test_cells_have_capture():
    tc=build_truth_cells(_cells(),_status(),_mols(),_reads())
    assert "captured_0" in tc.columns and "survived_0" in tc.columns and "n_molecules_0" in tc.columns
    row=tc.filter(pl.col("cell_id")==1).to_dicts()[0]
    assert row["captured_1"] is False
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement**
```python
import polars as pl

def _cell_chain_seq(cells):
    parts=[]
    for ch in (0,1):
        parts.append(cells.select([pl.col("cell_id").alias("origin_cell_id"),
            pl.lit(ch).cast(pl.Int8).alias("chain"), pl.col(f"chain{ch}_seq").alias("sequence"),
            pl.col(f"chain{ch}_locus").alias("locus"), pl.col("resident_well"),
            pl.col("barcode").alias("home_barcode")]))
    return pl.concat(parts)

def build_truth_components(cells, reads):
    cs=_cell_chain_seq(cells)
    agg=reads.group_by(["final_well","barcode","origin_cell_id","chain"]).agg([
        pl.col("source_pair_id").first(), pl.col("locus").first(), pl.len().alias("n_reads"),
        (~pl.col("is_free")&~pl.col("is_index_hopped")).sum().alias("n_reads_resident"),
        pl.col("is_free").sum().alias("n_reads_free"), pl.col("is_index_hopped").sum().alias("n_reads_index_hopped"),
        pl.col("umi").n_unique().alias("n_umis"),
        (pl.col("molecule_id").n_unique() if "molecule_id" in reads.columns else pl.col("umi").n_unique()).alias("n_source_molecules")])
    comp=agg.join(cs.select(["origin_cell_id","chain","sequence","resident_well","home_barcode"]),
                  on=["origin_cell_id","chain"], how="left")
    return comp.with_columns(((pl.col("resident_well")==pl.col("final_well"))&
        (pl.col("home_barcode")==pl.col("barcode"))).alias("is_resident_source")).drop(["resident_well","home_barcode"])

def build_truth_cells(cells, chain_status, molecules, reads):
    surv=(molecules.filter(pl.col("survived")).group_by(["origin_cell_id","chain"]).len()
          .rename({"origin_cell_id":"cell_id","len":"survived_n"}))
    rc=reads.group_by(["origin_cell_id","chain"]).agg([
        pl.len().alias("n_reads_generated"), (~pl.col("is_free")).sum().alias("n_reads_resident"),
        pl.col("is_free").sum().alias("n_reads_free_out"), pl.col("is_index_hopped").sum().alias("n_reads_index_hopped_out"),
        pl.col("umi").n_unique().alias("n_umis")]).rename({"origin_cell_id":"cell_id"})
    st=(chain_status.join(surv, on=["cell_id","chain"], how="left")
        .with_columns((pl.col("survived_n").fill_null(0)>0).alias("survived"))
        .join(rc, on=["cell_id","chain"], how="left").fill_null(0))
    wide=st.pivot(index="cell_id", on="chain",
        values=["captured","survived","n_molecules","n_umis","n_reads_generated",
                "n_reads_resident","n_reads_free_out","n_reads_index_hopped_out"])
    return cells.join(wide, on="cell_id", how="left")

def build_truth_barcodes(cells, molecules, reads, components):
    physical=cells.select([pl.col("resident_well").alias("well"), pl.col("barcode"),
                           pl.col("cell_id"), pl.col("source_pair_id")])
    occ=physical.group_by(["well","barcode"]).agg([
        pl.col("source_pair_id").unique().alias("resident_source_ids"),
        pl.col("cell_id").n_unique().alias("n_resident_cells")])
    observed=reads.select([pl.col("final_well").alias("well"), pl.col("barcode")]).unique()
    keys=pl.concat([occ.select(["well","barcode"]), observed]).unique()
    tb=keys.join(occ, on=["well","barcode"], how="left")
    # per-locus dominance by reads and umis among ALL sources at the (final_well,barcode) key
    def dom(loci, by, alias):
        f=components.filter(pl.col("locus").is_in(loci)).sort(by, descending=True)
        return (f.group_by([pl.col("final_well").alias("well"),"barcode"])
                 .agg(pl.col("source_pair_id").first().alias(alias)))
    for loci,name in [(["IGH"],"heavy"),(["IGK","IGL"],"light")]:
        tb=tb.join(dom(loci,"n_reads",f"dominant_{name}_source_by_reads"), on=["well","barcode"], how="left")
        tb=tb.join(dom(loci,"n_umis",f"dominant_{name}_source_by_umis"), on=["well","barcode"], how="left")
    return tb.with_columns([pl.col("n_resident_cells").fill_null(0),
        (pl.col("n_resident_cells").fill_null(0)>=2).alias("is_collision"),
        (pl.col("n_resident_cells").fill_null(0)==0).alias("is_ambient_only")])
```
> Collision counts (`n_captured_both/survived_both/sequenced_both/reference_pairable_resident_cells`) join from `truth_cells` per key — add in the same builder by joining `truth_cells` capture/survival/read columns onto `physical` and aggregating booleans per `(well,barcode)`. Test in Step 1 covers occupancy + capture; extend with a collision-count assertion.

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/truth.py simplex/tests/test_truth.py && git commit -m "feat(simplex): truth — capture/survival preserved, occupancy from cells, collision counts"`

---

### Task 10: IO writers + run() (guards, manifest)

**Files:** Create `simplex/io.py`, `simplex/run.py`, `simplex/tests/test_run.py`.

**Produces:** `io.write_merged_fastq(built, output_dir, compress=True)` (in-memory per-well writer); `io.write_truth(...)`; `run.run(input_data, output_directory, **knobs) -> Path`.

- [ ] **Step 1: failing test** (outputs exist; reproducible; **fails on non-empty output dir**; manifest has versions/fingerprint/counts)
```python
import gzip
from pathlib import Path
import polars as pl, pytest
from simplex.run import run
def _inp(tmp_path,n=60):
    d={"sequence_id:0":[f"h{i}" for i in range(n)],"sequence:0":["GATTACA"*30]*n,
       "sequence_id:1":[f"l{i}" for i in range(n)],"sequence:1":["CCGGTA"*30]*n,
       "name":[f"c{i}" for i in range(n)],"locus:0":["IGH"]*n,"locus:1":["IGK"]*n}
    p=tmp_path/"in.parquet"; pl.DataFrame(d).write_parquet(p); return p
def test_outputs_and_manifest(tmp_path):
    out=tmp_path/"o"; rd=run(input_data=_inp(tmp_path),output_directory=out,wells=4,
        cells_per_droplet_mean=1,cells_per_droplet_sd=0,variable_length=False,seed=0)
    assert list(Path(rd).glob("*.fastq.gz"))
    for f in ["truth_components","truth_cells","truth_barcodes"]:
        assert (out/"truth"/f"{f}.parquet").exists()
    man=__import__("json").loads((out/"run_manifest.json").read_text())
    assert "input_fingerprint" in man and "n_reads" in man
def test_refuses_nonempty_dir(tmp_path):
    out=tmp_path/"o"; run(input_data=_inp(tmp_path),output_directory=out,wells=4,seed=0)
    with pytest.raises(FileExistsError):
        run(input_data=_inp(tmp_path),output_directory=out,wells=4,seed=0)
def test_reproducible(tmp_path):
    def content(d): return sorted(gzip.open(p,"rt").read() for p in Path(d).glob("*.fastq.gz"))
    a=run(input_data=_inp(tmp_path),output_directory=tmp_path/"a",wells=4,seed=5)
    b=run(input_data=_inp(tmp_path),output_directory=tmp_path/"b",wells=4,seed=5)
    assert content(a)==content(b)
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement**

`simplex/io.py`:
```python
import gzip
from pathlib import Path
def _tag(w): return f"well{int(w):03d}"
def write_merged_fastq(built, output_dir, compress=True):
    rd=Path(output_dir)/"reads"; rd.mkdir(parents=True, exist_ok=True)
    ext="fastq.gz" if compress else "fastq"; op=(lambda p: gzip.open(p,"wt")) if compress else (lambda p: open(p,"w"))
    paths=[]
    for (well,),sub in built.group_by(["final_well"], maintain_order=True):
        p=rd/f"{_tag(well)}.{ext}"
        with op(p) as fh:
            fh.write("".join(f"@{i}\n{s}\n+\n{q}\n" for i,s,q in zip(sub["read_id"],sub["read_seq"],sub["qual"])))
        paths.append(p)
    return paths
def write_truth(output_dir, comp, cells, barcodes, reads=None):
    td=Path(output_dir)/"truth"; td.mkdir(parents=True, exist_ok=True)
    comp.write_parquet(td/"truth_components.parquet"); cells.write_parquet(td/"truth_cells.parquet")
    barcodes.write_parquet(td/"truth_barcodes.parquet")
    if reads is not None: reads.write_parquet(td/"truth_reads.parquet")
```
`simplex/run.py`:
```python
import hashlib, json
from pathlib import Path
from .cells import load_pairs, assign_droplets_and_barcodes, assign_wells
from .config import SimplexConfig
from .molecules import generate_molecules
from .routing import route_and_amplify
from .reads import apply_sequencing_errors, build_merged
from .truth import build_truth_components, build_truth_cells, build_truth_barcodes
from .io import write_merged_fastq, write_truth
try:
    from .version import __version__ as _SV
except Exception:
    _SV = "0.0.0"

def run(input_data, output_directory, **knobs):
    cfg=SimplexConfig(input_data=str(input_data), output_directory=str(output_directory), **knobs)
    out=Path(output_directory)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output dir {out} not empty; refusing to overwrite an experiment")
    out.mkdir(parents=True, exist_ok=True)
    cells=load_pairs(cfg.input_data, cfg.n_cells, cfg.seed)
    cfg.validate(actual_n_cells=cells.height)
    cells=assign_droplets_and_barcodes(cells, cfg.cells_per_droplet_mean, cfg.cells_per_droplet_sd,
                                       cfg.chemistry, cfg.barcode_pool_size, cfg.seed)
    cells=assign_wells(cells, cfg.wells, cfg.seed)
    mols, chain_status=generate_molecules(cells, cfg.recovery_rate, cfg.molecules_per_chain_mean,
        cfg.release_rate, cfg.umi_length, cfg.rt_sub_rate, cfg.rt_indel_rate, cfg.seed)
    mols, reads=route_and_amplify(mols, cfg.wells, cfg.molecule_survival_rate,
        cfg.reads_per_molecule_mean, cfg.index_hop_rate, cfg.seed)
    reads=apply_sequencing_errors(reads, cfg.sequencing_sub_rate, cfg.sequencing_indel_rate, cfg.seed)
    comp=build_truth_components(cells, reads)
    tcells=build_truth_cells(cells, chain_status, mols, reads)
    tbar=build_truth_barcodes(cells, mols, reads, comp)
    built=build_merged(reads, cfg.tso, cfg.rc_fraction, cfg.variable_length, cfg.seed)
    write_merged_fastq(built, out)
    write_truth(out, comp, tcells, tbar, reads if cfg.write_read_truth else None)
    cfg.to_json(out/"simplex_config.json")
    fp=hashlib.blake2b(Path(cfg.input_data).read_bytes(), digest_size=16).hexdigest()
    (out/"run_manifest.json").write_text(json.dumps({
        "simplex_version":_SV, "seed":cfg.seed, "n_cells":cells.height, "wells":cfg.wells,
        "input_fingerprint":fp, "config_hash":hashlib.blake2b(json.dumps(cfg.to_dict(),sort_keys=True).encode(),digest_size=16).hexdigest(),
        "rng_scheme":"per-stage blake2b (v1: order-dependent)", "n_reads":reads.height,
        "n_molecules":mols.height}, indent=2))
    return out/"reads"
```
> PairPlex version can be added to the manifest via `pairplex.__version__` when packaged.

- [ ] **Step 4: run → PASS.**   - [ ] **Step 5: commit** `git add simplex/io.py simplex/run.py simplex/tests/test_run.py && git commit -m "feat(simplex): writers + run() with output-dir guard and manifest"`

---

### Task 11: controlled deterministic fixtures + clean golden

**Files:** Create `simplex/_fixtures.py` (test helpers that build exact low-level tables + write FASTQ), `simplex/tests/test_mechanism.py`.

**Produces:** `_fixtures.reads_to_fastq(reads_df, output_dir, tso)` (build+write merged), and helpers to hand-construct `cells/reads` for a scenario. Downstream `pairplex.run` + `simplex.score` run free.

- [ ] **Step 1: write fixtures** — implement all seven (0 clean golden + 1–6). Each **forces** the condition via explicit tables, then asserts on `score()`. Sketch for the two hardest:

```python
# _fixtures.py
import polars as pl
from .reads import build_merged
from .io import write_merged_fastq
from .truth import build_truth_components, build_truth_cells, build_truth_barcodes
from .io import write_truth

def emit(cells, chain_status, molecules, reads, out, tso="TTTCTTATATGGG"):
    built = build_merged(reads, tso, 0.0, False, 0)
    write_merged_fastq(built, out)
    write_truth(out, build_truth_components(cells, reads),
                build_truth_cells(cells, chain_status, molecules, reads),
                build_truth_barcodes(cells, molecules, reads, build_truth_components(cells, reads)))
    return out/"reads"
```

```python
# test_mechanism.py (exact ambient mispair; wells>=2, forced routing)
import polars as pl, pairplex
from simplex._fixtures import emit
from simplex.scoring import score
from simplex._testseqs import HEAVY_A, LIGHT_A, HEAVY_B, LIGHT_B  # real abstar bnAb seqs, distinct

def test_exact_ambient_mispair(tmp_path):
    # A resident at well0/bcX with heavy only; B's LIGHT is FREE and routed to well0/bcX
    cells = pl.DataFrame({"cell_id":[0,1],"source_pair_id":["A","B"],
        "chain0_id":["hA","hB"],"chain0_seq":[HEAVY_A,HEAVY_B],"chain0_locus":["IGH","IGH"],
        "chain1_id":["lA","lB"],"chain1_seq":[LIGHT_A,LIGHT_B],"chain1_locus":["IGK","IGK"],
        "droplet_id":[0,0],"barcode":["X","X"],"resident_well":[0,1]})
    chain_status = pl.DataFrame({"cell_id":[0,0,1,1],"chain":[0,1,0,1],
        "captured":[True,False,True,True],"n_molecules":[4,0,4,4]})
    molecules = pl.DataFrame({"molecule_id":[0,1],"origin_cell_id":[0,1],"chain":[0,1],"survived":[True,True]})
    def fam(mid, cell, spid, chain, locus, seq, well, is_free):
        return pl.DataFrame({"read_id":[f"{mid}_{j}" for j in range(4)],"molecule_id":[mid]*4,
            "origin_cell_id":[cell]*4,"source_pair_id":[spid]*4,"chain":[chain]*4,"locus":[locus]*4,
            "umi":[f"UMI{mid}"]*4,"barcode":["X"]*4,"amplification_well":[well]*4,"final_well":[well]*4,
            "is_free":[is_free]*4,"is_index_hopped":[False]*4,"is_barcode_swapped":[False]*4,
            "cdna":[seq]*4,"n_seq_errors":[0]*4})
    reads = pl.concat([fam(0,0,"A",0,"IGH",HEAVY_A,0,False),      # A heavy resident at well0/X
                       fam(1,1,"B",1,"IGK",LIGHT_B,0,True)])       # B light FREE -> well0/X
    rd = emit(cells, chain_status, molecules, reads, tmp_path/"sim")
    ppo = tmp_path/"pp"; pairplex.run(sequences=str(rd), output_directory=str(ppo),
        min_cluster_reads=3, min_cluster_umis=1, quiet=True)
    ps, _ = score(ppo, (tmp_path/"sim"/"truth"))
    assert (ps["pairing_status"] == "mispaired").sum() >= 1   # A_H + B_L emitted at X
```
Implement the remaining: **clean golden** (1 cell/bc, no release/errors → all `resident_correct`, no `ambiguous`/`unmatchable`), **one-cell negative control**, **same-well collision** (both A,B `resident_well=0`, A-light & B-heavy absent), **route composition** (one read of a molecule forced `final_well != amplification_well`; assert via `write_read_truth`), **joint ambiguity** (two source pairs share `HEAVY_A`; distinct lights; correct pair still `correct`), **missing output** (resident pair present but an extra contaminant contig at the key makes PairPlex reject → `key_scores.output_status=="missing"`). Add `simplex/_testseqs.py` loading four distinct abstar bnAb H/L sequences at import.

- [ ] **Step 2–4:** Run `python -m pytest simplex/tests/test_mechanism.py -q`; iterate on generator/scorer if a fixture exposes a mechanism bug (that is their purpose).

- [ ] **Step 5: commit** `git add simplex/_fixtures.py simplex/_testseqs.py simplex/tests/test_mechanism.py && git commit -m "test(simplex): controlled deterministic mechanism fixtures + clean golden"`

---

### Task 12: statistical single-factor tests

**Files:** Create `simplex/tests/test_single_factor.py`. Assert **tradeoff directions / mechanistic stats**, not blanket monotonicity.

- [ ] **Step 1: write tests** — one generated contaminated dataset, scored under two PairPlex settings:
```python
import pairplex, polars as pl
from simplex.run import run
from simplex.scoring import score
from simplex._testseqs import many_pairs_parquet   # writes an N-pair input parquet

def _score_dir(ppo, truth):
    ps, ks = score(ppo, truth)
    mis = (ps["pairing_status"]=="mispaired").sum()
    recall = (ks["output_status"]=="unique").sum() / max(1,(ks["reference_pairable_both"]).sum())
    return mis, recall

def test_fraction_filter_trades_precision_for_yield(tmp_path):
    inp = many_pairs_parquet(tmp_path, n=60)
    out = tmp_path/"sim"
    rd = run(input_data=inp, output_directory=out, wells=2, cells_per_droplet_mean=2,
             cells_per_droplet_sd=0, recovery_rate=0.6, release_rate=0.15,
             molecule_survival_rate=1.0, index_hop_rate=0.0, sequencing_sub_rate=0.0,
             variable_length=False, seed=1)
    lo = tmp_path/"lo"; pairplex.run(sequences=str(rd), output_directory=str(lo),
             min_cluster_reads=3, min_cluster_umis=1, min_cluster_fraction=0.0, quiet=True)
    hi = tmp_path/"hi"; pairplex.run(sequences=str(rd), output_directory=str(hi),
             min_cluster_reads=3, min_cluster_umis=1, min_cluster_fraction=0.3, quiet=True)
    mis_lo, rec_lo = _score_dir(lo, out/"truth")
    mis_hi, rec_hi = _score_dir(hi, out/"truth")
    assert mis_hi <= mis_lo and rec_hi <= rec_lo   # fraction filter: fewer mispairs, not more recall
```

- [ ] **Step 2–4:** Run; iterate until the tradeoff assertions hold.   - [ ] **Step 5: commit** `git add simplex/tests/test_single_factor.py && git commit -m "test(simplex): single-factor precision/yield tradeoff tests"`

---

### Task 13 (optional, Phase 0A): real-data marginal audit

**Files:** Create `simplex/audit.py`, `simplex/tests/test_audit.py`.

**Produces:** `audit.audit_metadata(metadata_csv_glob, report_path) -> pl.DataFrame` — marginal summaries only (reads/UMIs/cluster_fraction quantiles; per-`(well,barcode)` contig-count profile → 1H+1L/1H+2L/2H+1L frequencies), dataset-agnostic, **no calibration gate**, report header states the no-labeled-truth limitation.

- [ ] **Step 1–5:** TDD against a synthetic `metadata/*.csv` fixture (columns `name, reads, umis, cluster_fraction, pass_filters`); assert the summary has the marginal rows and the caveat line. Commit `feat(simplex): optional Phase 0A marginal audit`.

---

## Self-Review

**Spec coverage (v3 → tasks):** keyed-RNG/config guards (§10,§9)→T1; bounded matcher + joint resolver (§6)→T2; scorer contract joint/orthogonal/key_scores/per-cell observability (§6)→T3; locus-required load (§9)→T4; droplets+`barcode_pool_size`+analytic collision (§4,§12)→T5; molecules **+chain_status** capture preserved (§5,§7)→T6; routing keeps survivors flagged, free redistribution, index-hop guard (§4)→T7; seq error + merged (§7,§8)→T8; truth capture/survival + occupancy-from-cells + collision counts + per-locus dominance by reads&umis (§5)→T9; writers/run guards+manifest (§9)→T10; controlled fixtures + clean golden (§12)→T11; single-factor tradeoff (§12)→T12; optional audit (§2)→T13.

**Three non-negotiables:** (1) capture/survival truth preserved via `chain_status`+kept molecules, occupancy from `cells` — T6,T7,T9. (2) scorer rewritten to its orthogonal spec (separate `pairing_status`/`source_resolution`, disjoint⇒mispaired, `origin` categories, `key_status=unknown`, per-cell observability) and scores **all** wells jointly — T2,T3. (3) fixtures are truly controlled via `_fixtures.emit` on hand-built tables; `wells≥2` for ambient — T11.

**Placeholder scan:** T8 defers to the archived plan's identical code (explicitly, with the construction named) rather than a silent TODO. T11 gives full code for the two hardest fixtures and exact specifications for the other five plus `_testseqs`. T9 notes the collision-count join to finish in-builder. No `...` left in executable steps.

**Type consistency:** schemas block is the contract; `chain_status`/`survived`/`molecule_id` flow T6→T7→T9; scorer axes stable T2→T3→T11/T12; `score()` takes a dir/list everywhere.

**Known execution risks (watch):** polars `pivot`/`group_by` tuple-unpacking on 1.39; ensure `route_and_amplify` keeps `molecule_id` (it does); edlib installed (pairplex dep); `pairplex.run` merged-mode on synthetic wells (used throughout, no fastp dependency).
