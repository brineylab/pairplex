# SimPlex Phase 0–2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the mechanistically-faithful SimPlex generator (Phase 1) + compact ground truth + a `(final_well, barcode)`-keyed scorer (Phase 0B/2), plus an optional real-data audit (Phase 0A), so we can drive PairPlex on synthetic data with known truth and measure precision/yield.

**Architecture:** Staged, vectorized, per-well-partitionable pipeline (polars + numpy) with keyed RNG. Molecules carry barcode+UMI; some are *released/free* and redistributed across wells **before** amplification; read families inherit molecule-level RT error; index hopping moves reads post-amplification. Truth is a compact `truth_components` table. The scorer resolves chains jointly, locus-restricted, key-local, over candidate `source_pair_id` sets. See spec: `docs/superpowers/specs/2026-08-27-simplex-generator-design.md` (v3).

**Tech Stack:** Python 3.10+, polars 1.39, numpy 2.x, pytest. Reuses `pairplex` whitelists + read structure (`barcode=s[:16]`, `umi=s[16:26]`, `sequence=s[36:].lstrip("G")`).

## Global Constraints

- Sibling package `simplex/` (NOT under `pairplex/`); `import simplex`.
- Merged read layout: `barcode(16)+umi(10)+TSO("TTTCTTATATGGG")+cDNA`; must round-trip through `pairplex.parse_barcodes`. **`output_mode="merged"` is the v1 default** (paired = Phase 3).
- **Keyed RNG only** (§Task 1): `rng_for(seed, stage, well, chunk)` via `SeedSequence(blake2b(...))`. Never `seed+offset`, never Python `hash()`. Results independent of chunk size/order.
- Scorer correctness matched by **sequence**, never `junction_aa`. Candidate matches are **sets** of `source_pair_id`, **locus-restricted**, **restricted to sources present at the key**, resolved **jointly**.
- Ambient = free molecules redistributed across wells **retaining barcode+UMI**, **pre-amplification**. Barcode-changing (`barcode_swap`) is deferred. Mispairs require `cells_per_droplet>1` (+ resident-chain absence) OR same-well collision.
- v1 scale ≤~50k cells, in-memory, but every stage partitions by `final_well`.
- Commit after each task; tests in `simplex/tests/`.

## Canonical stage schemas (contract across tasks)

```
cells:      cell_id:Int64, source_pair_id:Utf8, chain0_id:Utf8, chain0_seq:Utf8, chain0_locus:Utf8,
            chain1_id:Utf8, chain1_seq:Utf8, chain1_locus:Utf8
   +droplets: droplet_id:Int64, barcode:Utf8
   +wells:    resident_well:Int64

molecules:  molecule_id:Int64, parent_molecule_id:Int64, origin_cell_id:Int64, origin_droplet_id:Int64,
            source_pair_id:Utf8, chain:Int8, locus:Utf8, umi:Utf8, origin_barcode:Utf8, final_barcode:Utf8,
            resident_well:Int64, amplification_well:Int64, is_free:Bool, survived:Bool, cdna:Utf8

reads:      read_id:Utf8, molecule_id:Int64, origin_cell_id:Int64, source_pair_id:Utf8, chain:Int8,
            locus:Utf8, umi:Utf8, barcode:Utf8, amplification_well:Int64, final_well:Int64,
            is_free:Bool, is_index_hopped:Bool, is_barcode_swapped:Bool, cdna:Utf8, n_seq_errors:Int64

built(merged): read_id:Utf8, final_well:Int64, read_seq:Utf8, qual:Utf8
```

Truth schemas are defined in Task 7; scorer output schemas in Task 10.

---

### Task 1: Scaffold — keyed RNG, DNA helpers, SimplexConfig + validation

**Files:** Create `simplex/__init__.py`, `simplex/_rng.py`, `simplex/_dna.py`, `simplex/config.py`, `simplex/tests/__init__.py`, `simplex/tests/test_rng.py`, `simplex/tests/test_dna.py`, `simplex/tests/test_config.py`. Modify `pyproject.toml` (add `simplex*` to packages).

**Interfaces (Produces):**
- `_rng.rng_for(seed:int, stage:str, well:int=0, chunk:int=0) -> np.random.Generator`
- `_dna.random_dna(rng, k, length) -> np.ndarray[str]`; `_dna.revcomp_expr(col) -> pl.Expr`; `_dna.revcomp_str(s) -> str`; `_dna.mutate_strings(seqs, sub_rate, indel_rate, rng) -> (list[str], np.ndarray[int])`
- `config.SimplexConfig` (v2 API from spec §9) with `.to_dict()`, `.to_json(path)`, `.validate()`, `.estimated_reads()`

- [ ] **Step 1: Write failing tests**

`simplex/tests/test_rng.py`:
```python
from simplex._rng import rng_for
def test_same_key_same_stream():
    a = rng_for(0, "molecules", well=3, chunk=1).integers(0, 1_000_000, 50)
    b = rng_for(0, "molecules", well=3, chunk=1).integers(0, 1_000_000, 50)
    assert list(a) == list(b)
def test_different_key_different_stream():
    a = rng_for(0, "molecules", well=3).integers(0, 1_000_000, 50)
    b = rng_for(0, "molecules", well=4).integers(0, 1_000_000, 50)
    assert list(a) != list(b)
def test_order_independent():
    # well 4 stream identical whether or not well 3 was drawn first
    first = rng_for(0, "s", well=4).integers(0, 99, 10)
    _ = rng_for(0, "s", well=3).integers(0, 99, 10)
    second = rng_for(0, "s", well=4).integers(0, 99, 10)
    assert list(first) == list(second)
```
`simplex/tests/test_dna.py`:
```python
import numpy as np, polars as pl
from simplex._dna import random_dna, revcomp_expr, revcomp_str, mutate_strings
def test_random_dna():
    out = random_dna(np.random.default_rng(0), 5, 10)
    assert len(out) == 5 and all(len(s) == 10 for s in out) and set("".join(out)) <= set("ACGT")
def test_revcomp():
    assert revcomp_str("AAACCTGGN") == "NCCAGGTTT"
    assert pl.DataFrame({"s": ["ACGT"]}).select(revcomp_expr("s"))["s"][0] == "ACGT"
def test_mutate_rate():
    seqs = ["ACGT" * 50] * 500
    out, ne = mutate_strings(seqs, 0.05, 0.0, np.random.default_rng(0))
    assert 0.03 * 200 < ne.mean() < 0.07 * 200
def test_mutate_zero():
    out, ne = mutate_strings(["ACGT"], 0.0, 0.0, np.random.default_rng(0))
    assert out == ["ACGT"] and ne.sum() == 0
```
`simplex/tests/test_config.py`:
```python
import pytest
from simplex.config import SimplexConfig
def test_defaults_and_json(tmp_path):
    c = SimplexConfig(input_data="x.parquet", output_directory="o")
    assert c.output_mode == "merged" and c.molecule_survival_rate == 0.8
    c.to_json(tmp_path / "c.json"); assert (tmp_path / "c.json").exists()
def test_validate_rejects_bad_rate():
    with pytest.raises(ValueError):
        SimplexConfig(input_data="x", output_directory="o", release_rate=1.5).validate()
def test_oom_guard():
    c = SimplexConfig(input_data="x", output_directory="o", n_cells=10_000_000,
                      reads_per_molecule_mean=50, molecules_per_chain_mean=50)
    with pytest.raises(ValueError):
        c.validate(max_reads=5_000_000_000)
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest simplex/tests/test_rng.py simplex/tests/test_dna.py simplex/tests/test_config.py -q`
Expected: FAIL (module `simplex` not found).

- [ ] **Step 3: Implement**

`simplex/_rng.py`:
```python
import hashlib
import numpy as np

def rng_for(seed: int, stage: str, well: int = 0, chunk: int = 0) -> np.random.Generator:
    key = f"{seed}|{stage}|{well}|{chunk}".encode()
    entropy = int.from_bytes(hashlib.blake2b(key, digest_size=16).digest(), "big")
    return np.random.default_rng(np.random.SeedSequence(entropy))
```
`simplex/_dna.py`:
```python
import numpy as np
import polars as pl

_ASCII = np.array([65, 67, 71, 84], dtype=np.uint8)
_COMP = bytes.maketrans(b"ACGTN", b"TGCAN")
_BASES = np.array(list("ACGT"))

def random_dna(rng, k, length):
    if k == 0:
        return np.array([], dtype=object)
    b = _ASCII[rng.integers(0, 4, size=(k, length), dtype=np.uint8)]
    return b.view(f"S{length}").reshape(k).astype(str)

def revcomp_str(s):
    return s.translate(_COMP)[::-1]

def revcomp_expr(col):
    return pl.col(col).str.reverse().str.replace_many(["A", "C", "G", "T"], ["T", "G", "C", "A"])

def mutate_strings(seqs, sub_rate, indel_rate, rng):
    out, counts = [], np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        chars, n = list(s), 0
        if sub_rate > 0:
            for p in np.nonzero(rng.random(len(chars)) < sub_rate)[0]:
                alt = rng.choice(_BASES)
                while alt == chars[p]:
                    alt = rng.choice(_BASES)
                chars[p] = str(alt); n += 1
        if indel_rate > 0:
            res = []
            for ch in chars:
                u = rng.random()
                if u < indel_rate / 2:
                    n += 1; continue
                res.append(ch)
                if u > 1 - indel_rate / 2:
                    res.append(str(rng.choice(_BASES))); n += 1
            chars = res
        out.append("".join(chars)); counts[i] = n
    return out, counts
```
`simplex/config.py`:
```python
import json
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass
class SimplexConfig:
    input_data: str
    output_directory: str
    n_cells: int | None = None
    wells: int = 96
    cells_per_droplet_mean: float = 5.0
    cells_per_droplet_sd: float = 2.0
    barcode_reuse: bool = False
    recovery_rate: float = 0.5
    molecules_per_chain_mean: float = 10.0
    release_rate: float = 0.02
    molecule_survival_rate: float = 0.8
    reads_per_molecule_mean: float = 5.0
    rt_sub_rate: float = 0.0
    rt_indel_rate: float = 0.0
    sequencing_sub_rate: float = 0.001
    sequencing_indel_rate: float = 0.0
    index_hop_rate: float = 0.001
    barcode_length: int = 16
    umi_length: int = 10
    tso: str = "TTTCTTATATGGG"
    chemistry: str = "v2"
    output_mode: str = "merged"
    read_length: int = 300
    rc_fraction: float = 0.0
    platform: str = "illumina"
    variable_length: bool = True
    write_read_truth: bool = False
    seed: int = 0

    _RATES = ("recovery_rate", "release_rate", "molecule_survival_rate", "rt_sub_rate",
              "rt_indel_rate", "sequencing_sub_rate", "sequencing_indel_rate",
              "index_hop_rate", "rc_fraction")

    def to_dict(self):
        return asdict(self)

    def to_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def estimated_reads(self, n_cells: int | None = None):
        n = n_cells if n_cells is not None else (self.n_cells or 0)
        return int(n * 2 * self.recovery_rate * self.molecules_per_chain_mean
                   * self.molecule_survival_rate * self.reads_per_molecule_mean)

    def validate(self, max_reads: int = 3_000_000_000):
        for r in self._RATES:
            v = getattr(self, r)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{r}={v} must be in [0,1]")
        if self.output_mode not in ("merged", "paired"):
            raise ValueError("output_mode must be 'merged' or 'paired'")
        if self.n_cells and self.estimated_reads() > max_reads:
            raise ValueError(f"estimated reads {self.estimated_reads()} exceeds budget {max_reads}")
        return self
```
`simplex/__init__.py`:
```python
from .config import SimplexConfig
__all__ = ["SimplexConfig", "run", "score"]
def run(*a, **k):
    from .run import run as _r
    return _r(*a, **k)
def score(*a, **k):
    from .scoring import score as _s
    return _s(*a, **k)
```
`pyproject.toml`: ensure `simplex*` included in packages.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest simplex/tests/test_rng.py simplex/tests/test_dna.py simplex/tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/ pyproject.toml
git commit -m "feat(simplex): scaffold — keyed RNG, DNA helpers, v2 config + validation"
```

---

### Task 2: load_pairs (locus-aware) + barcode loader

**Files:** Create `simplex/barcodes.py`, `simplex/cells.py` (load_pairs only here), `simplex/tests/test_load.py`.

**Interfaces (Produces):**
- `barcodes.load_barcodes(chemistry, n, rng) -> list[str]`
- `cells.load_pairs(input_data, n_cells=None, seed=0) -> pl.DataFrame` (cells schema incl `chain{0,1}_locus`; `source_pair_id` from input `name` if present else row index; locus from input `locus:0/1` if present else `"unknown"`).

- [ ] **Step 1: Write failing test**

`simplex/tests/test_load.py`:
```python
import polars as pl
from simplex.cells import load_pairs
from simplex.barcodes import load_barcodes
from simplex._rng import rng_for

def _inp(tmp_path, n=20, with_locus=True):
    d = {"sequence_id:0":[f"h{i}" for i in range(n)], "sequence:0":["ACGT"*90]*n,
         "sequence_id:1":[f"l{i}" for i in range(n)], "sequence:1":["TTGG"*80]*n,
         "name":[f"cell{i}" for i in range(n)]}
    if with_locus:
        d["locus:0"] = ["IGH"]*n; d["locus:1"] = ["IGK"]*n
    p = tmp_path/"p.parquet"; pl.DataFrame(d).write_parquet(p); return p

def test_load_maps_and_locus(tmp_path):
    c = load_pairs(_inp(tmp_path, 8))
    assert c.height == 8
    assert c["chain0_locus"][0] == "IGH" and c["chain1_locus"][0] == "IGK"
    assert c["source_pair_id"][0] == "cell0"

def test_load_locus_default(tmp_path):
    c = load_pairs(_inp(tmp_path, 4, with_locus=False))
    assert set(c["chain0_locus"].unique()) == {"unknown"}

def test_load_subsample(tmp_path):
    assert load_pairs(_inp(tmp_path, 100), n_cells=10, seed=1).height == 10

def test_barcodes_distinct():
    bcs = load_barcodes("v2", 300, rng_for(0, "barcodes"))
    assert len(bcs) == len(set(bcs)) == 300 and all(len(b) == 16 for b in bcs)
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest simplex/tests/test_load.py -q` → FAIL.

- [ ] **Step 3: Implement**

`simplex/barcodes.py`:
```python
import gzip
from pathlib import Path
from pairplex.utils import get_whitelist_path

def _read(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return [l.strip() for l in f if l.strip()]

def load_barcodes(chemistry, n, rng):
    wl = _read(Path(get_whitelist_path(chemistry.lower())))
    if n > len(wl):
        raise ValueError(f"need {n} barcodes, whitelist has {len(wl)}")
    idx = rng.choice(len(wl), size=n, replace=False)
    return [wl[i] for i in idx]
```
`simplex/cells.py` (load_pairs; droplet/well funcs added in Task 3):
```python
import numpy as np
import polars as pl
from ._rng import rng_for

def load_pairs(input_data, n_cells=None, seed=0):
    df = pl.read_parquet(input_data)
    req = {"sequence_id:0":"chain0_id","sequence:0":"chain0_seq",
           "sequence_id:1":"chain1_id","sequence:1":"chain1_seq"}
    missing = [k for k in req if k not in df.columns]
    if missing:
        raise ValueError(f"input missing columns: {missing}")
    src = pl.col("name") if "name" in df.columns else pl.first().cum_count()
    out = df.select(
        [pl.col(k).alias(v) for k, v in req.items()]
        + [(pl.col("name").cast(pl.Utf8) if "name" in df.columns
            else pl.int_range(pl.len()).cast(pl.Utf8)).alias("source_pair_id"),
           (pl.col("locus:0").cast(pl.Utf8) if "locus:0" in df.columns
            else pl.lit("unknown")).alias("chain0_locus"),
           (pl.col("locus:1").cast(pl.Utf8) if "locus:1" in df.columns
            else pl.lit("unknown")).alias("chain1_locus")]
    )
    if n_cells is not None:
        rng = rng_for(seed, "subsample")
        idx = rng.choice(out.height, size=n_cells, replace=n_cells > out.height)
        out = out[idx]
    return out.with_row_index("cell_id").select(
        ["cell_id","source_pair_id","chain0_id","chain0_seq","chain0_locus",
         "chain1_id","chain1_seq","chain1_locus"])
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest simplex/tests/test_load.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/barcodes.py simplex/cells.py simplex/tests/test_load.py
git commit -m "feat(simplex): locus-aware load_pairs + barcode loader"
```

---

### Task 3: Droplets (+ barcode_reuse) + wells + analytic collision test

**Files:** Modify `simplex/cells.py`; Create `simplex/tests/test_cells.py`.

**Interfaces (Produces):**
- `cells.assign_droplets_and_barcodes(cells, mean, sd, chemistry, barcode_reuse, seed) -> pl.DataFrame` (+`droplet_id, barcode`)
- `cells.assign_wells(cells, wells, seed) -> pl.DataFrame` (+`resident_well`)

- [ ] **Step 1: Write failing test**

`simplex/tests/test_cells.py`:
```python
import polars as pl
from simplex.cells import load_pairs, assign_droplets_and_barcodes, assign_wells

def _cells(tmp_path, n=400):
    d = {"sequence_id:0":[f"h{i}" for i in range(n)],"sequence:0":["A"*300]*n,
         "sequence_id:1":[f"l{i}" for i in range(n)],"sequence:1":["T"*300]*n,
         "name":[f"c{i}" for i in range(n)],"locus:0":["IGH"]*n,"locus:1":["IGK"]*n}
    p=tmp_path/"p.parquet"; pl.DataFrame(d).write_parquet(p); return load_pairs(p)

def test_droplet_shares_barcode(tmp_path):
    c = assign_droplets_and_barcodes(_cells(tmp_path), 5, 1, "v2", False, 0)
    per = c.group_by("droplet_id").agg(pl.col("barcode").n_unique().alias("nb"))
    assert per["nb"].max() == 1                                # one barcode per droplet
    assert c["barcode"].n_unique() == c["droplet_id"].n_unique()  # unique across droplets
    assert c["barcode"].n_unique() < c.height                  # overloading shares barcodes

def test_barcode_reuse_allows_collision(tmp_path):
    c = assign_droplets_and_barcodes(_cells(tmp_path), 5, 1, "v2", True, 0)
    # with reuse, #distinct barcodes may be < #droplets (allowed)
    assert c["barcode"].n_unique() <= c["droplet_id"].n_unique()

def test_wells_uniform(tmp_path):
    c = assign_wells(_cells(tmp_path, 4000), 8, 0)
    counts = c.group_by("resident_well").len()["len"].to_list()
    assert min(counts) > 4000/8*0.7 and max(counts) < 4000/8*1.3
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement** (append to `simplex/cells.py`)

```python
from .barcodes import load_barcodes

def assign_droplets_and_barcodes(cells, mean, sd, chemistry, barcode_reuse, seed):
    rng = rng_for(seed, "droplets")
    n = cells.height
    order = rng.permutation(n)
    droplet = np.empty(n, dtype=np.int64)
    idx, d = 0, 0
    while idx < n:
        size = max(1, int(round(rng.normal(mean, sd))))
        for _ in range(size):
            if idx >= n:
                break
            droplet[order[idx]] = d; idx += 1
        d += 1
    n_droplets = d
    if barcode_reuse:
        pool = np.array(load_barcodes(chemistry, max(1, n_droplets // 2 or 1), rng_for(seed, "bc")))
        bc_of_droplet = pool[rng.integers(0, len(pool), size=n_droplets)]
    else:
        bc_of_droplet = np.array(load_barcodes(chemistry, n_droplets, rng_for(seed, "bc")))
    return cells.with_columns([pl.Series("droplet_id", droplet),
                               pl.Series("barcode", bc_of_droplet[droplet])])

def assign_wells(cells, wells, seed):
    rng = rng_for(seed, "wells")
    return cells.with_columns(
        pl.Series("resident_well", rng.integers(0, wells, size=cells.height).astype(np.int64)))
```

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/cells.py simplex/tests/test_cells.py
git commit -m "feat(simplex): droplet/barcode (+reuse) and well assignment"
```

---

### Task 4: Molecules — recovery, UMIs, resident/free split, inherited RT error

**Files:** Create `simplex/molecules.py`, `simplex/tests/test_molecules.py`.

**Interfaces (Produces):**
- `molecules.generate_molecules(cells, recovery_rate, molecules_per_chain_mean, release_rate, umi_length, rt_sub_rate, rt_indel_rate, seed) -> pl.DataFrame` (molecules schema; RT error already applied to `cdna`, so a molecule's whole read family inherits it; `origin_barcode==final_barcode` here; `amplification_well` set later).

- [ ] **Step 1: Write failing test**

`simplex/tests/test_molecules.py`:
```python
import polars as pl
from simplex.molecules import generate_molecules

def _cells(n=1000):
    return pl.DataFrame({
        "cell_id":list(range(n)),"source_pair_id":[f"c{i}" for i in range(n)],
        "chain0_id":[f"h{i}" for i in range(n)],"chain0_seq":["ACGT"*80]*n,"chain0_locus":["IGH"]*n,
        "chain1_id":[f"l{i}" for i in range(n)],"chain1_seq":["TTGG"*80]*n,"chain1_locus":["IGK"]*n,
        "droplet_id":list(range(n)),"barcode":["ACGTACGTACGTACGT"]*n,"resident_well":[0]*n})

def test_recovery_fraction():
    m = generate_molecules(_cells(2000), 0.5, 5, 0.0, 10, 0.0, 0.0, 0)
    captured = m.select(["cell_id","chain"]).unique().height
    assert 0.4*2*2000 < captured < 0.6*2*2000

def test_umi_and_barcode_carried():
    m = generate_molecules(_cells(50), 1.0, 4, 0.0, 10, 0.0, 0.0, 1)
    assert m["umi"].str.len_chars().max() == 10
    assert (m["origin_barcode"] == m["final_barcode"]).all()
    assert set(m["chain"].unique()) == {0, 1}
    assert m["locus"].n_unique() == 2

def test_release_fraction():
    m = generate_molecules(_cells(2000), 1.0, 6, 0.2, 10, 0.0, 0.0, 2)
    assert 0.15 < m["is_free"].mean() < 0.25

def test_rt_error_applied():
    m = generate_molecules(_cells(200), 1.0, 6, 0.0, 10, 0.2, 0.0, 3)
    # some molecule cdna differs from the pristine template
    assert (m.filter(pl.col("chain")==0)["cdna"] != "ACGT"*80).sum() > 0
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement**

`simplex/molecules.py`:
```python
import numpy as np
import polars as pl
from ._dna import random_dna, mutate_strings
from ._rng import rng_for

def generate_molecules(cells, recovery_rate, molecules_per_chain_mean, release_rate,
                       umi_length, rt_sub_rate, rt_indel_rate, seed):
    rng = rng_for(seed, "molecules")
    n = cells.height
    frames = []
    for chain in (0, 1):
        captured = rng.random(n) < recovery_rate
        nmol = np.where(captured, np.maximum(rng.poisson(molecules_per_chain_mean, n), 1), 0).astype(np.int64)
        rep = np.repeat(np.arange(n), nmol)
        if rep.size == 0:
            continue
        sub = cells[rep]
        k = rep.size
        cdna = list(sub[f"chain{chain}_seq"])
        if rt_sub_rate > 0 or rt_indel_rate > 0:
            cdna, _ = mutate_strings(cdna, rt_sub_rate, rt_indel_rate, rng_for(seed, "rt", well=chain))
        bc = sub["barcode"].to_numpy().astype(str)
        frames.append(pl.DataFrame({
            "origin_cell_id": sub["cell_id"], "origin_droplet_id": sub["droplet_id"],
            "source_pair_id": sub["source_pair_id"],
            "chain": np.full(k, chain, np.int8), "locus": sub[f"chain{chain}_locus"],
            "umi": random_dna(rng, k, umi_length),
            "origin_barcode": bc, "final_barcode": bc,
            "resident_well": sub["resident_well"],
            "is_free": rng.random(k) < release_rate,
            "cdna": cdna,
        }))
    m = pl.concat(frames)
    return m.with_row_index("molecule_id").with_columns([
        pl.col("molecule_id").cast(pl.Int64),
        pl.col("molecule_id").cast(pl.Int64).alias("parent_molecule_id"),
    ])
```

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/molecules.py simplex/tests/test_molecules.py
git commit -m "feat(simplex): molecules — recovery, UMIs, resident/free, inherited RT error"
```

---

### Task 5: Routing + survival + amplification + index hopping

**Files:** Create `simplex/routing.py`, `simplex/tests/test_routing.py`.

**Interfaces (Produces):**
- `routing.route_and_amplify(molecules, wells, molecule_survival_rate, reads_per_molecule_mean, index_hop_rate, seed) -> pl.DataFrame` (reads schema). Steps in order: set `amplification_well` (resident→resident_well; free→independent uniform well, barcode+UMI kept); drop non-survivors; expand survivors into read families (share UMI, inherit `cdna`); set `final_well` = amplification_well except index-hopped reads → different well; set flags.

- [ ] **Step 1: Write failing test**

`simplex/tests/test_routing.py`:
```python
import numpy as np, polars as pl
from simplex.routing import route_and_amplify

def _mol(n=2000, wells=4):
    rng = np.random.default_rng(0)
    return pl.DataFrame({
        "molecule_id":list(range(n)),"parent_molecule_id":list(range(n)),
        "origin_cell_id":rng.integers(0,500,n),"origin_droplet_id":rng.integers(0,300,n),
        "source_pair_id":[f"c{i%500}" for i in range(n)],
        "chain":rng.integers(0,2,n).astype(np.int8),"locus":["IGH"]*n,
        "umi":["AAAAAAAAAA"]*n,"origin_barcode":["BC"]*n,"final_barcode":["BC"]*n,
        "resident_well":rng.integers(0,wells,n).astype(np.int64),
        "is_free":rng.random(n)<0.2,"cdna":["ACGT"*50]*n})

def test_free_molecule_redistributes_keeping_barcode_umi():
    r = route_and_amplify(_mol(), wells=4, molecule_survival_rate=1.0,
                          reads_per_molecule_mean=3, index_hop_rate=0.0, seed=0)
    assert (r["barcode"] == "BC").all() and (r["umi"] == "AAAAAAAAAA").all()
    # for non-hopped reads, final_well == amplification_well
    assert (r["final_well"] == r["amplification_well"]).all()

def test_survival_thins_molecules():
    r0 = route_and_amplify(_mol(), 4, 1.0, 3, 0.0, 0)
    r1 = route_and_amplify(_mol(), 4, 0.5, 3, 0.0, 0)
    assert r1["molecule_id"].n_unique() < r0["molecule_id"].n_unique()

def test_read_family_shares_umi():
    r = route_and_amplify(_mol(50), 4, 1.0, 5, 0.0, 0)
    per = r.group_by("molecule_id").agg(pl.col("umi").n_unique().alias("u"))
    assert per["u"].max() == 1

def test_index_hop_moves_reads():
    r = route_and_amplify(_mol(), 4, 1.0, 4, 0.2, 0)
    hopped = r.filter(pl.col("is_index_hopped"))
    assert (hopped["final_well"] != hopped["amplification_well"]).all()
    assert 0.15 < hopped.height / r.height < 0.25
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement**

`simplex/routing.py`:
```python
import numpy as np
import polars as pl
from ._rng import rng_for

def route_and_amplify(molecules, wells, molecule_survival_rate, reads_per_molecule_mean,
                      index_hop_rate, seed):
    rng = rng_for(seed, "routing")
    m = molecules
    nmol = m.height
    # amplification_well: resident keep resident_well; free pick independent uniform well
    free = m["is_free"].to_numpy()
    free_well = rng.integers(0, wells, size=nmol).astype(np.int64)
    amp_well = np.where(free, free_well, m["resident_well"].to_numpy()).astype(np.int64)
    m = m.with_columns(pl.Series("amplification_well", amp_well))
    # survival before amplification
    survived = rng.random(nmol) < molecule_survival_rate
    m = m.filter(pl.Series(survived))
    # amplify: read family per surviving molecule
    depth = np.maximum(rng.poisson(reads_per_molecule_mean, m.height), 1).astype(np.int64)
    rep = np.repeat(np.arange(m.height), depth)
    reads = m[rep]
    k = reads.height
    # index hopping
    hop = rng.random(k) < index_hop_rate
    offset = rng.integers(1, max(2, wells), size=k)
    amp = reads["amplification_well"].to_numpy()
    final_well = np.where(hop, (amp + offset) % wells, amp).astype(np.int64)
    return reads.with_columns([
        pl.Series("read_id", [f"r{i}" for i in range(k)]),
        pl.col("final_barcode").alias("barcode"),
        pl.Series("final_well", final_well),
        pl.Series("is_index_hopped", hop),
        pl.lit(False).alias("is_barcode_swapped"),
        pl.lit(0, dtype=pl.Int64).alias("n_seq_errors"),
    ]).select(["read_id","molecule_id","origin_cell_id","source_pair_id","chain","locus","umi",
               "barcode","amplification_well","final_well","is_free","is_index_hopped",
               "is_barcode_swapped","cdna","n_seq_errors"])
```

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/routing.py simplex/tests/test_routing.py
git commit -m "feat(simplex): routing — free redistribution, survival, amplification, index hop"
```

---

### Task 6: Sequencing errors + build merged reads (round-trip)

**Files:** Create `simplex/reads.py`, `simplex/tests/test_reads.py`.

**Interfaces (Produces):**
- `reads.apply_sequencing_errors(reads, sub_rate, indel_rate, seed) -> pl.DataFrame` (mutates per-read `cdna`, sets `n_seq_errors`).
- `reads.build_merged(reads, tso, rc_fraction, variable_length, seed) -> pl.DataFrame` (`read_id, final_well, read_seq, qual`).

- [ ] **Step 1: Write failing test**

`simplex/tests/test_reads.py`:
```python
import polars as pl
from simplex.reads import apply_sequencing_errors, build_merged
from pairplex.utils import parse_barcodes  # structure reference

def _reads(n=3):
    return pl.DataFrame({
        "read_id":[f"r{i}" for i in range(n)],"final_well":[0]*n,
        "barcode":["ACGTACGTACGTACGT"]*n,"umi":["AAAAAAAAAA"]*n,
        "cdna":["GATTACAGGT"*20]*n,"n_seq_errors":[0]*n})

def test_seq_error_independent_per_read():
    r = apply_sequencing_errors(_reads(500).with_columns(pl.col("cdna")), 0.05, 0.0, 0)
    assert r["n_seq_errors"].sum() > 0

def test_merged_round_trips_parse():
    b = build_merged(_reads(), tso="TTTCTTATATGGG", rc_fraction=0.0, variable_length=False, seed=0)
    s = b["read_seq"][0]
    assert s[:16] == "ACGTACGTACGTACGT" and s[16:26] == "AAAAAAAAAA"
    assert s[36:].lstrip("G") == ("GATTACAGGT"*20).lstrip("G")
    assert len(b["qual"][0]) == len(s)

def test_rc_fraction_parses_via_rc():
    from simplex._dna import revcomp_str
    b = build_merged(_reads(), "TTTCTTATATGGG", rc_fraction=1.0, variable_length=False, seed=0)
    assert revcomp_str(b["read_seq"][0])[:16] == "ACGTACGTACGTACGT"
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement**

`simplex/reads.py`:
```python
import numpy as np
import polars as pl
from ._dna import mutate_strings, revcomp_expr
from ._rng import rng_for

def apply_sequencing_errors(reads, sub_rate, indel_rate, seed):
    if sub_rate == 0 and indel_rate == 0:
        return reads
    rng = rng_for(seed, "seqerr")
    cdna, ne = mutate_strings(list(reads["cdna"]), sub_rate, indel_rate, rng)
    return reads.with_columns([pl.Series("cdna", cdna),
                               (pl.col("n_seq_errors") + pl.Series(ne)).alias("n_seq_errors")])

def build_merged(reads, tso, rc_fraction, variable_length, seed):
    r = reads
    if variable_length:
        rng = rng_for(seed, "trunc")
        lens = r["cdna"].str.len_chars().to_numpy()
        t5 = rng.integers(0, np.maximum(1, lens // 10)).astype(np.int64)
        nl = np.maximum(1, lens - t5 - rng.integers(0, np.maximum(1, lens // 10))).astype(np.int64)
        r = r.with_columns(pl.col("cdna").str.slice(pl.Series(t5), pl.Series(nl)).alias("cdna"))
    frag = pl.concat_str([pl.col("barcode"), pl.col("umi"), pl.lit(tso), pl.col("cdna")])
    r = r.with_columns(frag.alias("_frag"))
    rng = rng_for(seed, "rc")
    is_rc = pl.Series(rng.random(r.height) < rc_fraction)
    r = r.with_columns(is_rc.alias("_rc")).with_columns(
        pl.when(pl.col("_rc")).then(revcomp_expr("_frag")).otherwise(pl.col("_frag")).alias("read_seq"))
    r = r.with_columns(pl.col("read_seq").str.replace_all(".", "I").alias("qual"))
    return r.select(["read_id", "final_well", "read_seq", "qual"])
```

- [ ] **Step 4: Run to verify pass** — PASS. (If `str.slice` rejects Series args in this polars build, switch to `pl.col`-based offset/length exprs; the test will confirm.)

- [ ] **Step 5: Commit**

```bash
git add simplex/reads.py simplex/tests/test_reads.py
git commit -m "feat(simplex): independent sequencing errors + merged read assembly"
```

---

### Task 7: Ground truth — components, cells, barcodes

**Files:** Create `simplex/truth.py`, `simplex/tests/test_truth.py`.

**Interfaces (Produces):**
- `truth.build_truth_components(cells, reads) -> pl.DataFrame` keyed `(final_well, barcode, origin_cell_id, chain)` with `source_pair_id, locus, sequence, is_resident_source, n_source_molecules, n_umis, n_reads, n_reads_resident, n_reads_free, n_reads_index_hopped`.
- `truth.build_truth_cells(cells, reads) -> pl.DataFrame`; `truth.build_truth_barcodes(cells, components) -> pl.DataFrame` (per-locus dominance from component read counts).

- [ ] **Step 1: Write failing test**

`simplex/tests/test_truth.py`:
```python
import polars as pl
from simplex.truth import build_truth_components, build_truth_barcodes, build_truth_cells

def _cells():
    return pl.DataFrame({
        "cell_id":[0,1,2],"source_pair_id":["A","B","C"],
        "chain0_id":["hA","hB","hC"],"chain0_seq":["H_A","H_B","H_C"],"chain0_locus":["IGH"]*3,
        "chain1_id":["lA","lB","lC"],"chain1_seq":["L_A","L_B","L_C"],"chain1_locus":["IGK"]*3,
        "droplet_id":[0,0,1],"barcode":["X","X","Y"],"resident_well":[0,0,0]})

def _reads():
    # cell0 heavy resident at X/w0; cell1(B) free light lands at X/w0 (ambient); cell2 hopped
    return pl.DataFrame({
        "read_id":["r0","r1","r2"],"origin_cell_id":[0,1,2],"source_pair_id":["A","B","C"],
        "chain":[0,1,0],"locus":["IGH","IGK","IGH"],"barcode":["X","X","Y"],
        "final_well":[0,0,0],"is_free":[False,True,False],"is_index_hopped":[False,False,True]})

def test_components_counts_and_resident_flag():
    comp = build_truth_components(_cells(), _reads())
    a = comp.filter((pl.col("barcode")=="X")&(pl.col("origin_cell_id")==0)).to_dicts()[0]
    assert a["is_resident_source"] is True and a["n_reads_resident"] == 1
    b = comp.filter((pl.col("barcode")=="X")&(pl.col("origin_cell_id")==1)).to_dicts()[0]
    assert b["is_resident_source"] is False and b["n_reads_free"] == 1  # ambient at X

def test_barcodes_dominance_and_ambient_only():
    comp = build_truth_components(_cells(), _reads())
    tb = build_truth_barcodes(_cells(), comp)
    x = tb.filter((pl.col("final_well")==0)&(pl.col("barcode")=="X")).to_dicts()[0]
    assert x["n_resident_cells"] == 2 and x["is_collision"] is True
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement**

`simplex/truth.py`:
```python
import polars as pl

def _cell_seq(cells):
    # long: (origin_cell_id, chain) -> sequence, source_pair_id, resident_well, barcode
    parts = []
    for ch in (0, 1):
        parts.append(cells.select([
            pl.col("cell_id").alias("origin_cell_id"), pl.lit(ch).cast(pl.Int8).alias("chain"),
            pl.col(f"chain{ch}_seq").alias("sequence"), pl.col(f"chain{ch}_locus").alias("locus"),
            pl.col("source_pair_id"), pl.col("resident_well"), pl.col("barcode").alias("home_barcode")]))
    return pl.concat(parts)

def build_truth_components(cells, reads):
    cs = _cell_seq(cells)
    agg = reads.group_by(["final_well", "barcode", "origin_cell_id", "chain"]).agg([
        pl.col("source_pair_id").first(), pl.col("locus").first(),
        pl.len().alias("n_reads"),
        (~pl.col("is_free") & ~pl.col("is_index_hopped")).sum().alias("n_reads_resident"),
        pl.col("is_free").sum().alias("n_reads_free"),
        pl.col("is_index_hopped").sum().alias("n_reads_index_hopped"),
        pl.col("umi").n_unique().alias("n_umis"),
        pl.col("molecule_id").n_unique().alias("n_source_molecules") if "molecule_id" in reads.columns
            else pl.col("umi").n_unique().alias("n_source_molecules"),
    ])
    comp = agg.join(cs.select(["origin_cell_id","chain","sequence","resident_well","home_barcode"]),
                    on=["origin_cell_id","chain"], how="left")
    return comp.with_columns(
        ((pl.col("resident_well") == pl.col("final_well")) & (pl.col("home_barcode") == pl.col("barcode")))
        .alias("is_resident_source")
    ).drop(["resident_well","home_barcode"])

def build_truth_cells(cells, reads):
    counts = reads.group_by(["origin_cell_id","chain"]).agg([
        pl.len().alias("n_reads_generated"),
        (~pl.col("is_free")).sum().alias("n_reads_resident"),
        pl.col("is_free").sum().alias("n_reads_free_out"),
        pl.col("is_index_hopped").sum().alias("n_reads_index_hopped_out"),
        pl.col("umi").n_unique().alias("n_umis")]).rename({"origin_cell_id":"cell_id"})
    wide = counts.pivot(index="cell_id", on="chain",
                        values=["n_reads_generated","n_reads_resident","n_reads_free_out",
                                "n_reads_index_hopped_out","n_umis"])
    return cells.join(wide, on="cell_id", how="left").fill_null(0)

def build_truth_barcodes(cells, components):
    resident = components.filter(pl.col("is_resident_source"))
    occ = resident.group_by(["final_well","barcode"]).agg([
        pl.col("origin_cell_id").unique().alias("resident_cell_ids"),
        pl.col("origin_cell_id").n_unique().alias("n_resident_cells")])
    # per-locus dominance (by reads) among ALL sources at the key
    def dom(locus_name, alias):
        return (components.filter(pl.col("locus")==locus_name)
                .sort("n_reads", descending=True)
                .group_by(["final_well","barcode"]).agg(pl.col("source_pair_id").first().alias(alias)))
    tb = components.select(["final_well","barcode"]).unique() \
        .join(occ, on=["final_well","barcode"], how="left") \
        .join(dom("IGH","dominant_heavy_source"), on=["final_well","barcode"], how="left") \
        .join(dom("IGK","dominant_light_source"), on=["final_well","barcode"], how="left")
    return tb.with_columns([
        pl.col("n_resident_cells").fill_null(0),
        (pl.col("n_resident_cells").fill_null(0) >= 2).alias("is_collision"),
        (pl.col("n_resident_cells").fill_null(0) == 0).alias("is_ambient_only")])
```
> Note: dominance uses IGH for heavy and IGK/IGL for light; extend the light filter to `locus in {IGK,IGL}` when real loci appear. `n_source_molecules` requires `molecule_id` in `reads`; Task 5 keeps it — ensure the `reads` frame passed here retains `molecule_id` (Task 8 keeps it out of the FASTQ path but feeds the full frame to truth).

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/truth.py simplex/tests/test_truth.py
git commit -m "feat(simplex): compact truth (components, cells, barcodes) with per-locus dominance"
```

---

### Task 8: IO writers + run() orchestration

**Files:** Create `simplex/io.py`, `simplex/run.py`, `simplex/tests/test_run.py`.

**Interfaces (Produces):**
- `io.write_merged_fastq(built, output_dir, compress=True) -> list[Path]` (one file per `final_well`, streaming).
- `io.write_truth(output_dir, truth_components, truth_cells, truth_barcodes, truth_reads=None)`.
- `run.run(input_data, output_directory, **knobs) -> Path` (returns `reads/`).

- [ ] **Step 1: Write failing test**

`simplex/tests/test_run.py`:
```python
import gzip
from pathlib import Path
import polars as pl
from simplex.run import run

def _inp(tmp_path, n=60):
    d = {"sequence_id:0":[f"h{i}" for i in range(n)],"sequence:0":["GATTACA"*30]*n,
         "sequence_id:1":[f"l{i}" for i in range(n)],"sequence:1":["CCGGTA"*30]*n,
         "name":[f"c{i}" for i in range(n)],"locus:0":["IGH"]*n,"locus:1":["IGK"]*n}
    p=tmp_path/"in.parquet"; pl.DataFrame(d).write_parquet(p); return p

def test_run_outputs(tmp_path):
    out = tmp_path/"o"
    rd = run(input_data=_inp(tmp_path), output_directory=out, wells=4,
             cells_per_droplet_mean=1, cells_per_droplet_sd=0, variable_length=False, seed=0)
    assert Path(rd).is_dir() and list(Path(rd).glob("*.fastq.gz"))
    for f in ["truth_components.parquet","truth_cells.parquet","truth_barcodes.parquet"]:
        assert (out/"truth"/f).exists()
    assert (out/"simplex_config.json").exists() and (out/"run_manifest.json").exists()

def test_run_reproducible(tmp_path):
    def content(d):
        return sorted(gzip.open(p,"rt").read() for p in Path(d).glob("*.fastq.gz"))
    a = run(input_data=_inp(tmp_path), output_directory=tmp_path/"a", wells=4, seed=5)
    b = run(input_data=_inp(tmp_path), output_directory=tmp_path/"b", wells=4, seed=5)
    assert content(a) == content(b)
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement**

`simplex/io.py`:
```python
import gzip, json
from pathlib import Path

def _well_tag(w): return f"well{int(w):03d}"

def write_merged_fastq(built, output_dir, compress=True):
    rd = Path(output_dir)/"reads"; rd.mkdir(parents=True, exist_ok=True)
    ext = "fastq.gz" if compress else "fastq"
    op = (lambda p: gzip.open(p,"wt")) if compress else (lambda p: open(p,"w"))
    paths=[]
    for (well,), sub in built.group_by(["final_well"], maintain_order=True):
        p = rd/f"{_well_tag(well)}.{ext}"
        with op(p) as fh:
            fh.write("".join(f"@{i}\n{s}\n+\n{q}\n"
                     for i,s,q in zip(sub["read_id"], sub["read_seq"], sub["qual"])))
        paths.append(p)
    return paths

def write_truth(output_dir, components, cells, barcodes, reads=None):
    td = Path(output_dir)/"truth"; td.mkdir(parents=True, exist_ok=True)
    components.write_parquet(td/"truth_components.parquet")
    cells.write_parquet(td/"truth_cells.parquet")
    barcodes.write_parquet(td/"truth_barcodes.parquet")
    if reads is not None:
        reads.write_parquet(td/"truth_reads.parquet")
```

`simplex/run.py`:
```python
import json
from pathlib import Path
from .cells import load_pairs, assign_droplets_and_barcodes, assign_wells
from .config import SimplexConfig
from .molecules import generate_molecules
from .routing import route_and_amplify
from .reads import apply_sequencing_errors, build_merged
from .truth import build_truth_components, build_truth_cells, build_truth_barcodes
from .io import write_merged_fastq, write_truth

def run(input_data, output_directory, **knobs):
    cfg = SimplexConfig(input_data=str(input_data), output_directory=str(output_directory), **knobs)
    cells = load_pairs(cfg.input_data, cfg.n_cells, cfg.seed)
    cfg.validate()  # after we know n_cells for the OOM estimate
    if cfg.n_cells is None:
        cfg.estimated_reads(cells.height)
    out = Path(output_directory); out.mkdir(parents=True, exist_ok=True)

    cells = assign_droplets_and_barcodes(cells, cfg.cells_per_droplet_mean, cfg.cells_per_droplet_sd,
                                         cfg.chemistry, cfg.barcode_reuse, cfg.seed)
    cells = assign_wells(cells, cfg.wells, cfg.seed)
    mols = generate_molecules(cells, cfg.recovery_rate, cfg.molecules_per_chain_mean,
                              cfg.release_rate, cfg.umi_length, cfg.rt_sub_rate, cfg.rt_indel_rate, cfg.seed)
    reads = route_and_amplify(mols, cfg.wells, cfg.molecule_survival_rate,
                              cfg.reads_per_molecule_mean, cfg.index_hop_rate, cfg.seed)
    # keep molecule_id for truth; add it back onto reads via route output (already present)
    reads = apply_sequencing_errors(reads, cfg.sequencing_sub_rate, cfg.sequencing_indel_rate, cfg.seed)

    comp = build_truth_components(cells, reads)
    tcells = build_truth_cells(cells, reads)
    tbar = build_truth_barcodes(cells, comp)
    built = build_merged(reads, cfg.tso, cfg.rc_fraction, cfg.variable_length, cfg.seed)

    write_merged_fastq(built, out)
    write_truth(out, comp, tcells, tbar, reads if cfg.write_read_truth else None)
    cfg.to_json(out/"simplex_config.json")
    Path(out/"run_manifest.json").write_text(json.dumps(
        {"seed": cfg.seed, "n_cells": cells.height, "wells": cfg.wells,
         "estimated_reads": cfg.estimated_reads(cells.height)}, indent=2))
    return out/"reads"
```
> Note: `route_and_amplify` output must retain `molecule_id` for `truth_components.n_source_molecules`; add `molecule_id` to its `select(...)` list (adjust Task 5 select to include `"molecule_id"`). The reads→truth path uses the full reads frame; only `build_merged` strips to FASTQ columns.

- [ ] **Step 4: Run to verify pass** — PASS. (If `n_source_molecules` errors, confirm Task 5 keeps `molecule_id`.)

- [ ] **Step 5: Commit**

```bash
git add simplex/io.py simplex/run.py simplex/tests/test_run.py
git commit -m "feat(simplex): merged FASTQ writer + run() orchestration + manifest"
```

---

### Task 9: Scorer core — locus-restricted, key-local, joint set resolution

**Files:** Create `simplex/matching.py`, `simplex/tests/test_matching.py`.

**Interfaces (Produces):**
- `matching.build_key_index(truth_components) -> dict[(well,barcode) -> {locus -> {sequence -> set(source_pair_id)}}]`
- `matching.candidates(seq, locus, key_index_entry) -> set[str]` (exact/substring, locus-restricted).
- `matching.resolve_pair(h_cands, l_cands) -> (pairing_status, resolved_source|None)` implementing the joint rule.

- [ ] **Step 1: Write failing test**

`simplex/tests/test_matching.py`:
```python
from simplex.matching import resolve_pair

def test_joint_resolves_when_intersection_unique():
    assert resolve_pair({"A","B"}, {"A"}) == ("correct", "A")

def test_mispaired_disjoint():
    assert resolve_pair({"A"}, {"B"}) == ("mispaired", None)

def test_ambiguous_multiple_pairs():
    assert resolve_pair({"A","B"}, {"A","B"}) == ("ambiguous", None)

def test_unmatchable_empty():
    assert resolve_pair(set(), {"A"}) == ("unmatchable", None)
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement**

`simplex/matching.py`:
```python
def build_key_index(truth_components):
    idx = {}
    for r in truth_components.iter_rows(named=True):
        key = (int(r["final_well"]), r["barcode"])
        loc = idx.setdefault(key, {}).setdefault(r["locus"], {})
        loc.setdefault(r["sequence"], set()).add(r["source_pair_id"])
    return idx

def candidates(seq, locus, key_entry):
    if not seq or key_entry is None:
        return set()
    hits = set()
    for full, sources in key_entry.get(locus, {}).items():
        if seq == full or seq in full or full.endswith(seq):
            hits |= sources
    return hits

def resolve_pair(h_cands, l_cands):
    if not h_cands or not l_cands:
        return ("unmatchable", None)
    inter = h_cands & l_cands
    if len(inter) == 1:
        return ("correct", next(iter(inter)))
    if inter:
        return ("ambiguous", None)
    # disjoint but each side singleton -> a clean cross-source mispair
    if len(h_cands) == 1 and len(l_cands) == 1:
        return ("mispaired", None)
    return ("ambiguous", None)
```
> Note: heavy vs light is decided by the PairPlex output's own `locus:0/1` for *which chain is which*, but candidate lookup uses **truth** loci at the key. When a pair's two chains are same-locus (rare/erroneous), treat as `ambiguous`.

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/matching.py simplex/tests/test_matching.py
git commit -m "feat(simplex): joint locus-restricted key-local pair resolution"
```

---

### Task 10: Scorer — pair_scores + key_scores + orthogonal axes

**Files:** Create `simplex/scoring.py`, `simplex/tests/test_scoring.py`.

**Interfaces (Produces):**
- `scoring.score(pairplex_paired_parquet, truth_dir, pairplex_metadata=None) -> (pair_scores: pl.DataFrame, key_scores: pl.DataFrame)`.
  - `pair_scores`: `well, barcode, pairing_status, origin_status, key_status, output_status, resolved_source`.
  - `key_scores`: one row per truth `(well,barcode)`: `well, barcode, key_status, output_count, output_status(unique|duplicate|missing), captured_both, sequenced_both, reference_pairable_both, no_output_reason`.

- [ ] **Step 1: Write failing test**

`simplex/tests/test_scoring.py`:
```python
import polars as pl
from simplex.scoring import score

def _truth(tmp_path):
    td = tmp_path/"truth"; td.mkdir()
    comp = pl.DataFrame({
        "final_well":[0,0],"barcode":["X","X"],"origin_cell_id":[0,0],
        "source_pair_id":["A","A"],"chain":[0,1],"locus":["IGH","IGK"],
        "sequence":["HSEQ_A","LSEQ_A"],"is_resident_source":[True,True],
        "n_source_molecules":[3,3],"n_umis":[3,3],"n_reads":[9,9],
        "n_reads_resident":[9,9],"n_reads_free":[0,0],"n_reads_index_hopped":[0,0]})
    comp.write_parquet(td/"truth_components.parquet")
    pl.DataFrame({"cell_id":[0],"source_pair_id":["A"]}).write_parquet(td/"truth_cells.parquet")
    pl.DataFrame({"final_well":[0],"barcode":["X"],"n_resident_cells":[1],
                  "is_collision":[False],"is_ambient_only":[False]}).write_parquet(td/"truth_barcodes.parquet")
    return td

def _pp(tmp_path, seq0="HSEQ_A", seq1="LSEQ_A"):
    p = tmp_path/"pp.parquet"
    pl.DataFrame({"name":["X_d0_w0"],"well":["0"],"sequence_id:0":["X_contig-0_d0_w0"],
                  "sequence:0":[seq0],"locus:0":["IGH"],"sequence_id:1":["X_contig-1_d0_w0"],
                  "sequence:1":[seq1],"locus:1":["IGK"]}).write_parquet(p)
    return p

def test_correct_resident(tmp_path):
    ps, ks = score(_pp(tmp_path), _truth(tmp_path))
    r = ps.to_dicts()[0]
    assert r["pairing_status"] == "correct" and r["origin_status"] == "resident"
    assert r["well"] == 0 and r["barcode"] == "X"

def test_key_missing_when_no_output(tmp_path):
    # PairPlex emits a different barcode -> truth key X has no output
    p = tmp_path/"pp2.parquet"
    pl.DataFrame({"name":["Z_d0_w0"],"well":["0"],"sequence_id:0":["Z_contig-0"],
                  "sequence:0":["HSEQ_A"],"locus:0":["IGH"],"sequence_id:1":["Z_contig-1"],
                  "sequence:1":["LSEQ_A"],"locus:1":["IGK"]}).write_parquet(p)
    ps, ks = score(p, _truth(tmp_path))
    xrow = ks.filter((pl.col("well")==0)&(pl.col("barcode")=="X")).to_dicts()[0]
    assert xrow["output_status"] == "missing"
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement**

`simplex/scoring.py`:
```python
import re
from pathlib import Path
import polars as pl
from .matching import build_key_index, candidates, resolve_pair

_REF_MIN_READS, _REF_MIN_UMIS = 3, 1  # frozen reference-pairable minimum (threshold-independent)

def _barcode_from_id(sid):
    return re.split(r"_contig", sid)[0] if sid else sid

def score(pairplex_paired_parquet, truth_dir, pairplex_metadata=None):
    truth_dir = Path(truth_dir)
    comp = pl.read_parquet(truth_dir/"truth_components.parquet")
    tbar = pl.read_parquet(truth_dir/"truth_barcodes.parquet")
    idx = build_key_index(comp)
    key_status = {(int(r["final_well"]), r["barcode"]):
                  ("collision" if r["is_collision"] else
                   "ambient_only" if r["is_ambient_only"] else "singleton")
                  for r in tbar.iter_rows(named=True)}

    df = pl.read_parquet(pairplex_paired_parquet)
    pair_rows, seen_keys = [], {}
    for r in df.to_dicts():
        well = int(r["well"]); bc = _barcode_from_id(r.get("sequence_id:0") or r.get("name",""))
        key = (well, bc); entry = idx.get(key)
        # locus-restricted candidate sets (chain roles from PairPlex loci)
        loc0, loc1 = r.get("locus:0"), r.get("locus:1")
        h_seq = r.get("sequence:0") if loc0 == "IGH" else r.get("sequence:1")
        l_seq = r.get("sequence:1") if loc0 == "IGH" else r.get("sequence:0")
        h_c = candidates(h_seq, "IGH", entry)
        l_c = candidates(l_seq, "IGK", entry) | candidates(l_seq, "IGL", entry)
        status, resolved = resolve_pair(h_c, l_c)
        resident_sources = {s for s in (h_c | l_c)
                            if entry and any(True for _ in [])}  # placeholder; refine below
        # origin status
        if status == "correct":
            row = comp.filter((pl.col("final_well")==well)&(pl.col("barcode")==bc)
                              &(pl.col("source_pair_id")==resolved))
            is_res = bool(row["is_resident_source"].any()) if row.height else False
            origin = "resident" if is_res else "ambient"
        else:
            origin = "resident_plus_ambient"
        seen_keys[key] = seen_keys.get(key, 0) + 1
        pair_rows.append({"well":well,"barcode":bc,"pairing_status":status,
                          "origin_status":origin,"key_status":key_status.get(key,"ambient_only"),
                          "output_status":"unique","resolved_source":resolved})
    # duplicate outputs
    for pr in pair_rows:
        if seen_keys[(pr["well"], pr["barcode"])] > 1:
            pr["output_status"] = "duplicate"
    pair_scores = pl.DataFrame(pair_rows) if pair_rows else pl.DataFrame(
        schema={"well":pl.Int64,"barcode":pl.Utf8,"pairing_status":pl.Utf8,
                "origin_status":pl.Utf8,"key_status":pl.Utf8,"output_status":pl.Utf8,
                "resolved_source":pl.Utf8})

    # key_scores: one row per truth (well,barcode)
    key_rows = []
    res = comp.filter(pl.col("is_resident_source"))
    obs = (res.group_by(["final_well","barcode","chain"])
           .agg([pl.col("n_reads").sum().alias("nr"), pl.col("n_umis").sum().alias("nu")]))
    for r in tbar.iter_rows(named=True):
        well, bc = int(r["final_well"]), r["barcode"]
        oc = seen_keys.get((well, bc), 0)
        chains = obs.filter((pl.col("final_well")==well)&(pl.col("barcode")==bc))
        seq_both = chains["chain"].n_unique() == 2
        ref_ok = seq_both and bool((chains["nr"] >= _REF_MIN_READS).all()) and bool((chains["nu"] >= _REF_MIN_UMIS).all())
        key_rows.append({"well":well,"barcode":bc,
                         "key_status":("collision" if r["is_collision"] else
                                       "ambient_only" if r["is_ambient_only"] else "singleton"),
                         "output_count":oc,
                         "output_status":("missing" if oc==0 else "unique" if oc==1 else "duplicate"),
                         "captured_both": seq_both, "sequenced_both": seq_both,
                         "reference_pairable_both": ref_ok,
                         "no_output_reason": None if oc>0 else "unknown"})
    key_scores = pl.DataFrame(key_rows)
    return pair_scores, key_scores
```
> Note: remove the placeholder `resident_sources` line during implementation (origin is computed from the resolved source's `is_resident_source`). `no_output_reason` refinement beyond `unknown` requires `pairplex_metadata` (best-effort); wire it when a real `metadata/*.csv` is supplied. Extend light locus set to `{IGK,IGL}` (already done for candidates).

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/scoring.py simplex/tests/test_scoring.py
git commit -m "feat(simplex): scorer — pair_scores + key_scores, orthogonal axes, missing outputs"
```

---

### Task 11: Deterministic mechanism fixtures (generator → PairPlex → scorer)

**Files:** Create `simplex/tests/test_mechanism.py`. Uses abstar's bundled bnAbs as realistic input.

**Interfaces (Consumes):** `simplex.run`, `simplex.score`, `pairplex.run`.

- [ ] **Step 1: Write failing/again-driving tests** (the six fixtures; two shown fully, the rest follow the same shape — implement all six)

```python
import os
import polars as pl
import pairplex
from simplex.run import run
from simplex.scoring import score

def _input(tmp_path, names):
    import abstar
    from abutils.io import parse_fastx
    td = os.path.dirname(abstar.__file__) + "/test_data"
    hcs = {s.id: s.sequence for s in parse_fastx(td+"/test_hiv_bnab_hcs.fasta")}
    lcs = {s.id: s.sequence for s in parse_fastx(td+"/test_hiv_bnab_lcs.fasta")}
    df = pl.DataFrame({"sequence_id:0":names,"sequence:0":[hcs[n] for n in names],
                       "locus:0":["IGH"]*len(names),
                       "sequence_id:1":names,"sequence:1":[lcs[n] for n in names],
                       "locus:1":["IGK"]*len(names),"name":names})
    p = tmp_path/"in.parquet"; df.write_parquet(p); return p

def _run_pp(reads_dir, out):
    pairplex.run(sequences=str(reads_dir), output_directory=str(out),
                 clustering_threshold=0.9, min_cluster_reads=3, min_cluster_umis=1, quiet=True)

def test_one_cell_negative_control(tmp_path):
    names = [n for n in _names() if True][:16]
    inp = _input(tmp_path, names)
    out = tmp_path/"sim"
    rd = run(input_data=inp, output_directory=out, wells=2,
             cells_per_droplet_mean=1, cells_per_droplet_sd=0, recovery_rate=0.7,
             molecules_per_chain_mean=6, reads_per_molecule_mean=5, molecule_survival_rate=1.0,
             release_rate=0.1, index_hop_rate=0.0, sequencing_sub_rate=0.0, variable_length=False, seed=0)
    ppo = tmp_path/"pp"; _run_pp(rd, ppo)
    for pf in (ppo/"annotated").glob("*_paired.parquet"):
        ps, _ = score(pf, out/"truth")
        assert (ps["pairing_status"] == "mispaired").sum() == 0   # no cross-source mispairs at 1 cell/bc

def test_exact_ambient_mispair(tmp_path):
    # two cells share a barcode (cells_per_droplet=2), light dropout -> ambient light mispair emerges
    names = _names()[:24]
    inp = _input(tmp_path, names)
    out = tmp_path/"sim2"
    rd = run(input_data=inp, output_directory=out, wells=1,
             cells_per_droplet_mean=2, cells_per_droplet_sd=0, recovery_rate=0.6,
             molecules_per_chain_mean=8, reads_per_molecule_mean=5, molecule_survival_rate=1.0,
             release_rate=0.2, index_hop_rate=0.0, sequencing_sub_rate=0.0, variable_length=False, seed=3)
    ppo = tmp_path/"pp2"; _run_pp(rd, ppo)
    mis = sum((score(pf, out/"truth")[0]["pairing_status"] == "mispaired").sum()
              for pf in (ppo/"annotated").glob("*_paired.parquet"))
    assert mis > 0

def _names():
    import os, abstar
    from abutils.io import parse_fastx
    td = os.path.dirname(abstar.__file__)+"/test_data"
    h = {s.id for s in parse_fastx(td+"/test_hiv_bnab_hcs.fasta")}
    l = {s.id for s in parse_fastx(td+"/test_hiv_bnab_lcs.fasta")}
    return sorted(h & l)
```
Implement the remaining four fixtures from spec §12: **same-well collision** (2 same-barcode cells one well, asymmetric loss → a mispair scored `collision` key_status), **route composition** (assert on `truth_reads` that a molecule has `amplification_well != final_well` with unchanged barcode+UMI when `write_read_truth=True, index_hop_rate` high), **joint ambiguity** (craft an input with a shared heavy across two source pairs but distinct lights; assert the correct pair is `correct`, not `ambiguous`), **missing output** (contaminant contig causes PairPlex to drop a resident pair → `key_scores` `output_status=missing`).

- [ ] **Step 2: Run to verify behavior** — Run: `python -m pytest simplex/tests/test_mechanism.py -q`. The negative control and ambient fixtures must pass; iterate on generator/scorer if a fixture reveals a mechanism bug (this is the point of these tests).

- [ ] **Step 3–4:** (fixtures are the tests; no separate impl beyond fixing any bug they expose)

- [ ] **Step 5: Commit**

```bash
git add simplex/tests/test_mechanism.py
git commit -m "test(simplex): deterministic mechanism fixtures (ambient, collision, routing, ambiguity, missing)"
```

---

### Task 12: Statistical single-factor tests

**Files:** Create `simplex/tests/test_single_factor.py`.

- [ ] **Step 1: Write tests** — assert **mechanistic statistics** (not blanket monotonicity):
  - free molecules + no dropout → `key_scores` shows mostly yield loss (missing/duplicate) with low `mispaired` fraction in `pair_scores`.
  - free + light dropout → nonzero `mispaired`.
  - raising `min_cluster_fraction` in the PairPlex call reduces `mispaired` while reducing recall (`sequenced_both` recall drops) — assert the *tradeoff direction*, not a single metric.

```python
import pairplex, polars as pl
from simplex.run import run
from simplex.scoring import score
# ... build input as in Task 11 ...
def test_fraction_filter_trades_precision_for_yield(tmp_path):
    # generate one contaminated dataset, score under two min_cluster_fraction settings
    ...  # run generator once; run pairplex twice (0.0 and 0.25); compare mispaired and recall
```

- [ ] **Step 2–4:** Run; iterate until the tradeoff assertions hold.

- [ ] **Step 5: Commit**

```bash
git add simplex/tests/test_single_factor.py
git commit -m "test(simplex): statistical single-factor mechanism tests"
```

---

### Task 13 (optional, Phase 0A): Real-data audit

**Files:** Create `simplex/audit.py`, `simplex/tests/test_audit.py`.

**Interfaces (Produces):** `audit.audit_metadata(metadata_csv_glob, output_report) -> pl.DataFrame` — marginal summaries only (reads/UMIs/cluster_fraction distributions; 1H+1L / 1H+2L / 2H+1L frequencies via contig counts per barcode). Dataset-agnostic; **documents the no-labeled-truth limitation** in the report header. Runs only when a real `metadata/*.csv` is supplied.

- [ ] **Step 1: Write failing test** against a synthetic metadata CSV fixture (columns `name, reads, umis, cluster_fraction, pass_filters` mimicking PairPlex `metadata/*.csv`), asserting the summary table has the expected marginal rows and that the report states the no-truth caveat.

- [ ] **Step 2–4:** Implement `audit_metadata` (polars groupby/quantiles); run; PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/audit.py simplex/tests/test_audit.py
git commit -m "feat(simplex): optional Phase 0A real-data marginal audit (agnostic, no calibration gate)"
```

---

## Self-Review

**Spec coverage (v3 → tasks):**
- keyed RNG (spec §10) → Task 1. Config/API §9 → Task 1. DNA/round-trip → Tasks 1,6.
- load_pairs + locus contract §5,§9 → Task 2. droplets+reuse+wells §4 → Task 3.
- molecules: recovery, UMIs, resident/free, inherited RT error §4,§7 → Task 4.
- routing: free redistribution (barcode+UMI kept), survival-before-amplification, amplification, index hop §4 → Task 5.
- independent seq error + merged assembly §7,§8 → Task 6.
- truth_components (compositional, resident/free/hopped counts), truth_cells, truth_barcodes (per-locus dominance) §5 → Task 7.
- writers + run + manifest §5,§13 → Task 8.
- scorer: joint locus-restricted key-local resolution §6 → Task 9; pair_scores + key_scores + orthogonal axes + observability §6 → Task 10.
- deterministic fixtures (all six) §12 → Task 11; statistical single-factor §12 → Task 12.
- Phase 0A optional audit §2 → Task 13.
- **Deferred/tracked (not in plan, intentionally):** paired-end/fastp (Phase 3), alt pairing strategies + `pairing_policy` (deferred), `barcode_swap`, PCR chimeras, empirical calibration, 1M streaming, PairPlex hygiene (separate branch). Matches spec §11/§14.

**Placeholder scan:** two spots explicitly flagged for cleanup during implementation — the `resident_sources` placeholder line in Task 10 (remove; origin derives from resolved source) and Task 5's `select` must retain `molecule_id`. Task 11's four remaining fixtures are described with exact assertions to implement. No silent "TODO"s.

**Type consistency:** column names match the canonical schemas block and flow consistently (`final_well`, `barcode`, `origin_cell_id`, `source_pair_id`, `chain`, `locus`, `umi`, `molecule_id`, `is_free/is_index_hopped`, `n_reads_resident/free/index_hopped`). Scorer keys `(final_well→well, barcode)`; `resolve_pair` signature stable across Tasks 9–11.

**Known execution risks (watch, not gaps):** polars `str.slice`/`pivot`/`group_by` call shapes on 1.39 (tests catch); ensure Task 5 keeps `molecule_id`; `pairplex.run` merged-mode on synthetic wells (Task 11 uses merged so no fastp dependency); light-locus set `{IGK,IGL}`.
