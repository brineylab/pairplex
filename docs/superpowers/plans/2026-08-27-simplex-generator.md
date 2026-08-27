# SimPlex Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `simplex`, a sibling package that turns real paired antibody parquet into synthetic "raw sequencing" FASTQ (per well) with tunable wet-lab knobs and known ground truth, so we can stress-test PairPlex and find the best pairing thresholds.

**Architecture:** A staged, vectorized pipeline (polars + numpy). Pure functions transform a cells → molecules → reads table; contamination/error/read-assembly stages are independent and testable; FASTQ is written streaming per well. Everything is reproducible from one `seed`. See spec: `docs/superpowers/specs/2026-08-27-simplex-generator-design.md`.

**Tech Stack:** Python 3.10+, polars 1.39, numpy 2.x, pytest. Reuses `pairplex` barcode whitelists and read-structure (`barcode=s[:16]`, `umi=s[16:26]`, `sequence=s[36:].lstrip("G")`).

## Global Constraints

- Package is a **sibling** top-level package `simplex/` (NOT under `pairplex/`); import as `import simplex`.
- Read layout every merged read must satisfy: `s[:16]=barcode`, `s[16:26]=umi`, `s[36:].lstrip("G")` recovers the cDNA. Achieved with `barcode(16)+umi(10)+TSO("TTTCTTATATGGG")+cDNA`.
- All randomness flows from a single `seed`; each stage derives its own generator as `np.random.default_rng(seed + <fixed offset>)`. Same seed ⇒ identical content.
- Pairing-correctness checks in tests MUST match by **sequence** (substring/identity), never `junction_aa` (abstar's bnAb test set shares CDR3s — see `INVESTIGATION_NOTES.md`).
- Target scale 100k–1M cells: stages are vectorized; no per-read Python loops in hot paths; FASTQ written streaming per well.
- Default chemistry `"v2"` → whitelist `pairplex/barcodes/737K-august-2016.txt`.
- Commit after every task. Tests live in `simplex/tests/`.

## File structure

```
simplex/
  __init__.py          # exposes run, SimplexConfig
  _dna.py              # vectorized DNA helpers: random_dna, revcomp_expr, mutate arrays
  config.py            # SimplexConfig dataclass (all knobs) + to_dict/to_json
  barcodes.py          # chemistry -> whitelist path; load_barcodes(n)
  cells.py             # load_pairs, assign_droplets_and_barcodes, assign_wells
  molecules.py         # generate_molecules, amplify_and_sequence
  contamination.py     # inject_ambient, index_hop
  errors.py            # apply_sequencing_errors
  reads.py             # build_reads (merged/paired, RC, variable length)
  truth.py             # build_truth_cells, build_truth_barcodes, build_truth_reads
  io.py                # write_fastq, write_truth
  run.py               # run(**knobs) orchestration
  tests/
    __init__.py
    test_dna.py test_config.py test_barcodes.py test_cells.py
    test_molecules.py test_contamination.py test_errors.py
    test_reads.py test_truth.py test_io.py test_integration.py
```

**Canonical dataframe schemas (contract across tasks):**

- `cells`: `cell_id:Int64, source_pair_id:Utf8, chain0_id:Utf8, chain0_seq:Utf8, chain1_id:Utf8, chain1_seq:Utf8` → +`droplet_id:Int64, barcode:Utf8` → +`well:Int64`
- `molecules`: `cell_id:Int64, true_cell_id:Int64, well:Int64, barcode:Utf8, chain:Int8, umi:Utf8, cdna:Utf8`
- `reads`: molecules columns + `read_id:Utf8, is_ambient:Bool, is_leakage:Bool, is_index_hopped:Bool, n_errors:Int64`
- `built` (merged): `read_id, well, read_seq, qual`; (paired): `read_id, well, r1_seq, r1_qual, r2_seq, r2_qual`

---

### Task 1: Scaffold, DNA helpers, and SimplexConfig

**Files:**
- Create: `simplex/__init__.py`, `simplex/_dna.py`, `simplex/config.py`
- Create: `simplex/tests/__init__.py`, `simplex/tests/test_dna.py`, `simplex/tests/test_config.py`
- Modify: `pyproject.toml` (add `simplex` to packages)

**Interfaces:**
- Produces: `_dna.random_dna(rng: np.random.Generator, k: int, length: int) -> np.ndarray[str]`; `_dna.revcomp_expr(col: str) -> pl.Expr`; `_dna.revcomp_str(s: str) -> str`; `config.SimplexConfig` dataclass with `.to_dict()` and `.to_json(path)`.

- [ ] **Step 1: Write the failing test**

`simplex/tests/test_dna.py`:
```python
import numpy as np
import polars as pl
from simplex._dna import random_dna, revcomp_expr, revcomp_str


def test_random_dna_shape_and_alphabet():
    rng = np.random.default_rng(0)
    out = random_dna(rng, 5, 10)
    assert len(out) == 5
    assert all(len(s) == 10 for s in out)
    assert set("".join(out)) <= set("ACGT")


def test_random_dna_reproducible():
    a = random_dna(np.random.default_rng(7), 100, 16)
    b = random_dna(np.random.default_rng(7), 100, 16)
    assert list(a) == list(b)


def test_revcomp_str():
    assert revcomp_str("AAACCTGGN") == "NCCAGGTTT"


def test_revcomp_expr():
    df = pl.DataFrame({"s": ["AAACCTG", "ACGT"]})
    got = df.select(revcomp_expr("s").alias("r"))["r"].to_list()
    assert got == ["CAGGTTT", "ACGT"]
```

`simplex/tests/test_config.py`:
```python
import json
from simplex.config import SimplexConfig


def test_defaults():
    c = SimplexConfig(input_data="x.parquet", output_directory="out")
    assert c.wells == 96
    assert c.output_mode == "paired"
    assert c.tso == "TTTCTTATATGGG"


def test_to_json_roundtrip(tmp_path):
    c = SimplexConfig(input_data="x.parquet", output_directory="out", seed=3)
    p = tmp_path / "cfg.json"
    c.to_json(p)
    data = json.loads(p.read_text())
    assert data["seed"] == 3
    assert data["ambient_rate"] == c.ambient_rate
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest simplex/tests/test_dna.py simplex/tests/test_config.py -q`
Expected: FAIL (module `simplex` not found).

- [ ] **Step 3: Write minimal implementation**

`simplex/_dna.py`:
```python
import numpy as np
import polars as pl

_ASCII = np.array([65, 67, 71, 84], dtype=np.uint8)  # A C G T
_COMP = bytes.maketrans(b"ACGTN", b"TGCAN")


def random_dna(rng: np.random.Generator, k: int, length: int) -> np.ndarray:
    """Return array of k random DNA strings of the given length (vectorized)."""
    if k == 0:
        return np.array([], dtype=object)
    idx = rng.integers(0, 4, size=(k, length), dtype=np.uint8)
    b = _ASCII[idx]  # k x length uint8
    return b.view(f"S{length}").reshape(k).astype(str)


def revcomp_str(s: str) -> str:
    return s.translate(_COMP)[::-1]


def revcomp_expr(col: str) -> pl.Expr:
    """Vectorized reverse-complement of a polars string column."""
    return (
        pl.col(col)
        .str.reverse()
        .str.replace_many(["A", "C", "G", "T"], ["T", "G", "C", "A"])
    )
```

`simplex/config.py`:
```python
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SimplexConfig:
    input_data: str
    output_directory: str
    n_cells: int | None = None
    wells: int = 96
    cells_per_droplet_mean: float = 5.0
    cells_per_droplet_sd: float = 2.0
    recovery_rate: float = 0.5
    molecules_per_chain_mean: float = 10.0
    reads_per_molecule_mean: float = 5.0
    seq_efficiency: float = 0.8
    ambient_rate: float = 0.02
    leakage_rate: float = 0.01
    ambient_only_barcodes: int = 0
    index_hop_rate: float = 0.001
    sub_rate: float = 0.001
    indel_rate: float = 0.0
    errors_per_read: float | None = None
    error_regions: tuple[str, ...] = ("cdna",)
    barcode_length: int = 16
    umi_length: int = 10
    tso: str = "TTTCTTATATGGG"
    chemistry: str = "v2"
    output_mode: str = "paired"
    read_length: int = 300
    rc_fraction: float = 0.0
    platform: str = "illumina"
    variable_length: bool = True
    write_read_truth: bool = False
    seed: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["error_regions"] = list(self.error_regions)
        return d

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
```

`simplex/__init__.py`:
```python
from .config import SimplexConfig

__all__ = ["SimplexConfig", "run"]


def run(*args, **kwargs):
    from .run import run as _run

    return _run(*args, **kwargs)
```

Modify `pyproject.toml`: add `"simplex"` (and `"simplex.tests"` if packages are listed explicitly) to the package list. If it uses `find`/`packages.find`, ensure `simplex*` is included.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest simplex/tests/test_dna.py simplex/tests/test_config.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add simplex/ pyproject.toml
git commit -m "feat(simplex): scaffold package, DNA helpers, and SimplexConfig"
```

---

### Task 2: Barcode whitelist loader

**Files:**
- Create: `simplex/barcodes.py`, `simplex/tests/test_barcodes.py`

**Interfaces:**
- Consumes: `pairplex.utils.get_whitelist_path`.
- Produces: `barcodes.load_barcodes(chemistry: str, n: int, seed: int) -> list[str]` (n distinct whitelist barcodes).

- [ ] **Step 1: Write the failing test**

`simplex/tests/test_barcodes.py`:
```python
from simplex.barcodes import load_barcodes


def test_load_barcodes_distinct_and_valid():
    bcs = load_barcodes("v2", 500, seed=0)
    assert len(bcs) == 500
    assert len(set(bcs)) == 500
    assert all(len(b) == 16 and set(b) <= set("ACGT") for b in bcs)


def test_load_barcodes_reproducible():
    assert load_barcodes("v2", 50, seed=1) == load_barcodes("v2", 50, seed=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest simplex/tests/test_barcodes.py -q`
Expected: FAIL (no module `simplex.barcodes`).

- [ ] **Step 3: Write minimal implementation**

`simplex/barcodes.py`:
```python
import gzip
from pathlib import Path

import numpy as np

from pairplex.utils import get_whitelist_path

_CHEMISTRY = {"v2": "v2", "v3": "v3"}


def _read_whitelist(path: Path) -> list[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [line.strip() for line in f if line.strip()]


def load_barcodes(chemistry: str, n: int, seed: int) -> list[str]:
    """Return n distinct 10x barcodes sampled from the chemistry whitelist."""
    key = _CHEMISTRY.get(chemistry.lower(), chemistry)
    path = Path(get_whitelist_path(key))
    whitelist = _read_whitelist(path)
    if n > len(whitelist):
        raise ValueError(f"requested {n} barcodes but whitelist has {len(whitelist)}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(whitelist), size=n, replace=False)
    return [whitelist[i] for i in idx]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest simplex/tests/test_barcodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/barcodes.py simplex/tests/test_barcodes.py
git commit -m "feat(simplex): barcode whitelist loader"
```

---

### Task 3: load_pairs, droplet/barcode assignment, well assignment

**Files:**
- Create: `simplex/cells.py`, `simplex/tests/test_cells.py`

**Interfaces:**
- Consumes: `barcodes.load_barcodes`.
- Produces:
  - `cells.load_pairs(input_data, n_cells=None, seed=0) -> pl.DataFrame` with columns `cell_id, source_pair_id, chain0_id, chain0_seq, chain1_id, chain1_seq`.
  - `cells.assign_droplets_and_barcodes(cells, mean, sd, chemistry, seed) -> pl.DataFrame` (+`droplet_id, barcode`).
  - `cells.assign_wells(cells, wells, seed) -> pl.DataFrame` (+`well`).

- [ ] **Step 1: Write the failing test**

`simplex/tests/test_cells.py`:
```python
import numpy as np
import polars as pl
import pytest

from simplex.cells import assign_droplets_and_barcodes, assign_wells, load_pairs


def _fake_parquet(tmp_path, n=20):
    df = pl.DataFrame({
        "sequence_id:0": [f"h{i}" for i in range(n)],
        "sequence:0": ["ACGTACGTAC" for _ in range(n)],
        "sequence_id:1": [f"l{i}" for i in range(n)],
        "sequence:1": ["TTGGCCAATT" for _ in range(n)],
    })
    p = tmp_path / "pairs.parquet"
    df.write_parquet(p)
    return p


def test_load_pairs_maps_columns(tmp_path):
    p = _fake_parquet(tmp_path, 8)
    c = load_pairs(p)
    assert c.height == 8
    assert set(["cell_id", "chain0_id", "chain0_seq", "chain1_id", "chain1_seq"]) <= set(c.columns)
    assert c["chain0_id"].to_list()[0] == "h0"


def test_load_pairs_subsample(tmp_path):
    p = _fake_parquet(tmp_path, 100)
    assert load_pairs(p, n_cells=10, seed=0).height == 10


def test_droplets_share_barcode_within_droplet(tmp_path):
    c = load_pairs(_fake_parquet(tmp_path, 200), seed=0)
    c = assign_droplets_and_barcodes(c, mean=5, sd=1, chemistry="v2", seed=0)
    # every droplet_id maps to exactly one barcode
    per = c.group_by("droplet_id").agg(pl.col("barcode").n_unique().alias("nb"))
    assert per["nb"].max() == 1
    # distinct droplets have distinct barcodes
    assert c.select("barcode").n_unique() == c.select("droplet_id").n_unique()
    # overloading actually shares barcodes (fewer barcodes than cells)
    assert c.select("barcode").n_unique() < c.height


def test_assign_wells_range(tmp_path):
    c = load_pairs(_fake_parquet(tmp_path, 100), seed=0)
    c = assign_wells(c, wells=8, seed=0)
    assert c["well"].min() >= 0 and c["well"].max() <= 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest simplex/tests/test_cells.py -q`
Expected: FAIL (no module `simplex.cells`).

- [ ] **Step 3: Write minimal implementation**

`simplex/cells.py`:
```python
import numpy as np
import polars as pl

from .barcodes import load_barcodes


def load_pairs(input_data, n_cells=None, seed=0) -> pl.DataFrame:
    df = pl.read_parquet(input_data)
    rename = {
        "sequence_id:0": "chain0_id", "sequence:0": "chain0_seq",
        "sequence_id:1": "chain1_id", "sequence:1": "chain1_seq",
    }
    missing = [k for k in rename if k not in df.columns]
    if missing:
        raise ValueError(f"input parquet missing required columns: {missing}")
    df = df.select(list(rename.keys())).rename(rename)
    df = df.with_row_index("source_pair_id").with_columns(
        pl.col("source_pair_id").cast(pl.Utf8)
    )
    if n_cells is not None:
        rng = np.random.default_rng(seed)
        replace = n_cells > df.height
        idx = rng.choice(df.height, size=n_cells, replace=replace)
        df = df[idx]
    df = df.with_row_index("cell_id")
    return df.select(
        ["cell_id", "source_pair_id", "chain0_id", "chain0_seq", "chain1_id", "chain1_seq"]
    )


def assign_droplets_and_barcodes(cells, mean, sd, chemistry, seed) -> pl.DataFrame:
    rng = np.random.default_rng(seed + 10)
    n = cells.height
    order = rng.permutation(n)
    droplet_of_cell = np.empty(n, dtype=np.int64)
    idx, d = 0, 0
    while idx < n:
        size = max(1, int(round(rng.normal(mean, sd))))
        for _ in range(size):
            if idx >= n:
                break
            droplet_of_cell[order[idx]] = d
            idx += 1
        d += 1
    n_droplets = d
    barcodes = np.array(load_barcodes(chemistry, n_droplets, seed + 11))
    return cells.with_columns([
        pl.Series("droplet_id", droplet_of_cell),
        pl.Series("barcode", barcodes[droplet_of_cell]),
    ])


def assign_wells(cells, wells, seed) -> pl.DataFrame:
    rng = np.random.default_rng(seed + 12)
    return cells.with_columns(
        pl.Series("well", rng.integers(0, wells, size=cells.height).astype(np.int64))
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest simplex/tests/test_cells.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/cells.py simplex/tests/test_cells.py
git commit -m "feat(simplex): load pairs + droplet/barcode/well assignment"
```

---

### Task 4: Molecule generation and amplification

**Files:**
- Create: `simplex/molecules.py`, `simplex/tests/test_molecules.py`

**Interfaces:**
- Consumes: `cells` frame with `well, barcode` and `chain0_seq/chain1_seq`; `_dna.random_dna`.
- Produces:
  - `molecules.generate_molecules(cells, recovery_rate, molecules_per_chain_mean, umi_length, seed) -> pl.DataFrame` with `cell_id, true_cell_id, well, barcode, chain, umi, cdna`.
  - `molecules.amplify_and_sequence(molecules, reads_per_molecule_mean, seq_efficiency, seed) -> pl.DataFrame` (adds `read_id` + flag columns `is_ambient/is_leakage/is_index_hopped=False`, `n_errors=0`; one row per read).

- [ ] **Step 1: Write the failing test**

`simplex/tests/test_molecules.py`:
```python
import numpy as np
import polars as pl

from simplex.molecules import amplify_and_sequence, generate_molecules


def _cells(n=200):
    return pl.DataFrame({
        "cell_id": list(range(n)),
        "source_pair_id": [str(i) for i in range(n)],
        "chain0_id": [f"h{i}" for i in range(n)],
        "chain0_seq": ["AAAACCCCGGGGTTTT" for _ in range(n)],
        "chain1_id": [f"l{i}" for i in range(n)],
        "chain1_seq": ["TTTTGGGGCCCCAAAA" for _ in range(n)],
        "droplet_id": list(range(n)),
        "barcode": ["ACGTACGTACGTACGT" for _ in range(n)],
        "well": [0] * n,
    })


def test_recovery_rate_controls_captured_fraction():
    m = generate_molecules(_cells(2000), recovery_rate=0.5, molecules_per_chain_mean=5,
                           umi_length=10, seed=0)
    # fraction of (cell,chain) combos captured ~ 0.5
    captured = m.select(["cell_id", "chain"]).unique().height
    assert 0.4 * 2 * 2000 < captured < 0.6 * 2 * 2000


def test_full_recovery_both_chains_present():
    m = generate_molecules(_cells(50), recovery_rate=1.0, molecules_per_chain_mean=3,
                           umi_length=10, seed=1)
    assert set(m["chain"].unique().to_list()) == {0, 1}
    assert m.filter(pl.col("chain") == 0).height >= 50  # >=1 molecule per captured chain
    assert m["umi"].str.len_chars().max() == 10


def test_amplify_produces_reads_and_flags():
    m = generate_molecules(_cells(20), recovery_rate=1.0, molecules_per_chain_mean=3,
                           umi_length=10, seed=2)
    r = amplify_and_sequence(m, reads_per_molecule_mean=4, seq_efficiency=1.0, seed=2)
    assert r.height > m.height  # amplification
    assert r["read_id"].n_unique() == r.height
    for col in ["is_ambient", "is_leakage", "is_index_hopped"]:
        assert r[col].sum() == 0
    assert r["n_errors"].sum() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest simplex/tests/test_molecules.py -q`
Expected: FAIL (no module `simplex.molecules`).

- [ ] **Step 3: Write minimal implementation**

`simplex/molecules.py`:
```python
import numpy as np
import polars as pl

from ._dna import random_dna


def generate_molecules(cells, recovery_rate, molecules_per_chain_mean, umi_length, seed) -> pl.DataFrame:
    rng = np.random.default_rng(seed + 20)
    n = cells.height
    frames = []
    for chain in (0, 1):
        captured = rng.random(n) < recovery_rate
        nmol = rng.poisson(molecules_per_chain_mean, n)
        nmol = np.where(captured, np.maximum(nmol, 1), 0).astype(np.int64)
        rep = np.repeat(np.arange(n), nmol)
        if rep.size == 0:
            continue
        sub = cells[rep]
        umis = random_dna(rng, rep.size, umi_length)
        frames.append(pl.DataFrame({
            "cell_id": sub["cell_id"],
            "true_cell_id": sub["cell_id"],
            "well": sub["well"],
            "barcode": sub["barcode"],
            "chain": np.full(rep.size, chain, dtype=np.int8),
            "umi": umis,
            "cdna": sub[f"chain{chain}_seq"],
        }))
    return pl.concat(frames) if frames else cells.head(0).select([])


def amplify_and_sequence(molecules, reads_per_molecule_mean, seq_efficiency, seed) -> pl.DataFrame:
    rng = np.random.default_rng(seed + 21)
    m = molecules.height
    depth = rng.poisson(reads_per_molecule_mean, m)
    # thin by sequencing efficiency
    depth = rng.binomial(depth, seq_efficiency).astype(np.int64)
    rep = np.repeat(np.arange(m), depth)
    reads = molecules[rep]
    k = reads.height
    return reads.with_columns([
        pl.Series("read_id", [f"r{i}" for i in range(k)]),
        pl.lit(False).alias("is_ambient"),
        pl.lit(False).alias("is_leakage"),
        pl.lit(False).alias("is_index_hopped"),
        pl.lit(0, dtype=pl.Int64).alias("n_errors"),
    ])
```

> Note: `read_id` list-comprehension is the one acceptable O(k) Python pass; if profiling shows it dominates at 1M cells, replace with `pl.int_range` cast to Utf8. Keep simple for now.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest simplex/tests/test_molecules.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/molecules.py simplex/tests/test_molecules.py
git commit -m "feat(simplex): molecule generation + amplification"
```

---

### Task 5: Contamination — ambient soup, leakage, index hopping

**Files:**
- Create: `simplex/contamination.py`, `simplex/tests/test_contamination.py`

**Interfaces:**
- Consumes: `reads` frame from Task 4.
- Produces:
  - `contamination.inject_ambient(reads, ambient_rate, leakage_rate, seed) -> pl.DataFrame` — reassigns `barcode` for selected reads to another barcode **in the same well**; sets `is_ambient`/`is_leakage`. `true_cell_id` unchanged.
  - `contamination.index_hop(reads, index_hop_rate, wells, seed) -> pl.DataFrame` — reassigns `well` for selected reads to a different well; sets `is_index_hopped`.

- [ ] **Step 1: Write the failing test**

`simplex/tests/test_contamination.py`:
```python
import numpy as np
import polars as pl

from simplex.contamination import index_hop, inject_ambient


def _reads(n=4000):
    rng = np.random.default_rng(0)
    wells = rng.integers(0, 4, n)
    # 50 barcodes per well
    bc = np.array([f"BC{w}_{rng.integers(0,50)}" for w in wells])
    return pl.DataFrame({
        "read_id": [f"r{i}" for i in range(n)],
        "cell_id": rng.integers(0, 500, n),
        "true_cell_id": rng.integers(0, 500, n),
        "well": wells.astype(np.int64),
        "barcode": bc,
        "chain": rng.integers(0, 2, n).astype(np.int8),
        "umi": ["AAAAAAAAAA"] * n,
        "cdna": ["ACGT"] * n,
        "is_ambient": [False] * n,
        "is_leakage": [False] * n,
        "is_index_hopped": [False] * n,
        "n_errors": [0] * n,
    })


def test_ambient_rate_and_flag():
    r = inject_ambient(_reads(), ambient_rate=0.1, leakage_rate=0.0, seed=1)
    frac = r["is_ambient"].sum() / r.height
    assert 0.07 < frac < 0.13
    # ambient reads stay in same well (soup is per-well)
    assert r.filter(pl.col("is_ambient"))["well"].n_unique() <= 4


def test_leakage_flag():
    r = inject_ambient(_reads(), ambient_rate=0.0, leakage_rate=0.05, seed=2)
    frac = r["is_leakage"].sum() / r.height
    assert 0.03 < frac < 0.07


def test_index_hop_moves_wells():
    r = index_hop(_reads(), index_hop_rate=0.1, wells=4, seed=3)
    hopped = r.filter(pl.col("is_index_hopped"))
    assert 0.07 < hopped.height / r.height < 0.13
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest simplex/tests/test_contamination.py -q`
Expected: FAIL (no module `simplex.contamination`).

- [ ] **Step 3: Write minimal implementation**

`simplex/contamination.py`:
```python
import numpy as np
import polars as pl


def _reassign_barcode_within_well(reads, mask, rng) -> np.ndarray:
    """For rows in mask, pick a random *different* barcode present in the same well."""
    wells = reads["well"].to_numpy()
    barcodes = reads["barcode"].to_numpy().astype(object)
    # barcodes available per well
    by_well = {}
    for w, b in zip(wells, barcodes):
        by_well.setdefault(int(w), set()).add(b)
    by_well = {w: np.array(sorted(v)) for w, v in by_well.items()}
    new_bc = barcodes.copy()
    for i in np.nonzero(mask)[0]:
        pool = by_well[int(wells[i])]
        if pool.size <= 1:
            continue
        choice = pool[rng.integers(0, pool.size)]
        while choice == barcodes[i] and pool.size > 1:
            choice = pool[rng.integers(0, pool.size)]
        new_bc[i] = choice
    return new_bc


def inject_ambient(reads, ambient_rate, leakage_rate, seed) -> pl.DataFrame:
    rng = np.random.default_rng(seed + 30)
    n = reads.height
    u = rng.random(n)
    ambient_mask = u < ambient_rate
    leakage_mask = (u >= ambient_rate) & (u < ambient_rate + leakage_rate)
    move_mask = ambient_mask | leakage_mask
    new_bc = _reassign_barcode_within_well(reads, move_mask, rng)
    return reads.with_columns([
        pl.Series("barcode", new_bc.astype(str)),
        (pl.col("is_ambient") | pl.Series(ambient_mask)).alias("is_ambient"),
        (pl.col("is_leakage") | pl.Series(leakage_mask)).alias("is_leakage"),
    ])


def index_hop(reads, index_hop_rate, wells, seed) -> pl.DataFrame:
    rng = np.random.default_rng(seed + 31)
    n = reads.height
    mask = rng.random(n) < index_hop_rate
    cur = reads["well"].to_numpy()
    offset = rng.integers(1, wells, size=n)  # nonzero -> guarantees different well
    new_well = np.where(mask, (cur + offset) % wells, cur).astype(np.int64)
    return reads.with_columns([
        pl.Series("well", new_well),
        (pl.col("is_index_hopped") | pl.Series(mask)).alias("is_index_hopped"),
    ])
```

> Note: `_reassign_barcode_within_well` loops over contaminated rows only (a small fraction), so it stays cheap. If contamination fractions are large at scale, vectorize per-well later.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest simplex/tests/test_contamination.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/contamination.py simplex/tests/test_contamination.py
git commit -m "feat(simplex): ambient soup, leakage, and index hopping"
```

---

### Task 6: Sequencing errors (substitutions + indels + per-read override)

**Files:**
- Create: `simplex/errors.py`, `simplex/tests/test_errors.py`

**Interfaces:**
- Consumes: `reads` frame (mutates `barcode`/`umi`/`cdna` per `regions`).
- Produces: `errors.apply_sequencing_errors(reads, sub_rate, indel_rate, errors_per_read, regions, seed) -> pl.DataFrame` (updates chosen columns, adds to `n_errors`).

- [ ] **Step 1: Write the failing test**

`simplex/tests/test_errors.py`:
```python
import numpy as np
import polars as pl

from simplex.errors import apply_sequencing_errors


def _reads(n=2000, L=200):
    seq = "ACGT" * (L // 4)
    return pl.DataFrame({
        "read_id": [f"r{i}" for i in range(n)],
        "barcode": ["ACGTACGTACGTACGT"] * n,
        "umi": ["AAAAAAAAAA"] * n,
        "cdna": [seq] * n,
        "n_errors": [0] * n,
    })


def test_substitutions_change_bases_at_rate():
    r = apply_sequencing_errors(_reads(), sub_rate=0.05, indel_rate=0.0,
                                errors_per_read=None, regions=("cdna",), seed=0)
    orig = "ACGT" * 50
    diffs = [sum(a != b for a, b in zip(orig, s)) for s in r["cdna"].to_list()]
    mean_diffs = np.mean(diffs)
    assert 0.03 * 200 < mean_diffs < 0.07 * 200
    assert r["n_errors"].sum() > 0


def test_zero_rate_no_change():
    r = apply_sequencing_errors(_reads(), sub_rate=0.0, indel_rate=0.0,
                                errors_per_read=None, regions=("cdna",), seed=0)
    assert r["cdna"].to_list()[0] == "ACGT" * 50
    assert r["n_errors"].sum() == 0


def test_regions_respected():
    r = apply_sequencing_errors(_reads(), sub_rate=0.5, indel_rate=0.0,
                                errors_per_read=None, regions=("cdna",), seed=0)
    assert r["barcode"].to_list()[0] == "ACGTACGTACGTACGT"  # barcode untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest simplex/tests/test_errors.py -q`
Expected: FAIL (no module `simplex.errors`).

- [ ] **Step 3: Write minimal implementation**

`simplex/errors.py`:
```python
import numpy as np
import polars as pl

_BASES = np.array(list("ACGT"))


def _mutate_one(s: str, sub_rate: float, indel_rate: float,
                forced: int | None, rng: np.random.Generator) -> tuple[str, int]:
    chars = list(s)
    n_err = 0
    # substitutions
    if forced is not None:
        positions = rng.choice(len(chars), size=min(forced, len(chars)), replace=False)
        for p in positions:
            alt = rng.choice(_BASES)
            while alt == chars[p]:
                alt = rng.choice(_BASES)
            chars[p] = str(alt)
            n_err += 1
    elif sub_rate > 0:
        hit = rng.random(len(chars)) < sub_rate
        for p in np.nonzero(hit)[0]:
            alt = rng.choice(_BASES)
            while alt == chars[p]:
                alt = rng.choice(_BASES)
            chars[p] = str(alt)
            n_err += 1
    # indels
    if indel_rate > 0:
        out = []
        for ch in chars:
            u = rng.random()
            if u < indel_rate / 2:  # deletion
                n_err += 1
                continue
            out.append(ch)
            if u > 1 - indel_rate / 2:  # insertion
                out.append(str(rng.choice(_BASES)))
                n_err += 1
        chars = out
    return "".join(chars), n_err


def apply_sequencing_errors(reads, sub_rate, indel_rate, errors_per_read, regions, seed) -> pl.DataFrame:
    if sub_rate == 0 and indel_rate == 0 and not errors_per_read:
        return reads
    rng = np.random.default_rng(seed + 40)
    out_cols = {}
    total_err = np.zeros(reads.height, dtype=np.int64)
    forced_counts = (
        rng.poisson(errors_per_read, reads.height) if errors_per_read else [None] * reads.height
    )
    for region in regions:
        vals = reads[region].to_list()
        new_vals = []
        for i, s in enumerate(vals):
            ns, ne = _mutate_one(s, sub_rate, indel_rate, forced_counts[i], rng)
            new_vals.append(ns)
            total_err[i] += ne
        out_cols[region] = new_vals
    result = reads.with_columns([pl.Series(k, v) for k, v in out_cols.items()])
    return result.with_columns((pl.col("n_errors") + pl.Series(total_err)).alias("n_errors"))
```

> Note: error injection is the heaviest per-read Python work. It runs only on `regions` (default just `cdna`) and only when a rate is nonzero. For 1M-cell production runs this stage is the first candidate for numpy-vectorization (documented follow-up); correctness first.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest simplex/tests/test_errors.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/errors.py simplex/tests/test_errors.py
git commit -m "feat(simplex): layered sequencing error model"
```

---

### Task 7: build_reads (merged + paired, RC, variable length) with round-trip

**Files:**
- Create: `simplex/reads.py`, `simplex/tests/test_reads.py`

**Interfaces:**
- Consumes: `reads` frame; `_dna.revcomp_expr`; `config` values.
- Produces: `reads.build_reads(reads, output_mode, read_length, rc_fraction, barcode_length, umi_length, tso, variable_length, seed) -> pl.DataFrame`.
  - merged → columns `read_id, well, read_seq, qual`
  - paired → columns `read_id, well, r1_seq, r1_qual, r2_seq, r2_qual`

- [ ] **Step 1: Write the failing test**

`simplex/tests/test_reads.py`:
```python
import polars as pl

from simplex.reads import build_reads
from pairplex.utils import parse_barcodes  # for structure reference only


def _reads():
    return pl.DataFrame({
        "read_id": ["r0", "r1"],
        "well": [0, 1],
        "barcode": ["ACGTACGTACGTACGT", "TTTTCCCCAAAAGGGG"],
        "umi": ["AAAAAAAAAA", "CCCCCCCCCC"],
        "cdna": ["GATTACAGATTACA" * 20, "CCGGAATT" * 25],
    })


def test_merged_layout_parses_back():
    b = build_reads(_reads(), output_mode="merged", read_length=300, rc_fraction=0.0,
                    barcode_length=16, umi_length=10, tso="TTTCTTATATGGG",
                    variable_length=False, seed=0)
    s = b["read_seq"].to_list()[0]
    assert s[:16] == "ACGTACGTACGTACGT"
    assert s[16:26] == "AAAAAAAAAA"
    # cDNA recoverable exactly as pairplex does it (s[36:].lstrip('G'))
    assert s[36:].lstrip("G") == ("GATTACAGATTACA" * 20).lstrip("G")
    assert len(b["qual"].to_list()[0]) == len(s)


def test_paired_overlap_for_merge():
    b = build_reads(_reads(), output_mode="paired", read_length=300, rc_fraction=0.0,
                    barcode_length=16, umi_length=10, tso="TTTCTTATATGGG",
                    variable_length=False, seed=0)
    assert set(["r1_seq", "r2_seq", "r1_qual", "r2_qual"]) <= set(b.columns)
    r1 = b["r1_seq"].to_list()[0]
    assert r1[:16] == "ACGTACGTACGTACGT"
    assert len(r1) <= 300


def test_rc_fraction_all_flips_still_parses_via_rc():
    from pairplex.utils import correct_barcode, load_barcode_whitelist
    b = build_reads(_reads(), output_mode="merged", read_length=300, rc_fraction=1.0,
                    barcode_length=16, umi_length=10, tso="TTTCTTATATGGG",
                    variable_length=False, seed=0)
    s = b["read_seq"].to_list()[0]
    # forward no longer starts with barcode, but revcomp does
    from simplex._dna import revcomp_str
    assert revcomp_str(s)[:16] == "ACGTACGTACGTACGT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest simplex/tests/test_reads.py -q`
Expected: FAIL (no module `simplex.reads`).

- [ ] **Step 3: Write minimal implementation**

`simplex/reads.py`:
```python
import numpy as np
import polars as pl

from ._dna import revcomp_expr


def _maybe_truncate(reads, variable_length, seed):
    if not variable_length:
        return reads
    rng = np.random.default_rng(seed + 50)
    lens = reads["cdna"].str.len_chars().to_numpy()
    trim5 = rng.integers(0, np.maximum(1, lens // 10))
    trim3 = rng.integers(0, np.maximum(1, lens // 10))
    new_len = np.maximum(1, lens - trim5 - trim3).astype(np.int64)
    return reads.with_columns([
        pl.col("cdna").str.slice(pl.Series(trim5.astype(np.int64)), pl.Series(new_len)).alias("cdna")
    ])


def build_reads(reads, output_mode, read_length, rc_fraction, barcode_length,
                umi_length, tso, variable_length, seed) -> pl.DataFrame:
    reads = _maybe_truncate(reads, variable_length, seed)
    frag = pl.concat_str([pl.col("barcode"), pl.col("umi"), pl.lit(tso), pl.col("cdna")])
    reads = reads.with_columns(frag.alias("_frag"))

    if output_mode == "merged":
        rng = np.random.default_rng(seed + 51)
        is_rc = pl.Series(rng.random(reads.height) < rc_fraction)
        reads = reads.with_columns(is_rc.alias("_rc"))
        reads = reads.with_columns(
            pl.when(pl.col("_rc"))
            .then(revcomp_expr("_frag"))
            .otherwise(pl.col("_frag"))
            .alias("read_seq")
        )
        reads = reads.with_columns(
            pl.col("read_seq").str.replace_all(".", "I").alias("qual")
        )
        return reads.select(["read_id", "well", "read_seq", "qual"])

    # paired: R1 from 5' end, R2 = revcomp(frag) from 3' end
    reads = reads.with_columns([
        pl.col("_frag").str.slice(0, read_length).alias("r1_seq"),
        revcomp_expr("_frag").str.slice(0, read_length).alias("r2_seq"),
    ])
    reads = reads.with_columns([
        pl.col("r1_seq").str.replace_all(".", "I").alias("r1_qual"),
        pl.col("r2_seq").str.replace_all(".", "I").alias("r2_qual"),
    ])
    return reads.select(["read_id", "well", "r1_seq", "r1_qual", "r2_seq", "r2_qual"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest simplex/tests/test_reads.py -q`
Expected: PASS. (If `str.slice` rejects Series args in this polars version, wrap offset/length via `pl.col`-based expressions; tests will confirm.)

- [ ] **Step 5: Commit**

```bash
git add simplex/reads.py simplex/tests/test_reads.py
git commit -m "feat(simplex): assemble merged/paired reads with RC + variable length"
```

---

### Task 8: Ground-truth tables

**Files:**
- Create: `simplex/truth.py`, `simplex/tests/test_truth.py`

**Interfaces:**
- Consumes: `cells` (with `well, barcode`), `reads` (post-contamination, pre-build).
- Produces:
  - `truth.build_truth_cells(cells, reads) -> pl.DataFrame`
  - `truth.build_truth_barcodes(cells, reads) -> pl.DataFrame` with `well, barcode, true_cell_ids, n_true_cells, is_collision, dominant_cell, is_ambient_only`
  - `truth.build_truth_reads(reads) -> pl.DataFrame`

- [ ] **Step 1: Write the failing test**

`simplex/tests/test_truth.py`:
```python
import polars as pl

from simplex.truth import build_truth_barcodes, build_truth_cells, build_truth_reads


def _cells():
    return pl.DataFrame({
        "cell_id": [0, 1, 2],
        "source_pair_id": ["0", "1", "2"],
        "chain0_id": ["h0", "h1", "h2"], "chain0_seq": ["A", "A", "A"],
        "chain1_id": ["l0", "l1", "l2"], "chain1_seq": ["T", "T", "T"],
        "droplet_id": [0, 0, 1],
        "barcode": ["BC_X", "BC_X", "BC_Y"],  # cells 0 and 1 collide on BC_X
        "well": [0, 0, 0],
    })


def _reads():
    # cell 2's read leaked into BC_X (ambient); plus a purely-ambient BC_Z
    return pl.DataFrame({
        "read_id": ["r0", "r1", "r2", "r3"],
        "true_cell_id": [0, 1, 2, 2],
        "well": [0, 0, 0, 0],
        "barcode": ["BC_X", "BC_X", "BC_X", "BC_Z"],
        "chain": [0, 1, 1, 0],
        "is_ambient": [False, False, True, True],
        "is_leakage": [False, False, False, False],
        "is_index_hopped": [False, False, False, False],
        "n_errors": [0, 0, 0, 0],
    })


def test_truth_barcodes_collision_and_ambient_only():
    tb = build_truth_barcodes(_cells(), _reads())
    row = tb.filter((pl.col("well") == 0) & (pl.col("barcode") == "BC_X")).to_dicts()[0]
    assert row["n_true_cells"] == 2 and row["is_collision"] is True
    zrow = tb.filter(pl.col("barcode") == "BC_Z").to_dicts()[0]
    assert zrow["is_ambient_only"] is True and zrow["n_true_cells"] == 0


def test_truth_cells_shape():
    tc = build_truth_cells(_cells(), _reads())
    assert tc.height == 3
    assert "barcode" in tc.columns and "well" in tc.columns


def test_truth_reads_passthrough():
    assert build_truth_reads(_reads()).height == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest simplex/tests/test_truth.py -q`
Expected: FAIL (no module `simplex.truth`).

- [ ] **Step 3: Write minimal implementation**

`simplex/truth.py`:
```python
import polars as pl


def build_truth_cells(cells, reads) -> pl.DataFrame:
    counts = (
        reads.filter(~pl.col("is_ambient") & ~pl.col("is_leakage") & ~pl.col("is_index_hopped"))
        .group_by(["true_cell_id", "chain"])
        .agg(pl.len().alias("n"))
        .pivot(values="n", index="true_cell_id", on="chain")
        .rename({"true_cell_id": "cell_id"})
    )
    rename = {c: f"n_reads_chain{c}" for c in counts.columns if c != "cell_id"}
    counts = counts.rename(rename)
    return cells.join(counts, on="cell_id", how="left").fill_null(0)


def build_truth_barcodes(cells, reads) -> pl.DataFrame:
    # ground-truth occupancy: which real cells were assigned to each (well, barcode)
    occ = (
        cells.group_by(["well", "barcode"])
        .agg([
            pl.col("cell_id").alias("true_cell_ids"),
            pl.col("cell_id").n_unique().alias("n_true_cells"),
            pl.col("cell_id").first().alias("dominant_cell"),
        ])
    )
    # every (well, barcode) that shows up in reads
    seen = reads.select(["well", "barcode"]).unique()
    tb = seen.join(occ, on=["well", "barcode"], how="left")
    tb = tb.with_columns([
        pl.col("n_true_cells").fill_null(0),
        (pl.col("n_true_cells").fill_null(0) >= 2).alias("is_collision"),
        (pl.col("n_true_cells").fill_null(0) == 0).alias("is_ambient_only"),
    ])
    return tb


def build_truth_reads(reads) -> pl.DataFrame:
    keep = ["read_id", "well", "barcode", "umi", "true_cell_id", "chain",
            "is_ambient", "is_leakage", "is_index_hopped", "n_errors"]
    return reads.select([c for c in keep if c in reads.columns])
```

> Note: `dominant_cell` here is the first assigned cell; refine to max-read cell if the scorer needs it (deferred with the scorer).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest simplex/tests/test_truth.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/truth.py simplex/tests/test_truth.py
git commit -m "feat(simplex): ground-truth cell/barcode/read tables"
```

---

### Task 9: Output writers (streaming FASTQ per well + truth + config)

**Files:**
- Create: `simplex/io.py`, `simplex/tests/test_io.py`

**Interfaces:**
- Consumes: `built` frame (Task 7), truth frames (Task 8), `SimplexConfig`.
- Produces:
  - `io.write_fastq(built, output_directory, output_mode, platform="illumina", compress=True) -> list[Path]`
  - `io.write_truth(output_directory, truth_cells, truth_barcodes, truth_reads=None) -> None`

- [ ] **Step 1: Write the failing test**

`simplex/tests/test_io.py`:
```python
import gzip
from pathlib import Path

import polars as pl

from simplex.io import write_fastq, write_truth


def _built_merged():
    return pl.DataFrame({
        "read_id": ["r0", "r1"],
        "well": [0, 1],
        "read_seq": ["ACGTACGT", "TTTTGGGG"],
        "qual": ["IIIIIIII", "IIIIIIII"],
    })


def _built_paired():
    return pl.DataFrame({
        "read_id": ["r0"], "well": [0],
        "r1_seq": ["ACGT"], "r1_qual": ["IIII"],
        "r2_seq": ["TTGG"], "r2_qual": ["IIII"],
    })


def test_write_merged_one_file_per_well(tmp_path):
    paths = write_fastq(_built_merged(), tmp_path, "merged")
    names = sorted(p.name for p in paths)
    assert any("well000" in n for n in names) and any("well001" in n for n in names)
    content = gzip.open([p for p in paths if "well000" in p.name][0], "rt").read()
    assert content.startswith("@r0\nACGTACGT\n+\nIIIIIIII\n")


def test_write_paired_r1_r2_named(tmp_path):
    paths = write_fastq(_built_paired(), tmp_path, "paired")
    names = sorted(p.name for p in paths)
    assert any("_R1_001.fastq.gz" in n for n in names)
    assert any("_R2_001.fastq.gz" in n for n in names)


def test_write_truth(tmp_path):
    tc = pl.DataFrame({"cell_id": [0]})
    tb = pl.DataFrame({"well": [0], "barcode": ["BC"]})
    write_truth(tmp_path, tc, tb)
    assert (Path(tmp_path) / "truth" / "truth_cells.parquet").exists()
    assert (Path(tmp_path) / "truth" / "truth_barcodes.parquet").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest simplex/tests/test_io.py -q`
Expected: FAIL (no module `simplex.io`).

- [ ] **Step 3: Write minimal implementation**

`simplex/io.py`:
```python
import gzip
from pathlib import Path

import polars as pl


def _well_tag(w: int) -> str:
    return f"well{int(w):03d}"


def _write_fastq_records(fh, ids, seqs, quals):
    fh.write("".join(f"@{i}\n{s}\n+\n{q}\n" for i, s, q in zip(ids, seqs, quals)))


def write_fastq(built, output_directory, output_mode, platform="illumina", compress=True) -> list:
    reads_dir = Path(output_directory) / "reads"
    reads_dir.mkdir(parents=True, exist_ok=True)
    ext = "fastq.gz" if compress else "fastq"
    opener = (lambda p: gzip.open(p, "wt")) if compress else (lambda p: open(p, "w"))
    paths = []
    for (well,), sub in built.group_by(["well"], maintain_order=True):
        tag = _well_tag(well)
        if output_mode == "merged":
            p = reads_dir / f"{tag}.{ext}"
            with opener(p) as fh:
                _write_fastq_records(fh, sub["read_id"], sub["read_seq"], sub["qual"])
            paths.append(p)
        else:
            p1 = reads_dir / f"{tag}_S1_L001_R1_001.{ext}"
            p2 = reads_dir / f"{tag}_S1_L001_R2_001.{ext}"
            with opener(p1) as fh:
                _write_fastq_records(fh, sub["read_id"], sub["r1_seq"], sub["r1_qual"])
            with opener(p2) as fh:
                _write_fastq_records(fh, sub["read_id"], sub["r2_seq"], sub["r2_qual"])
            paths.extend([p1, p2])
    return paths


def write_truth(output_directory, truth_cells, truth_barcodes, truth_reads=None) -> None:
    tdir = Path(output_directory) / "truth"
    tdir.mkdir(parents=True, exist_ok=True)
    truth_cells.write_parquet(tdir / "truth_cells.parquet")
    truth_barcodes.write_parquet(tdir / "truth_barcodes.parquet")
    if truth_reads is not None:
        truth_reads.write_parquet(tdir / "truth_reads.parquet")
```

> Note: `truth_barcodes` may contain a list column (`true_cell_ids`) — parquet handles it. The Illumina-style `_S1_L001_R1_001` naming is confirmed against `abstar.pp.merge_fastqs` in Task 11's integration test (if it fails to pair files, adjust the token pattern here).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest simplex/tests/test_io.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/io.py simplex/tests/test_io.py
git commit -m "feat(simplex): streaming FASTQ + truth writers"
```

---

### Task 10: `run()` orchestration

**Files:**
- Create: `simplex/run.py`, `simplex/tests/test_run.py`

**Interfaces:**
- Consumes: every stage above + `SimplexConfig`.
- Produces: `run.run(input_data, output_directory, **knobs) -> Path` (returns `reads/` dir); writes reads, truth, `simplex_config.json`.

- [ ] **Step 1: Write the failing test**

`simplex/tests/test_run.py`:
```python
from pathlib import Path

import polars as pl

from simplex.run import run


def _fake_input(tmp_path, n=60):
    hi = "GATTACA" * 30
    lo = "CCGGTTAA" * 24
    df = pl.DataFrame({
        "sequence_id:0": [f"h{i}" for i in range(n)],
        "sequence:0": [hi] * n,
        "sequence_id:1": [f"l{i}" for i in range(n)],
        "sequence:1": [lo] * n,
    })
    p = tmp_path / "pairs.parquet"
    df.write_parquet(p)
    return p


def test_run_writes_reads_truth_and_config(tmp_path):
    inp = _fake_input(tmp_path)
    out = tmp_path / "out"
    reads_dir = run(input_data=inp, output_directory=out, wells=4,
                    cells_per_droplet_mean=1, cells_per_droplet_sd=0,
                    output_mode="merged", variable_length=False, seed=0)
    assert Path(reads_dir).is_dir()
    assert list(Path(reads_dir).glob("*.fastq.gz"))
    assert (out / "truth" / "truth_cells.parquet").exists()
    assert (out / "truth" / "truth_barcodes.parquet").exists()
    assert (out / "simplex_config.json").exists()


def test_run_reproducible(tmp_path):
    inp = _fake_input(tmp_path)
    r1 = run(input_data=inp, output_directory=tmp_path / "a", wells=4, seed=5, output_mode="merged")
    r2 = run(input_data=inp, output_directory=tmp_path / "b", wells=4, seed=5, output_mode="merged")
    import gzip
    def read_all(d):
        return sorted(gzip.open(p, "rt").read() for p in Path(d).glob("*.fastq.gz"))
    assert read_all(r1) == read_all(r2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest simplex/tests/test_run.py -q`
Expected: FAIL (no module `simplex.run`).

- [ ] **Step 3: Write minimal implementation**

`simplex/run.py`:
```python
from pathlib import Path

from .cells import assign_droplets_and_barcodes, assign_wells, load_pairs
from .config import SimplexConfig
from .contamination import index_hop, inject_ambient
from .errors import apply_sequencing_errors
from .io import write_fastq, write_truth
from .molecules import amplify_and_sequence, generate_molecules
from .reads import build_reads
from .truth import build_truth_barcodes, build_truth_cells, build_truth_reads


def run(input_data, output_directory, **knobs) -> Path:
    cfg = SimplexConfig(input_data=str(input_data), output_directory=str(output_directory), **knobs)
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)

    cells = load_pairs(cfg.input_data, cfg.n_cells, cfg.seed)
    cells = assign_droplets_and_barcodes(
        cells, cfg.cells_per_droplet_mean, cfg.cells_per_droplet_sd, cfg.chemistry, cfg.seed
    )
    cells = assign_wells(cells, cfg.wells, cfg.seed)

    molecules = generate_molecules(
        cells, cfg.recovery_rate, cfg.molecules_per_chain_mean, cfg.umi_length, cfg.seed
    )
    reads = amplify_and_sequence(molecules, cfg.reads_per_molecule_mean, cfg.seq_efficiency, cfg.seed)
    reads = inject_ambient(reads, cfg.ambient_rate, cfg.leakage_rate, cfg.seed)
    reads = index_hop(reads, cfg.index_hop_rate, cfg.wells, cfg.seed)
    reads = apply_sequencing_errors(
        reads, cfg.sub_rate, cfg.indel_rate, cfg.errors_per_read, cfg.error_regions, cfg.seed
    )

    truth_cells = build_truth_cells(cells, reads)
    truth_barcodes = build_truth_barcodes(cells, reads)
    truth_reads = build_truth_reads(reads) if cfg.write_read_truth else None

    built = build_reads(
        reads, cfg.output_mode, cfg.read_length, cfg.rc_fraction, cfg.barcode_length,
        cfg.umi_length, cfg.tso, cfg.variable_length, cfg.seed
    )

    write_fastq(built, out, cfg.output_mode, cfg.platform)
    write_truth(out, truth_cells, truth_barcodes, truth_reads)
    cfg.to_json(out / "simplex_config.json")
    return out / "reads"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest simplex/tests/test_run.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add simplex/run.py simplex/tests/test_run.py
git commit -m "feat(simplex): run() orchestration end to end"
```

---

### Task 11: Golden integration + degradation tests (SimPlex → PairPlex)

**Files:**
- Create: `simplex/tests/test_integration.py`
- Create: `simplex/scoring.py` (minimal sequence-based scorer helper used by the test; the full scorer is a later sub-project)

**Interfaces:**
- Consumes: `simplex.run`, `pairplex.run`, truth parquet, `truth_cells`.
- Produces: `scoring.score_pairs(paired_parquet, truth_cells) -> dict` with keys `n_pairs, correct, mispaired, false_positive` (matched by **sequence**, not junction).

- [ ] **Step 1: Write the failing test**

`simplex/tests/test_integration.py`:
```python
import polars as pl
import pytest

from simplex.run import run
from simplex.scoring import score_pairs


def _abstar_input(tmp_path, n=24):
    """Use abstar's bundled paired bnAbs as realistic input (H and L that share a name)."""
    import os, abstar
    from abutils.io import parse_fastx
    td = os.path.dirname(abstar.__file__) + "/test_data"
    hcs = {s.id: s.sequence for s in parse_fastx(td + "/test_hiv_bnab_hcs.fasta")}
    lcs = {s.id: s.sequence for s in parse_fastx(td + "/test_hiv_bnab_lcs.fasta")}
    names = [x for x in hcs if x in lcs][:n]
    df = pl.DataFrame({
        "sequence_id:0": names,
        "sequence:0": [hcs[x] for x in names],
        "sequence_id:1": names,
        "sequence:1": [lcs[x] for x in names],
    })
    p = tmp_path / "real_pairs.parquet"
    df.write_parquet(p)
    return p


def test_clean_run_pairs_perfectly(tmp_path):
    import pairplex
    inp = _abstar_input(tmp_path)
    out = tmp_path / "sim"
    reads_dir = run(
        input_data=inp, output_directory=out, wells=2,
        cells_per_droplet_mean=1, cells_per_droplet_sd=0,       # no barcode sharing
        recovery_rate=1.0, molecules_per_chain_mean=4, reads_per_molecule_mean=4,
        seq_efficiency=1.0, ambient_rate=0.0, leakage_rate=0.0, index_hop_rate=0.0,
        sub_rate=0.0, indel_rate=0.0, variable_length=False, rc_fraction=0.0,
        output_mode="merged", seed=0,
    )
    pp_out = tmp_path / "pp"
    pairplex.run(sequences=str(reads_dir), output_directory=str(pp_out),
                 clustering_threshold=0.9, min_cluster_reads=3, min_cluster_umis=1,
                 quiet=True)
    truth_cells = pl.read_parquet(out / "truth" / "truth_cells.parquet")
    total = {"correct": 0, "mispaired": 0, "n_pairs": 0}
    for pf in (pp_out / "annotated").glob("*_paired.parquet"):
        s = score_pairs(pf, truth_cells)
        for k in total:
            total[k] += s[k]
    assert total["n_pairs"] > 0
    assert total["mispaired"] == 0
    assert total["correct"] == total["n_pairs"]


def test_ambient_introduces_mispairs(tmp_path):
    import pairplex
    inp = _abstar_input(tmp_path)
    out = tmp_path / "sim2"
    reads_dir = run(
        input_data=inp, output_directory=out, wells=1,
        cells_per_droplet_mean=1, cells_per_droplet_sd=0,
        recovery_rate=0.6, molecules_per_chain_mean=6, reads_per_molecule_mean=5,
        seq_efficiency=1.0, ambient_rate=0.15, leakage_rate=0.05, index_hop_rate=0.0,
        sub_rate=0.0, variable_length=False, output_mode="merged", seed=1,
    )
    pp_out = tmp_path / "pp2"
    pairplex.run(sequences=str(reads_dir), output_directory=str(pp_out),
                 min_cluster_reads=3, min_cluster_umis=1, min_cluster_fraction=0.0, quiet=True)
    truth_cells = pl.read_parquet(out / "truth" / "truth_cells.parquet")
    mis = sum(score_pairs(pf, truth_cells)["mispaired"]
              for pf in (pp_out / "annotated").glob("*_paired.parquet"))
    assert mis > 0  # ambient + weak filters produce wrong pairs, as investigated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest simplex/tests/test_integration.py -q`
Expected: FAIL (no module `simplex.scoring`).

- [ ] **Step 3: Write minimal implementation**

`simplex/scoring.py`:
```python
import polars as pl


def _origin_by_sequence(seq: str, seq_to_cell: dict) -> set:
    """Return the set of true cell_ids whose chain0/chain1 sequence contains `seq`."""
    hits = set()
    for full, cid in seq_to_cell.items():
        if seq and (seq in full or full.endswith(seq)):
            hits.add(cid)
    return hits


def score_pairs(paired_parquet, truth_cells) -> dict:
    """Score a PairPlex *_paired.parquet against SimPlex truth, matching by SEQUENCE."""
    df = pl.read_parquet(paired_parquet)
    seq_to_cell = {}
    for r in truth_cells.iter_rows(named=True):
        seq_to_cell[r["chain0_seq"]] = r["cell_id"]
        seq_to_cell[r["chain1_seq"]] = r["cell_id"]
    n_pairs = correct = mispaired = false_positive = 0
    for r in df.to_dicts():
        s0, s1 = r.get("sequence:0"), r.get("sequence:1")
        c0 = _origin_by_sequence(s0 or "", seq_to_cell)
        c1 = _origin_by_sequence(s1 or "", seq_to_cell)
        n_pairs += 1
        if not c0 or not c1:
            false_positive += 1
        elif c0 & c1:
            correct += 1
        else:
            mispaired += 1
    return {"n_pairs": n_pairs, "correct": correct,
            "mispaired": mispaired, "false_positive": false_positive}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest simplex/tests/test_integration.py -q`
Expected: PASS. If PairPlex's `merge_fastqs` (paired mode) can't pair files, this test uses `output_mode="merged"` so it exercises the pairing core; a separate paired-mode merge check is a follow-up once file naming is confirmed against `abstar.pp.merge_fastqs`.

- [ ] **Step 5: Commit**

```bash
git add simplex/scoring.py simplex/tests/test_integration.py
git commit -m "test(simplex): golden clean-pairing + ambient-degradation integration"
```

---

## Self-Review

**Spec coverage:**
- Packaging sibling `simplex/` → Task 1. ✓
- Staged vectorized pipeline → Tasks 3–7, 10. ✓
- Knobs/`run()` signature → `SimplexConfig` (Task 1) + `run` (Task 10). ✓
- load_pairs / droplets+barcodes (overloading) / wells → Task 3. ✓
- molecules + recovery/dropout + UMIs + amplification/efficiency → Task 4. ✓
- ambient soup + leakage + ambient-only barcodes → Task 5 + truth Task 8. ✓
- index hopping → Task 5. ✓
- layered errors (sub+indel+per-read+regions) → Task 6. ✓
- build_reads merged/paired + RC + variable length + parse round-trip → Task 7. ✓
- ground truth (cells/barcodes/reads) → Task 8. ✓
- output layout + Illumina naming + streaming writes → Task 9. ✓
- reproducibility → Task 10 test. ✓
- golden clean-pairing (sequence-scored) + degradation → Task 11. ✓
- Deferred (PCR chimeras, scorer, sweep) → out of scope, noted. ✓

**Placeholder scan:** No TBD/TODO in steps; every code step has concrete code; notes flag known follow-ups (perf vectorization, merge_fastqs naming) without leaving gaps in the deliverable.

**Type consistency:** Column schemas fixed in the header and used consistently (`cell_id, true_cell_id, well, barcode, chain, umi, cdna, read_id, is_ambient, is_leakage, is_index_hopped, n_errors`). Function signatures in each Task's Interfaces block match their call sites in `run.py` (Task 10). `revcomp_expr`/`revcomp_str` names consistent across Tasks 1 and 7.

**Known execution risks to watch (not gaps):**
- polars `str.slice` with Series offset/length (Task 7) — if the installed API differs, use expression-based slicing; tests will catch it.
- `pl.DataFrame.group_by([...])` tuple-key unpacking in Task 9 — matches polars 1.39.
- `merge_fastqs` filename tokens (Task 9/11) — verify against source; adjust pattern if needed.
