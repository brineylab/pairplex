"""Golden-output helpers: run PairPlex on a tiny fixed input and reduce outputs to comparable tables.

The input (``data/golden/input``) is two simulated wells, each under 1000 reads. That is a single
1000-read chunk per well, so read order into clustering is deterministic. The expected tables
(``data/golden/expected``) were produced by the reference code named in ``versions.json``;
regenerate them only on purpose, with ``make_golden.py``.
"""

from __future__ import annotations

import json
import os
import shutil
from importlib import metadata
from pathlib import Path

import polars as pl

GOLDEN = Path(__file__).parent / "data" / "golden"
INPUT = GOLDEN / "input"
EXPECTED = GOLDEN / "expected"

META_COLS = ["name", "reads", "umis", "cluster_fraction", "consensus", "pass_filters"]
UNPAIRED_COLS = ["sequence_id", "sequence", "locus", "productive", "v_gene", "j_gene", "junction", "cdr3_aa"]
PAIRED_COLS = ["name", "sequence_id:0", "sequence_id:1", "locus:0", "locus:1"]
# annotation calls that may legitimately change with the abstar version / germline database
ANNOTATION_CALLS = ["productive", "v_gene", "j_gene", "junction", "cdr3_aa"]


def run_pairplex(output_directory: Path, **kwargs) -> Path:
    """Run PairPlex on a copy of the golden input with 2 workers; returns the output directory."""
    import pairplex.pairplex as ppx

    work = Path(output_directory)
    inp = work / "input"
    shutil.copytree(INPUT, inp)
    old = os.environ.get("PAIRPLEX_MAX_WORKERS")
    os.environ["PAIRPLEX_MAX_WORKERS"] = "2"
    try:
        ppx.run(str(inp), str(work / "out"), quiet=True, **kwargs)
    finally:
        if old is None:
            os.environ.pop("PAIRPLEX_MAX_WORKERS", None)
        else:
            os.environ["PAIRPLEX_MAX_WORKERS"] = old
    return work / "out"


def _well(path: Path) -> str:
    return path.name.split(".")[0]


def extract(out: Path) -> dict[str, pl.DataFrame]:
    """Reduce a PairPlex output directory to order-independent tables."""
    out = Path(out)
    meta, cons, unp, par = [], [], [], []
    for f in sorted((out / "metadata").glob("*.csv")):
        d = pl.read_csv(f)
        if d.height:
            meta.append(d.select(META_COLS).with_columns(pl.lit(_well(f)).alias("well"),
                                                         pl.col("cluster_fraction").round(6)))
    for f in sorted((out / "consensus").glob("*.fasta")):
        name = None
        for line in f.read_text().splitlines():
            if line.startswith(">"):
                name = line[1:].strip()
            elif name:
                cons.append({"well": _well(f), "name": name, "sequence": line.strip()})
                name = None
    for f in sorted((out / "annotated").glob("*_unpaired.parquet")):
        d = pl.read_parquet(f)
        if d.height:
            unp.append(d.select([c for c in UNPAIRED_COLS if c in d.columns]).with_columns(pl.lit(_well(f)).alias("well")))
    for f in sorted((out / "annotated").glob("*_paired.parquet")):
        d = pl.read_parquet(f)
        if d.height:
            par.append(d.select(PAIRED_COLS).with_columns(pl.lit(_well(f)).alias("well")))

    def cat(frames, keys):
        return pl.concat(frames, how="vertical_relaxed").sort(keys) if frames else pl.DataFrame()

    return {
        "metadata": cat(meta, ["well", "name"]),
        "consensus": pl.DataFrame(cons).sort(["well", "name"]) if cons else pl.DataFrame(),
        "unpaired": cat(unp, ["well", "sequence_id"]),
        "paired": cat(par, ["well", "name"]),
    }


def abstar_version() -> str | None:
    try:
        return metadata.version("abstar")
    except metadata.PackageNotFoundError:
        return None


def load_expected() -> tuple[dict[str, pl.DataFrame], dict]:
    tables = {p.stem: pl.read_parquet(p) for p in EXPECTED.glob("*.parquet")}
    versions = json.loads((EXPECTED / "versions.json").read_text())
    return tables, versions
