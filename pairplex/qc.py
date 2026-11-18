# Copyright (c) 2025 Benjamin Nemoz
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""
Run QC for ``pairplex run``: support columns, per-stage drop counts, abstar failure table and a
per-run QC report.

Everything here counts from data ``run()`` already holds (the parsed reads of a well, the cluster
metadata, the abstar results, the assigned pairs) or reads back a few columns of the per-well
outputs at the end of the run. Nothing here changes which pairs are emitted.

Outputs (under ``<output_directory>/qc/``):

- ``stage_counts.csv``: one row per well plus a ``TOTAL`` row, columns ``STAGE_FIELDS``;
- ``annotation_failures.csv``: abstar failure / exclusion modes with counts per well and in total;
- ``report.json`` and ``report.md``: run-level QC metrics (see ``build_report``).
"""

from __future__ import annotations

import csv
import json
import math
import re
import warnings
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import polars as pl

# --------------------------------------------------------------------------------------------
#   per-stage counts
# --------------------------------------------------------------------------------------------

# 0H0L = a barcode whose contigs all failed annotation (no locus); "other" = any other combination
COMPOSITIONS = ["1H1L", "1H0L", "0H1L", "1H2L", "2H1L", "2H2L", "0H0L", "other"]

STAGE_FIELDS = [
    # reads
    "input_reads",
    "valid_barcode_reads",
    "dropped_orientation_or_barcode",
    # barcode groups
    "barcode_groups",
    "groups_dropped_min_cluster_reads",
    "reads_in_dropped_groups",
    "empty_sequences_skipped",
    # clusters
    "clusters",
    "clusters_fail_reads",
    "clusters_fail_umis",
    "clusters_fail_fraction",
    "clusters_pass",
    # annotation
    "consensus_to_annotation",
    "annotation_returned",
    "annotated",
    "failed_annotation",
    "failed_unassigned",
    "failed_record_error",
    "productive",
    "unproductive",
    "no_junction",
    # pairing
    "barcodes_with_chains",
    *[f"barcodes_{c}" for c in COMPOSITIONS],
    "emitted_pairs",
]


def new_counts(well: str) -> dict:
    """An all-zero stage-count row for ``well``."""
    return {"well": well, **{f: 0 for f in STAGE_FIELDS}}


def read_stage_counts(input_reads: int, parsed: pl.DataFrame) -> dict:
    """Reads in, reads with a valid barcode, reads dropped (no orientation gave a valid barcode)."""
    valid = int(parsed.height)
    return {
        "input_reads": int(input_reads),
        "valid_barcode_reads": valid,
        "dropped_orientation_or_barcode": int(input_reads) - valid,
    }


def barcode_stage_counts(parsed: pl.DataFrame, min_cluster_reads: int) -> dict:
    """
    Barcode groups, groups (and their reads) dropped by ``min_cluster_reads``, and the empty
    sequences in kept groups, which ``greedy_clustering`` skips silently.
    """
    if parsed.height == 0:
        return {"barcode_groups": 0, "groups_dropped_min_cluster_reads": 0,
                "reads_in_dropped_groups": 0, "empty_sequences_skipped": 0}
    g = parsed.group_by("barcode").agg(
        pl.len().alias("n"),
        (pl.col("sequence").fill_null("").str.len_chars() == 0).sum().alias("empty"),
    )
    dropped = g.filter(pl.col("n") < min_cluster_reads)
    kept = g.filter(pl.col("n") >= min_cluster_reads)
    return {
        "barcode_groups": int(g.height),
        "groups_dropped_min_cluster_reads": int(dropped.height),
        "reads_in_dropped_groups": int(dropped["n"].sum()),
        "empty_sequences_skipped": int(kept["empty"].sum()),
    }


def cluster_stage_counts(
    metadata: pl.DataFrame,
    min_cluster_reads: int,
    min_cluster_umis: int,
    min_cluster_fraction: float,
) -> dict:
    """Clusters and clusters failing each filter (a cluster can fail several)."""
    if metadata.height == 0:
        return {"clusters": 0, "clusters_fail_reads": 0, "clusters_fail_umis": 0,
                "clusters_fail_fraction": 0, "clusters_pass": 0}
    return {
        "clusters": int(metadata.height),
        "clusters_fail_reads": int((metadata["reads"] < min_cluster_reads).sum()),
        "clusters_fail_umis": int((metadata["umis"] < min_cluster_umis).sum()),
        "clusters_fail_fraction": int((metadata["cluster_fraction"] < min_cluster_fraction).sum()),
        "clusters_pass": int(metadata["pass_filters"].sum()),
    }


# columns pulled from abstar results (kept small: this is per consensus sequence)
ANN_SCHEMA = {
    "sequence_id": pl.String,
    "annotation_status": pl.String,
    "failure_reason": pl.String,
    "locus": pl.String,
    "productive": pl.Boolean,
    "productivity_issues": pl.String,
    "junction": pl.String,
    "complete_vdj": pl.Boolean,
    "v_germline_start": pl.Int64,
}


def annotation_frame(sequences) -> pl.DataFrame:
    """A small polars frame (``ANN_SCHEMA`` columns) from abstar's returned ``Sequence`` objects."""
    if sequences is None:
        sequences = []
    elif not isinstance(sequences, list | tuple):
        sequences = [sequences]
    rows = []
    for s in sequences:
        ann = s.annotations or {}
        rows.append({c: ann.get(c) for c in ANN_SCHEMA})
    return pl.DataFrame(rows, schema=ANN_SCHEMA, strict=False)


def annotation_stage_counts(ann: pl.DataFrame, n_sent: int, n_record_failures: int = 0) -> dict:
    """
    Consensus sequences sent to abstar and what came back.

    ``failed_annotation`` = ``failed_unassigned`` (returned with ``annotation_status`` other than
    ``annotated``) + ``failed_record_error`` (no row returned: abstar record failures).
    ``unproductive`` and ``no_junction`` are counted among annotated sequences.
    """
    annotated = ann.filter(pl.col("annotation_status") == "annotated")
    unassigned = ann.height - annotated.height
    record = max(int(n_record_failures), int(n_sent) - ann.height)
    return {
        "consensus_to_annotation": int(n_sent),
        "annotation_returned": int(ann.height),
        "annotated": int(annotated.height),
        "failed_unassigned": int(unassigned),
        "failed_record_error": int(record),
        "failed_annotation": int(unassigned + record),
        "productive": int(annotated["productive"].fill_null(False).sum()),
        "unproductive": int((~annotated["productive"].fill_null(False)).sum()),
        "no_junction": int(annotated["junction"].is_null().sum()),
    }


def composition_label(n_heavy: int, n_light: int) -> str:
    label = f"{n_heavy}H{n_light}L"
    return label if label in COMPOSITIONS else "other"


def chain_composition_counts(pairs) -> dict:
    """Barcodes by chain composition, from ``abutils.tl.assign_pairs`` output (before the 1H1L rule)."""
    c = Counter(composition_label(len(p.heavies), len(p.lights)) for p in pairs)
    out = {f"barcodes_{k}": int(c.get(k, 0)) for k in COMPOSITIONS}
    out["barcodes_with_chains"] = int(sum(c.values()))
    return out


def stage_counts_frame(rows: list[dict]) -> pl.DataFrame:
    """Per-well rows plus a ``TOTAL`` row."""
    schema = {"well": pl.String, **{f: pl.Int64 for f in STAGE_FIELDS}}
    df = pl.DataFrame(rows, schema=schema, strict=False).fill_null(0)
    total = df.select(pl.lit("TOTAL").alias("well"), *[pl.col(f).sum() for f in STAGE_FIELDS])
    return pl.concat([df, total], how="vertical_relaxed")


# --------------------------------------------------------------------------------------------
#   abstar failure modes
# --------------------------------------------------------------------------------------------

_FAILURES_TSV = re.compile(r"see (\S*failures\.tsv)")


@contextmanager
def capture_abstar_failures():
    """
    Collect abstar record failures (records abstar drops, with no output row).

    abstar reports them only through a ``RuntimeWarning`` pointing at a ``failures.tsv`` in its
    log directory. The warning is re-emitted unchanged; the parsed rows (``stage``, ``category``,
    ``exception_type``, ``message``) are appended to the yielded list.
    """
    failures: list[dict] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield failures
    for w in caught:
        m = _FAILURES_TSV.search(str(w.message))
        if m and issubclass(w.category, RuntimeWarning):
            failures.extend(read_failures_tsv(m.group(1)))
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


def read_failures_tsv(path: str | Path) -> list[dict]:
    keep = ("sequence_id", "stage", "category", "exception_type", "message")
    try:
        with open(path, newline="") as f:
            return [{k: r.get(k) for k in keep} for r in csv.DictReader(f, delimiter="\t")]
    except OSError:
        return []


FAILURE_COLUMNS = {"well": pl.String, "kind": pl.String, "mode": pl.String, "count": pl.Int64}


def annotation_failure_rows(well: str, ann: pl.DataFrame, record_failures: list[dict]) -> list[dict]:
    """
    Failure / exclusion modes for one well. ``kind`` is one of:

    - ``record_error``: abstar dropped the record (mode = ``stage/category exception: message``);
    - ``unassigned``: returned but not annotated (mode = abstar ``failure_reason``);
    - ``unproductive``: annotated but unproductive (mode = each ``productivity_issues`` item, so a
      sequence with several issues counts once per issue);
    - ``no_junction``: annotated with no junction.
    """
    c: Counter = Counter()
    for r in record_failures:
        exc = f" {r['exception_type']}" if r.get("exception_type") else ""
        c[("record_error", f"{r.get('stage')}/{r.get('category')}{exc}: {r.get('message')}"[:200])] += 1
    annotated = ann.filter(pl.col("annotation_status") == "annotated")
    for reason in ann.filter(pl.col("annotation_status") != "annotated")["failure_reason"].to_list():
        c[("unassigned", reason or "unknown")] += 1
    unprod = annotated.filter(~pl.col("productive").fill_null(False))
    for issues in unprod["productivity_issues"].to_list():
        items = [i.strip() for i in (issues or "").split("|") if i.strip()] or ["unspecified"]
        for i in items:
            c[("unproductive", i)] += 1
    n_nojunc = int(annotated["junction"].is_null().sum())
    if n_nojunc:
        c[("no_junction", "junction is null")] += n_nojunc
    return [{"well": well, "kind": k, "mode": m, "count": n} for (k, m), n in sorted(c.items())]


def failure_table(rows: list[dict]) -> pl.DataFrame:
    """Per-well failure rows plus ``TOTAL`` rows (one per kind/mode)."""
    df = pl.DataFrame(rows, schema=FAILURE_COLUMNS)
    if df.height == 0:
        return df
    total = df.group_by("kind", "mode").agg(pl.col("count").sum()).with_columns(pl.lit("TOTAL").alias("well"))
    total = total.select(list(FAILURE_COLUMNS)).sort(["kind", "count"], descending=[False, True])
    return pl.concat([df, total])


# --------------------------------------------------------------------------------------------
#   fragment fraction
# --------------------------------------------------------------------------------------------

# a contig is "truncated" when its V alignment starts this many nt or more into the germline V
# (i.e. most of FR1 is missing) or when it has no junction
TRUNCATION_NT = 30


def fragment_counts(ann: pl.DataFrame, n_record_failures: int = 0) -> dict:
    """
    Per-locus counts of passing consensus sequences: ``{locus: {"n", "truncated", "incomplete_vdj"}}``,
    with failed annotations (unassigned + record errors) under ``"unknown"``.
    """
    out: dict = {}
    failed = ann.filter(pl.col("annotation_status") != "annotated").height + int(n_record_failures)
    if failed:
        out["unknown"] = {"n": failed, "truncated": failed, "incomplete_vdj": failed}
    annotated = ann.filter(pl.col("annotation_status") == "annotated").with_columns(
        ((pl.col("v_germline_start").fill_null(10**6) > TRUNCATION_NT) | pl.col("junction").is_null()).alias("_trunc"),
        (~pl.col("complete_vdj").fill_null(False)).alias("_inc"),
    )
    for r in annotated.group_by(pl.col("locus").fill_null("unknown")).agg(
        pl.len().alias("n"), pl.col("_trunc").sum().alias("truncated"), pl.col("_inc").sum().alias("incomplete_vdj")
    ).iter_rows(named=True):
        d = out.setdefault(r["locus"], {"n": 0, "truncated": 0, "incomplete_vdj": 0})
        for k in ("n", "truncated", "incomplete_vdj"):
            d[k] += int(r[k])
    return out


def merge_fragment_counts(acc: dict, new: dict) -> dict:
    for locus, d in new.items():
        a = acc.setdefault(locus, {"n": 0, "truncated": 0, "incomplete_vdj": 0})
        for k, v in d.items():
            a[k] += v
    return acc


def fragment_summary(frag: dict) -> dict:
    n = sum(d["n"] for d in frag.values())
    bad = sum(d["truncated"] for d in frag.values())
    by_locus = {
        locus: {**d, "fragment_fraction": _ratio(d["truncated"], d["n"])} for locus, d in sorted(frag.items())
    }
    return {
        "definition": (
            "share of passing consensus sequences that fail annotation or are truncated "
            f"(V alignment starts > {TRUNCATION_NT} nt into germline V, or no junction); "
            "failed annotations have no locus ('unknown')"
        ),
        "passing_consensus": n,
        "fragment_fraction": _ratio(bad, n),
        "failed_annotation_fraction": _ratio(frag.get("unknown", {}).get("n", 0), n),
        "by_locus": by_locus,
    }


# --------------------------------------------------------------------------------------------
#   UMI error signature
# --------------------------------------------------------------------------------------------


def _hamming1_pairs(u: pl.DataFrame) -> tuple[int, int]:
    """(Hamming-1 UMI pairs within barcode, distinct UMIs with >=1 Hamming-1 neighbour) for unique (barcode, umi)."""
    if u.height < 2:
        return 0, 0
    L = int(u["umi"].str.len_chars().max())
    keyed = pl.concat(
        [
            u.select(
                "barcode", "umi", pl.lit(i, pl.Int8).alias("pos"),
                (pl.col("umi").str.slice(0, i) + pl.col("umi").str.slice(i + 1)).alias("key"),
            )
            for i in range(L)
        ]
    )
    g = keyed.group_by("barcode", "pos", "key").agg(pl.len().alias("k"), pl.col("umi"))
    g = g.filter(pl.col("k") >= 2)
    if g.height == 0:
        return 0, 0
    pairs = int((g["k"] * (g["k"] - 1) // 2).sum())
    with_nb = g.select("barcode", "umi").explode("umi").unique().height
    return pairs, with_nb


def umi_hamming1_stats(parsed: pl.DataFrame, max_umis: int = 200_000, seed: int = 0) -> dict:
    """
    Hamming-1 UMI pairs within barcode, observed vs a null that shuffles barcodes across the
    distinct (barcode, UMI) rows (keeps each barcode's UMI count and the UMI composition).

    Exact (no pairwise loop): each Hamming-1 pair shares exactly one "UMI with position i
    masked" key. Barcodes are subsampled when a well has more than ``max_umis`` distinct
    (barcode, UMI) rows.
    """
    empty = {"umis": 0, "barcodes": 0, "sampled": False, "h1_pairs": 0, "h1_pairs_null": 0,
             "umis_with_h1_neighbour": 0, "umis_with_h1_neighbour_null": 0}
    if parsed.height == 0:
        return empty
    u = parsed.select("barcode", "umi").drop_nulls().unique()
    if u.height == 0:
        return empty
    L = u["umi"].str.len_chars().mode().sort()[-1]
    u = u.filter(pl.col("umi").str.len_chars() == L)
    sampled = False
    if u.height > max_umis:
        bcs = u["barcode"].unique().sort()
        keep = bcs.sample(fraction=max_umis / u.height, seed=seed)
        u = u.filter(pl.col("barcode").is_in(keep.implode()))
        sampled = True
    obs, obs_nb = _hamming1_pairs(u)
    shuffled = u.with_columns(pl.col("barcode").shuffle(seed=seed)).unique()
    null, null_nb = _hamming1_pairs(shuffled)
    return {"umis": int(u.height), "barcodes": int(u["barcode"].n_unique()), "sampled": sampled,
            "h1_pairs": obs, "h1_pairs_null": null,
            "umis_with_h1_neighbour": obs_nb, "umis_with_h1_neighbour_null": null_nb}


def umi_summary(per_well: list[dict]) -> dict:
    keys = ["umis", "h1_pairs", "h1_pairs_null", "umis_with_h1_neighbour", "umis_with_h1_neighbour_null"]
    t = {k: int(sum(d.get(k, 0) for d in per_well)) for k in keys}
    t["sampled_wells"] = int(sum(bool(d.get("sampled")) for d in per_well))
    t["excess_h1_pairs"] = _ratio(t["h1_pairs"], t["h1_pairs_null"]) if t["h1_pairs_null"] else None
    obs_f = _ratio(t["umis_with_h1_neighbour"], t["umis"])
    null_f = _ratio(t["umis_with_h1_neighbour_null"], t["umis"])
    t["error_like_umi_fraction"] = obs_f
    t["error_like_umi_fraction_null"] = null_f
    t["excess_error_like_umi_fraction"] = (
        None if obs_f is None else round(max(0.0, obs_f - (null_f or 0.0)), 6)
    )
    t["definition"] = (
        "Hamming-1 UMI pairs among distinct UMIs of the same barcode vs a null with barcodes "
        "shuffled across (barcode, UMI) rows; excess_h1_pairs = observed / null"
    )
    return t


# --------------------------------------------------------------------------------------------
#   run-level metrics from the per-well outputs
# --------------------------------------------------------------------------------------------

_LIGHT = ["IGK", "IGL"]


def load_unpaired(files: list[tuple[str, Path]]) -> pl.DataFrame:
    """Productive annotated contigs (barcode, well, locus, v_gene, j_gene, junction, cdr3_aa) of the run."""
    cols = ["sequence_id", "locus", "productive", "v_gene", "j_gene", "junction", "cdr3_aa"]
    out = []
    for well, f in files:
        schema = pl.read_parquet_schema(f)
        if not all(c in schema for c in cols):
            continue
        out.append(
            pl.read_parquet(f, columns=cols)
            .filter(pl.col("productive").fill_null(False) & pl.col("junction").is_not_null())
            .with_columns(pl.col("sequence_id").str.split("_").list.first().alias("barcode"), pl.lit(well).alias("well"))
            .drop("productive")
        )
    if not out:
        return pl.DataFrame(schema={c: pl.String for c in ["sequence_id", "locus", "v_gene", "j_gene", "junction", "cdr3_aa", "barcode", "well"]})
    return pl.concat(out, how="diagonal_relaxed")


def ambient_proxy(unpaired: pl.DataFrame) -> dict:
    """Share of heavy-chain junctions (nt) seen under >=2 and >=10 distinct droplet barcodes, run-wide."""
    h = unpaired.filter(pl.col("locus") == "IGH")
    if h.height == 0:
        return {"heavy_junctions": 0, "share_ge2_barcodes": None, "share_ge10_barcodes": None}
    nb = h.group_by("junction").agg(pl.col("barcode").n_unique().alias("nb"))
    n = nb.height
    occ = h.join(nb, on="junction")
    return {
        "heavy_junctions": int(n),
        "share_ge2_barcodes": _ratio(int((nb["nb"] >= 2).sum()), n),
        "share_ge10_barcodes": _ratio(int((nb["nb"] >= 10).sum()), n),
        "contigs_with_junction_ge2_barcodes": _ratio(int((occ["nb"] >= 2).sum()), occ.height),
        "contigs_with_junction_ge10_barcodes": _ratio(int((occ["nb"] >= 10).sum()), occ.height),
        "definition": "distinct productive IGH junctions (nt) by number of distinct barcodes carrying them, across wells",
    }


def leakage_by_chain(output_directory: str | Path, min_wells: int = 8) -> dict:
    """
    Leakage f per chain with ``pooled_pairing.estimate_leakage`` on this run's per-well outputs.
    Skipped (``None`` values) when fewer than ``min_wells`` wells have outputs.
    """
    from .pooled_pairing import SPECIES, estimate_leakage, load_run

    try:
        occ = load_run(output_directory)
    except FileNotFoundError:
        return {"wells": 0, "skipped": "no per-well outputs"}
    W = int(occ["well"].n_unique())
    if W < min_wells:
        return {"wells": W, "skipped": f"fewer than {min_wells} wells"}
    sw = occ.group_by(*SPECIES, "well").agg(pl.col("umis").sum().alias("n"))
    out = {"wells": W}
    for name, loci in (("heavy", ["IGH"]), ("light", _LIGHT), ("IGK", ["IGK"]), ("IGL", ["IGL"])):
        f = estimate_leakage(sw.filter(pl.col("locus").is_in(loci)), W)
        out[name] = None if f is None else round(f, 4)
    return out


CONF_BINS = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0001]
CONF_LABELS = ["<0.1", "0.1-0.3", "0.3-0.6", "0.6-0.9", ">=0.9"]


def confidence_distribution(conf: pl.Series | None) -> dict | None:
    if conf is None:
        return None
    c = conf.drop_nulls().cast(pl.Float64)
    if c.len() == 0:
        return {"n": 0}
    bins = {}
    for lo, hi, lab in zip(CONF_BINS[:-1], CONF_BINS[1:], CONF_LABELS):
        bins[lab] = int(((c >= lo) & (c < hi)).sum())
    q = {f"q{int(p * 100):02d}": round(float(c.quantile(p, "linear")), 4) for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
    return {"n": int(c.len()), "mean": round(float(c.mean()), 4), **q, "bins": bins}


def _norm_light_v(expr: pl.Expr) -> pl.Expr:
    # IGKV1D-39 and IGKV1-39 are the same gene in practice: count them as one light V
    return expr.str.replace(r"^(IG[KL]V\d+)D-", "${1}-")


def clonal_coherence(paired: pl.DataFrame) -> dict:
    """
    Clonal light-chain coherence. Clone = heavy V gene + CDRH3 aa; purity = share of a clone's
    cells carrying its modal light V gene, pooled over clones with >= 2 cells (cell-weighted);
    ``purity_mean_clone`` is the unweighted mean over those clones. ``chance`` = sum p^2 over the
    light V frequencies of all pairs (the purity expected if light chains were random).
    """
    need = ["v_gene:0", "cdr3_aa:0", "v_gene:1"]
    if paired.height == 0 or not all(c in paired.columns for c in need):
        return {"pairs": int(paired.height), "clones_ge2": 0, "purity": None, "chance": None}
    d = paired.select(
        pl.col("v_gene:0").alias("hv"), pl.col("cdr3_aa:0").alias("hcdr3"), _norm_light_v(pl.col("v_gene:1")).alias("lv")
    ).drop_nulls()
    p = d["lv"].value_counts()["count"].to_numpy().astype(float)
    chance = round(float(((p / p.sum()) ** 2).sum()), 4) if p.size else None
    c = d.group_by("hv", "hcdr3", "lv").len()
    cl = c.group_by("hv", "hcdr3").agg(pl.col("len").sum().alias("size"), pl.col("len").max().alias("top"))
    cl = cl.filter(pl.col("size") >= 2)
    if cl.height == 0:
        return {"pairs": int(paired.height), "clones_ge2": 0, "purity": None, "chance": chance}
    return {
        "pairs": int(paired.height),
        "clones_ge2": int(cl.height),
        "cells_in_clones": int(cl["size"].sum()),
        "purity": round(float(cl["top"].sum() / cl["size"].sum()), 4),
        "purity_mean_clone": round(float((cl["top"] / cl["size"]).mean()), 4),
        "chance": chance,
    }


# --------------------------------------------------------------------------------------------
#   report
# --------------------------------------------------------------------------------------------

PURITY_CAVEAT = (
    "Clonal light-chain purity is only meaningful for memory / clonally expanded repertoires; "
    "in naive repertoires clones with >= 2 cells are rare and mostly chance coincidences, "
    "so purity there says little about pairing accuracy. Use it to rank runs, not as an accuracy."
)


def build_report(
    output_directory: str | Path,
    stage_counts: pl.DataFrame,
    failures: pl.DataFrame,
    fragments: dict,
    umi_stats: list[dict],
    wells: list[str],
) -> dict:
    """Assemble the run-level QC report from the counters and the per-well outputs on disk."""
    output_directory = Path(output_directory)
    ann_dir = output_directory / "annotated"
    unpaired_files = [(w, ann_dir / f"{w}_unpaired.parquet") for w in wells]
    unpaired_files = [(w, f) for w, f in unpaired_files if f.exists() and f.stat().st_size > 0]
    paired_files = [ann_dir / f"{w}_paired.parquet" for w in wells]
    paired_files = [f for f in paired_files if f.exists() and f.stat().st_size > 0]

    paired = (
        pl.concat([pl.read_parquet(f) for f in paired_files], how="diagonal_relaxed")
        if paired_files else pl.DataFrame()
    )
    conf = None
    for src in (output_directory / "pairs" / "pairs.parquet",):
        if src.exists():
            p = pl.read_parquet(src)
            if "confidence" in p.columns:
                conf = p["confidence"]
    if conf is None and "confidence" in paired.columns:
        conf = paired["confidence"]

    total = stage_counts.filter(pl.col("well") == "TOTAL").to_dicts()
    total = total[0] if total else {}
    total.pop("well", None)

    try:
        leakage = leakage_by_chain(output_directory)
    except Exception as e:  # QC must never fail a run
        leakage = {"error": repr(e)}

    top_fail = (
        failures.filter(pl.col("well") == "TOTAL").sort("count", descending=True).head(10).drop("well").to_dicts()
        if failures.height else []
    )
    return {
        "wells": len(wells),
        "stage_totals": total,
        "heavy_fragment_fraction": fragment_summary(fragments),
        "umi_error_signature": umi_summary(umi_stats),
        "ambient_proxy": ambient_proxy(load_unpaired(unpaired_files)),
        "leakage": leakage,
        "pair_confidence": confidence_distribution(conf),
        "clonal_coherence": {**clonal_coherence(paired), "caveat": PURITY_CAVEAT},
        "top_annotation_failures": top_fail,
    }


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def report_markdown(rep: dict) -> str:
    L = ["# PairPlex QC report", "", f"Wells: {rep['wells']}", ""]
    L += ["## Stage totals", "", "| stage | count |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in rep["stage_totals"].items()]
    fr = rep["heavy_fragment_fraction"]
    L += ["", "## Fragment fraction", "", fr["definition"], "",
          f"Passing consensus sequences: {fr['passing_consensus']}; fragment fraction: {_fmt(fr['fragment_fraction'])}; "
          f"failed annotation: {_fmt(fr['failed_annotation_fraction'])}", "",
          "| locus | n | truncated | incomplete VDJ | fragment fraction |", "|---|---|---|---|---|"]
    L += [f"| {k} | {d['n']} | {d['truncated']} | {d['incomplete_vdj']} | {_fmt(d['fragment_fraction'])} |"
          for k, d in fr["by_locus"].items()]
    u = rep["umi_error_signature"]
    L += ["", "## UMI error signature", "", u["definition"], "",
          f"- distinct (barcode, UMI): {u['umis']} (wells subsampled: {u['sampled_wells']})",
          f"- Hamming-1 pairs: {u['h1_pairs']} observed vs {u['h1_pairs_null']} shuffled null "
          f"(excess x{_fmt(u['excess_h1_pairs'])})",
          f"- UMIs with a Hamming-1 neighbour: {_fmt(u['error_like_umi_fraction'])} "
          f"(null {_fmt(u['error_like_umi_fraction_null'])}; excess {_fmt(u['excess_error_like_umi_fraction'])})"]
    a = rep["ambient_proxy"]
    L += ["", "## Ambient proxy", "", a.get("definition", ""), "",
          f"- heavy junctions: {a['heavy_junctions']}",
          f"- share under >=2 barcodes: {_fmt(a['share_ge2_barcodes'])}",
          f"- share under >=10 barcodes: {_fmt(a['share_ge10_barcodes'])}"]
    lk = rep["leakage"]
    L += ["", "## Leakage f (pooled_pairing.estimate_leakage)", ""]
    if "skipped" in lk or "error" in lk:
        L.append(f"Not computed: {lk.get('skipped') or lk.get('error')} (wells: {lk.get('wells', 'n/a')})")
    else:
        L += [f"- {k}: {_fmt(lk.get(k))}" for k in ("heavy", "light", "IGK", "IGL")] + [f"- wells: {lk['wells']}"]
    c = rep["pair_confidence"]
    L += ["", "## Pair confidence", ""]
    if not c:
        L.append("No confidence column in this run's pair outputs.")
    else:
        L.append(f"n = {c['n']}" + (f", mean {_fmt(c['mean'])}, median {_fmt(c['q50'])}" if c.get("n") else ""))
        if c.get("bins"):
            L += ["", "| bin | pairs |", "|---|---|"] + [f"| {k} | {v} |" for k, v in c["bins"].items()]
    cc = rep["clonal_coherence"]
    L += ["", "## Clonal light-chain coherence", "", f"> **Caveat:** {cc['caveat']}", "",
          "Clone = heavy V gene + CDRH3 aa; purity = share of a clone's cells with its modal light V "
          "(clones with >= 2 cells, cell-weighted); chance = sum p^2 over light V genes.", "",
          f"- pairs: {cc['pairs']}; clones with >= 2 cells: {cc['clones_ge2']}",
          f"- purity: {_fmt(cc.get('purity'))} (mean per clone {_fmt(cc.get('purity_mean_clone'))}); chance: {_fmt(cc.get('chance'))}"]
    if rep["top_annotation_failures"]:
        L += ["", "## Top annotation failure modes", "", "| kind | mode | count |", "|---|---|---|"]
        L += [f"| {r['kind']} | {r['mode']} | {r['count']} |" for r in rep["top_annotation_failures"]]
    return "\n".join(L) + "\n"


def write_qc(
    output_directory: str | Path,
    stage_rows: list[dict],
    failure_rows: list[dict],
    fragments: dict,
    umi_stats: list[dict],
    wells: list[str],
) -> dict:
    """Write ``qc/stage_counts.csv``, ``qc/annotation_failures.csv``, ``qc/report.json`` and ``qc/report.md``."""
    qc_dir = Path(output_directory) / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    sc = stage_counts_frame(stage_rows)
    sc.write_csv(qc_dir / "stage_counts.csv")
    ft = failure_table(failure_rows)
    ft.write_csv(qc_dir / "annotation_failures.csv")
    rep = build_report(output_directory, sc, ft, fragments, umi_stats, wells)
    (qc_dir / "report.json").write_text(json.dumps(rep, indent=2, default=_json_default) + "\n")
    (qc_dir / "report.md").write_text(report_markdown(rep))
    return rep


# --------------------------------------------------------------------------------------------
#   support columns
# --------------------------------------------------------------------------------------------

SUPPORT_FIELDS = ("umis", "reads", "cluster_fraction")


def add_support_columns(sequences, metadata: pl.DataFrame) -> None:
    """
    Add cluster support to abstar's ``Sequence`` objects in place, joined on the contig name
    (= ``sequence_id``): ``umis``, ``reads``, ``cluster_fraction``, ``barcode`` and ``contig``
    (``contig-N``, N = rank by reads within the barcode). Paired outputs built from these objects
    get them as ``umis:0/1`` etc.

    abstar's ``umi`` field is removed: it holds one read's UMI parsed from the read header with
    ``umi_pattern``, which PairPlex never sets; a consensus contig stands for many UMIs, so the
    field is always null here and the count lives in ``umis``.
    """
    if sequences is None:
        return
    if not isinstance(sequences, list | tuple):
        sequences = [sequences]
    cols = [c for c in ("name", *SUPPORT_FIELDS) if c in metadata.columns]
    support = {r[0]: r[1:] for r in metadata.select(cols).iter_rows()}
    names = cols[1:]
    for s in sequences:
        sid = s.id
        vals = support.get(sid)
        for k, v in zip(names, vals if vals is not None else [None] * len(names)):
            s[k] = v
        bc, _, contig = (sid or "").rpartition("_")
        s["barcode"] = bc or None
        s["contig"] = contig or None
        if s.annotations is not None:
            s.annotations.pop("umi", None)


# --------------------------------------------------------------------------------------------
#   helpers
# --------------------------------------------------------------------------------------------


def _ratio(a, b):
    return None if not b else round(float(a) / float(b), 6)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if math.isnan(o) else float(o)
    return str(o)
