# Copyright (c) 2025 Benjamin Nemoz
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""
Pooled, confidence-scored heavy/light pairing for split-emulsion (UDA-seq style) libraries.

Why
---
``pairplex.run`` pairs chains within each (droplet barcode, well) and keeps a barcode only if
it holds exactly one heavy and one light. That assumes a cell's barcoded cDNA stays in the well
the cell was dispensed to. Diagnosis of real data showed it does not: ~65-85% of a cell's
distinct molecules end up in OTHER wells (leakage of barcoded cDNA out of fixed cells before the
split), so a well holds a fraction of each resident cell plus leaked chains of every droplet-mate
sharing the barcode.

What this does
--------------
Works on a finished per-well PairPlex run (``annotated/*_unpaired.parquet`` + ``metadata/*.csv``):

1. pools every productive chain of a reaction by droplet barcode across wells; a *chain species*
   is (barcode, locus, V gene, J gene, junction nt), with its UMIs counted per well;
2. estimates the leakage ``f`` from the reaction itself (species with >= 6 UMIs) unless given;
3. computes, for each species and well, the posterior that the well is the chain's home under
   "each molecule stays home with prob 1-f, else lands in a uniform random well":
   ``P(home=w) = r**n_w / (sum_observed r**n_v + (W - k))`` with ``r = (1-f+f/W)/(f/W)``;
4. pairs heavy and light species of the same barcode that share a candidate home well, scoring
   ``confidence = P_H(w) * P_L(w)``, and keeps reciprocal-best pairs;
5. reports every pair with its confidence, home well, pooled support and an ambiguity margin
   instead of a hard yes/no, so downstream analyses can choose their own precision/yield point.

Output rows carry ``well`` (= home well), ``sequence_id:0/1`` and ``sequence:0/1`` so they can be
scored directly with ``simplex.score``.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import polars as pl

SPECIES = ["barcode", "locus", "v_gene", "j_gene", "junction"]
_ANN_COLS = ["sequence_id", "locus", "productive", "v_gene", "j_gene", "junction", "cdr3_aa", "sequence"]
_WELL_RE = re.compile(r"(\d+)")  # first number in the file name: VDJ_74_unpaired.parquet, well000.fastq.csv


def _well_files(run_dir: Path, kind: str, wells: tuple[int, int] | None) -> list[tuple[int, str]]:
    pat = {"ann": "annotated/*_unpaired.parquet", "meta": "metadata/*.csv"}[kind]
    out = []
    for f in glob.glob(str(run_dir / pat)):
        m = _WELL_RE.search(os.path.basename(f))
        if not m or os.path.getsize(f) < 100:
            continue
        w = int(m.group(1))
        if wells is None or wells[0] <= w <= wells[1]:
            out.append((w, f))
    return sorted(out)


def load_run(run_dir: str | Path, wells: tuple[int, int] | None = None) -> pl.DataFrame:
    """
    Load productive annotated contigs of a per-well PairPlex run, with per-contig UMI counts.

    Returns one row per contig: barcode, well, locus, v_gene, j_gene, junction, cdr3_aa,
    sequence_id, sequence, umis.
    """
    run_dir = Path(run_dir)
    ann = []
    for w, f in _well_files(run_dir, "ann", wells):
        cols = [c for c in _ANN_COLS if c in pl.read_parquet_schema(f)]
        ann.append(
            pl.scan_parquet(f).select(cols)
            .filter(pl.col("productive") & pl.col("locus").is_in(["IGH", "IGK", "IGL"]) & pl.col("junction").is_not_null())
            .with_columns(pl.col("sequence_id").str.split("_").list.first().alias("barcode"), pl.lit(w).alias("well"))
            .drop("productive").collect()
        )
    if not ann:
        raise FileNotFoundError(f"no annotated *_unpaired.parquet under {run_dir}")
    occ = pl.concat(ann, how="diagonal_relaxed")
    keys = occ.select("sequence_id", "well")
    meta = []
    for w, f in _well_files(run_dir, "meta", wells):
        meta.append(
            pl.scan_csv(f).select("name", "umis").with_columns(pl.lit(w).alias("well"))
            .join(keys.lazy(), left_on=["name", "well"], right_on=["sequence_id", "well"], how="semi").collect()
        )
    if meta:
        occ = occ.join(pl.concat(meta), left_on=["sequence_id", "well"], right_on=["name", "well"], how="left")
    else:
        occ = occ.with_columns(pl.lit(None, pl.Int64).alias("umis"))
    return occ.with_columns(pl.col("umis").fill_null(1).cast(pl.Int64))


def estimate_leakage(species_wells: pl.DataFrame, n_wells: int, min_umis: int = 6) -> float | None:
    """
    Leakage f from species with >= ``min_umis`` UMIs: outside-modal-well fraction / (1 - 1/W).
    ``species_wells`` has one row per (species, well) with column ``n``.
    """
    tot = species_wells.group_by(SPECIES).agg(pl.col("n").sum().alias("N"), pl.col("n").max().alias("nmax"))
    big = tot.filter(pl.col("N") >= min_umis)
    if big.height == 0:
        return None
    out = 1 - big["nmax"].sum() / big["N"].sum()
    return float(min(0.99, max(0.0, out / (1 - 1 / n_wells))))


def home_posteriors(species_wells: pl.DataFrame, n_wells: int, leakage: float) -> pl.DataFrame:
    """Add ``p_home`` (posterior that the row's well is the species' home) and ``k`` (wells observed)."""
    f = min(max(leakage, 1e-6), 0.999)
    log_r = float(np.log((1 - f + f / n_wells) / (f / n_wells)))
    sw = species_wells.with_columns((pl.col("n") * log_r).alias("_ln"))
    z = sw.group_by(SPECIES).agg(pl.col("_ln").max().alias("_m"), pl.col("_ln").alias("_lns"), pl.len().alias("k"))
    z = z.with_columns(
        (pl.col("_m") + (pl.col("_lns").list.eval((pl.element() - pl.element().max()).exp()).list.sum()
                         + (n_wells - pl.col("k")) * (-pl.col("_m")).exp()).log()).alias("_logZ")
    ).select(*SPECIES, "k", "_logZ")
    return sw.join(z, on=SPECIES).with_columns((pl.col("_ln") - pl.col("_logZ")).exp().alias("p_home")).drop("_ln", "_logZ")


def pair_pooled(
    occ: pl.DataFrame,
    n_wells: int | None = None,
    leakage: float | None = None,
    min_confidence: float = 0.0,
) -> tuple[pl.DataFrame, dict]:
    """
    Pool contigs by barcode, assign home-well posteriors, pair H/L by shared home well.

    Parameters
    ----------
    occ : DataFrame
        One row per contig (see ``load_run``): barcode, well, locus, v_gene, j_gene, junction,
        cdr3_aa, sequence_id, sequence, umis.
    n_wells : int | None
        Wells in the reaction (default: wells observed in ``occ``).
    leakage : float | None
        Pre-split leakage f; estimated from the data if None.
    min_confidence : float
        Drop pairs below this confidence (default keeps all, with their score).

    Returns
    -------
    (pairs, info) : pairs has one row per reciprocal-best (H, L) pair with ``well`` (home well),
    ``confidence``, ``p_home:0/1``, home and pooled UMIs per chain, ``margin`` (confidence minus
    the best competing candidate for either chain), plus ``sequence_id:0/1``, ``sequence:0/1``,
    ``v_gene``/``j_gene``/``cdr3_aa`` for both chains (``:0`` heavy, ``:1`` light).
    info has W, the leakage used and whether it was estimated.
    """
    W = int(n_wells or occ["well"].n_unique())
    # per (species, well): UMIs, and the representative contig there (most UMIs)
    sw = (occ.sort("umis", descending=True)
             .group_by(*SPECIES, "well")
             .agg(pl.col("umis").sum().alias("n"), pl.col("sequence_id").first(), pl.col("sequence").first(),
                  pl.col("cdr3_aa").first()))
    est = leakage is None
    f = estimate_leakage(sw, W) if est else float(leakage)
    if f is None:
        f = 0.8  # documented fallback when too few well-supported species exist
    sw = home_posteriors(sw, W, f)
    pooled = sw.group_by(SPECIES).agg(pl.col("n").sum().alias("N"))
    sw = sw.join(pooled, on=SPECIES)

    def side(locus_filter, s):
        return sw.filter(locus_filter).select(
            "barcode", "well",
            pl.col("locus").alias(f"locus:{s}"), pl.col("v_gene").alias(f"v_gene:{s}"), pl.col("j_gene").alias(f"j_gene:{s}"),
            pl.col("junction").alias(f"junction:{s}"), pl.col("cdr3_aa").alias(f"cdr3_aa:{s}"),
            pl.col("sequence_id").alias(f"sequence_id:{s}"), pl.col("sequence").alias(f"sequence:{s}"),
            pl.col("p_home").alias(f"p_home:{s}"), pl.col("n").alias(f"umis_home:{s}"), pl.col("N").alias(f"umis_pooled:{s}"),
            pl.col("k").alias(f"n_wells:{s}"))

    H = side(pl.col("locus") == "IGH", 0)
    L = side(pl.col("locus") != "IGH", 1)
    cand = H.join(L, on=["barcode", "well"]).with_columns((pl.col("p_home:0") * pl.col("p_home:1")).alias("confidence"))
    hkey = ["barcode", "locus:0", "v_gene:0", "j_gene:0", "junction:0"]
    lkey = ["barcode", "locus:1", "v_gene:1", "j_gene:1", "junction:1"]
    cand = cand.with_columns(
        pl.col("confidence").rank("ordinal", descending=True).over(hkey).alias("_rh"),
        pl.col("confidence").rank("ordinal", descending=True).over(lkey).alias("_rl"),
        pl.col("confidence").sort(descending=True).slice(1, 1).first().over(hkey).fill_null(0.0).alias("_h2"),
        pl.col("confidence").sort(descending=True).slice(1, 1).first().over(lkey).fill_null(0.0).alias("_l2"),
    )
    pairs = (cand.filter((pl.col("_rh") == 1) & (pl.col("_rl") == 1))
                 .with_columns((pl.col("confidence") - pl.max_horizontal("_h2", "_l2")).alias("margin"))
                 .drop("_rh", "_rl", "_h2", "_l2")
                 .filter(pl.col("confidence") >= min_confidence)
                 .with_columns(pl.col("barcode").alias("name"))
                 .sort("confidence", descending=True))
    info = {"wells": W, "leakage": f, "leakage_estimated": est, "species": int(pooled.height), "pairs": int(pairs.height)}
    return pairs, info


def run_pooled_pairing(
    run_dir: str | Path,
    output_file: str | Path | None = None,
    wells: tuple[int, int] | None = None,
    leakage: float | None = None,
    min_confidence: float = 0.0,
) -> tuple[pl.DataFrame, dict]:
    """Load a per-well PairPlex run, pair with ``pair_pooled`` and write a parquet (default
    ``<run_dir>/pooled/pooled_pairs.parquet``)."""
    occ = load_run(run_dir, wells)
    pairs, info = pair_pooled(occ, leakage=leakage, min_confidence=min_confidence)
    out = Path(output_file) if output_file else Path(run_dir) / "pooled" / "pooled_pairs.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.write_parquet(out)
    info["output"] = str(out)
    return pairs, info
