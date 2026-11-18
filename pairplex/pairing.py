# Copyright (c) 2025 Benjamin Nemoz
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""
Per-well heavy/light pairing of droplet barcodes.

Two rules are available:

``"exact"`` (legacy, ``v2-base``)
    Keep a barcode only if ``abutils.tl.assign_pairs`` finds exactly one heavy and one light
    chain (any productivity). Barcodes with several chains are dropped.

``"scored"`` (default)
    Dominance-aware resolution. For each (well, barcode), the *productive* heavy chains (IGH)
    and the *productive* light chains (IGK and IGL together: a cell expresses one light chain,
    so kappa and lambda compete in the same group) are ranked by support: corrected UMIs from
    the metadata CSV, then reads as a tie-break, then contig name for determinism.
    Same-locus contigs sharing V gene, J gene and CDR3 aa are first merged into one chain
    (summed support; ``collapse_clonal``), so a chain split into two consensus clusters, or two
    clonal relatives, never competes with itself.
    The top heavy and the top light are emitted when, in each group,

        u1 >= min_dominance * u2

    where ``u1``/``u2`` are the UMIs of the best and second-best chain (``u2 = 0`` for a
    single chain). Each emitted pair gets

    * ``confidence = s_H * s_L`` with ``s_g = u1_g / (U_g + pseudocount)``, ``U_g`` the total
      UMIs of the group. ``s_g`` is the top chain's share of the group's support after adding
      ``pseudocount`` UMIs for an unobserved competing chain (Laplace-style). It is in [0, 1),
      increases with the top chain's UMIs and decreases with every competitor's UMIs, so a
      deep 1H+1L barcode scores close to 1, a 1-UMI 1H+1L barcode scores 0.25 and an even
      split scores low (and is not emitted unless ``min_dominance <= 1``).
    * ``margin = min_g (u1_g - u2_g) / (U_g + pseudocount)``: the smaller, over the two
      groups, of the gap between the top and the second-best chain's share. In [0, 1); equal
      to ``min(s_H, s_L)`` for a 1H+1L barcode.
    * ``umis`` on each chain (``umis:0`` heavy, ``umis:1`` light in the paired outputs): the
      emitted contig's own UMI count from the metadata (same meaning as the support columns
      of the unpaired outputs).
    * ``chain_umis`` on each chain: the support used for ranking, i.e. ``umis`` summed over
      the same-clonotype contigs merged with it (equal to ``umis`` when nothing was merged).

Default ``min_dominance = 2.0``: the top chain must hold at least two thirds of a 2-chain
group's UMIs. In real data about half of the 1H+2L / 2H+1L barcodes meet it; near-even splits
(where picking the top chain is close to a coin flip) are left unpaired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import abutils
import polars as pl
from natsort import natsorted

HEAVY_LOCI = ("IGH",)
LIGHT_LOCI = ("IGK", "IGL")
PAIRING_MODES = ("scored", "exact")
DEFAULT_MIN_DOMINANCE = 2.0
DEFAULT_PSEUDOCOUNT = 1.0

# contract columns of pairs/pairs.parquet (see diagnosis/bench/README.md), per chain
_CHAIN_COLS = ["locus", "v_gene", "j_gene", "junction", "cdr3_aa", "sequence_id", "sequence", "umis", "chain_umis"]
_INT_COLS = ("umis", "chain_umis")
_WELL_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class Chain:
    """Support of one contig: name, UMIs, reads."""

    name: str
    umis: int
    reads: int = 0


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving one barcode. ``heavy``/``light`` are None when not emitted."""

    heavy: Chain | None
    light: Chain | None
    confidence: float | None
    margin: float | None
    reason: str  # "paired", "no_heavy", "no_light", "heavy_ambiguous", "light_ambiguous", "both_ambiguous"


def _rank(chains: Iterable[Chain]) -> list[Chain]:
    return sorted(chains, key=lambda c: (-c.umis, -c.reads, c.name))


def _group_stats(ranked: list[Chain], pseudocount: float) -> tuple[float, float, int, int]:
    u1 = ranked[0].umis
    u2 = ranked[1].umis if len(ranked) > 1 else 0
    denom = sum(c.umis for c in ranked) + pseudocount
    if denom <= 0:
        return 0.0, 0.0, u1, u2
    return u1 / denom, (u1 - u2) / denom, u1, u2


def resolve_barcode(
    heavies: Iterable[Chain],
    lights: Iterable[Chain],
    min_dominance: float = DEFAULT_MIN_DOMINANCE,
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
) -> Resolution:
    """
    Pick the top heavy and top light chain of one (well, barcode) and score the pair.

    Parameters
    ----------
    heavies, lights : iterable of Chain
        Productive heavy chains and productive light chains (IGK and IGL together).
    min_dominance : float
        Emit only if, in each group, the top chain has >= ``min_dominance`` x the UMIs of the
        second-best chain. Values <= 1 accept any split (ties broken by reads, then name).
    pseudocount : float
        UMIs added to each group's total for an unobserved competing chain (see module doc).

    Returns
    -------
    Resolution
    """
    hs, ls = _rank(heavies), _rank(lights)
    if not hs or not ls:
        return Resolution(None, None, None, None, "no_heavy" if not hs else "no_light")
    sh, mh, h1, h2 = _group_stats(hs, pseudocount)
    sl, ml, l1, l2 = _group_stats(ls, pseudocount)
    h_ok = h1 >= min_dominance * h2
    l_ok = l1 >= min_dominance * l2
    if not (h_ok and l_ok):
        reason = "both_ambiguous" if not (h_ok or l_ok) else ("heavy_ambiguous" if not h_ok else "light_ambiguous")
        return Resolution(None, None, None, None, reason)
    return Resolution(hs[0], ls[0], sh * sl, min(mh, ml), "paired")


def _gene(name) -> str | None:
    return name.split("*")[0] if isinstance(name, str) else None


def collapse_clonotypes(chains: list[Chain], annotations: dict) -> list[Chain]:
    """
    Merge chains of one locus group that share a clonotype (locus, V gene, J gene, CDR3 aa).

    Within one (well, barcode), two contigs with the same clonotype are one chain split into two
    consensus clusters (sequencing or PCR errors, fragments) or clonal relatives; either way they
    support the same pairing and must not compete. The merged chain keeps the name of its best
    contig (UMIs, reads, name) and the summed UMIs and reads. Contigs without a CDR3 aa are kept
    as they are.
    """
    groups: dict[tuple, list[Chain]] = {}
    for c in chains:
        a = annotations.get(c.name)
        cdr3 = a["cdr3_aa"] if a is not None else None
        key = (a["locus"], _gene(a["v_gene"]), _gene(a["j_gene"]), cdr3) if cdr3 else ("_id", c.name)
        groups.setdefault(key, []).append(c)
    out = []
    for members in groups.values():
        best = _rank(members)[0]
        out.append(Chain(best.name, sum(m.umis for m in members), sum(m.reads for m in members)))
    return out


def _barcode(sequence_id: str) -> str:
    # same grouping as abutils.tl.assign_pairs(delim="_", delim_occurance=-1)
    return "_".join(sequence_id.split("_")[:-1])


def _productive(seq) -> bool:
    p = seq["productive"]
    if isinstance(p, str):
        return p.strip().lower() in ("true", "t", "1", "yes")
    return bool(p)


def exact_pairs(sequences: list) -> list:
    """Legacy rule (``v2-base``): barcodes with exactly one heavy and one light chain."""
    pairs = abutils.tl.assign_pairs(sequences, delim="_", delim_occurance=-1)
    return [p for p in pairs if len(p.heavies) == 1 and len(p.lights) == 1]


def scored_pairs(
    sequences: list,
    support: dict[str, tuple[int, int]],
    min_dominance: float = DEFAULT_MIN_DOMINANCE,
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
    collapse_clonal: bool = True,
) -> tuple[list, dict[str, int]]:
    """
    Resolve every barcode of one well with ``resolve_barcode``.

    Parameters
    ----------
    sequences : list of abutils Sequence
        Annotated contigs of one well (``sequence_id`` = ``<barcode>_<contig>``).
    support : dict
        ``name -> (umis, reads)`` from the metadata CSV. Missing contigs count as 1 UMI, 0 reads.
    collapse_clonal : bool
        Merge same-locus contigs of a barcode that share V gene, J gene and CDR3 aa before
        ranking (see ``collapse_clonotypes``).

    Returns
    -------
    (pairs, reasons) : list of abutils ``Pair`` (heavy and light chains carry a ``umis``
    annotation; the pair has ``confidence`` and ``margin`` attributes) and a count per
    resolution reason.
    """
    groups: dict[str, tuple[list, list]] = {}
    by_id = {}
    for s in sequences:
        if not _productive(s):
            continue
        locus = s["locus"]
        if locus in HEAVY_LOCI:
            side = 0
        elif locus in LIGHT_LOCI:
            side = 1
        else:
            continue
        sid = s["sequence_id"]
        by_id[sid] = s
        groups.setdefault(_barcode(sid), ([], []))[side].append(sid)

    pairs, reasons = [], {}
    for bc, (h_ids, l_ids) in groups.items():
        def chains(ids):
            out = [Chain(i, *(int(x) for x in support.get(i, (1, 0)))) for i in ids]
            return collapse_clonotypes(out, by_id) if collapse_clonal else out

        res = resolve_barcode(chains(h_ids), chains(l_ids), min_dominance, pseudocount)
        reasons[res.reason] = reasons.get(res.reason, 0) + 1
        if res.heavy is None:
            continue
        heavy, light = by_id[res.heavy.name], by_id[res.light.name]
        for seq, ch in ((heavy, res.heavy), (light, res.light)):
            seq["umis"] = int(support.get(ch.name, (1, 0))[0])
            seq["chain_umis"] = ch.umis
        pairs.append(
            abutils.Pair(
                [heavy, light],
                name=bc,
                properties={"confidence": res.confidence, "margin": res.margin},
            )
        )
    return pairs, reasons


def support_from_metadata(metadata_df: pl.DataFrame) -> dict[str, tuple[int, int]]:
    """``name -> (umis, reads)`` from a PairPlex metadata table."""
    if metadata_df.is_empty() or "name" not in metadata_df.columns:
        return {}
    umis = metadata_df["umis"] if "umis" in metadata_df.columns else pl.Series([1] * metadata_df.height)
    reads = metadata_df["reads"] if "reads" in metadata_df.columns else pl.Series([0] * metadata_df.height)
    return {n: (int(u if u is not None else 1), int(r if r is not None else 0))
            for n, u, r in zip(metadata_df["name"], umis, reads)}


def write_paired(pairs: list, airr_file: str | Path, parquet_file: str | Path, scored: bool) -> None:
    """Write per-well paired files; scored pairs get ``confidence`` and ``margin`` columns."""
    props = ["confidence", "margin"] if scored else None
    abutils.io.to_airr(pairs, str(airr_file), properties=props, drop_na_columns=False)
    abutils.io.to_parquet(pairs, str(parquet_file), properties=props, drop_na_columns=False)


def pair_well(
    sequences: list,
    paired_airr_file: str | Path,
    paired_parquet_file: str | Path,
    mode: str = "scored",
    support: dict[str, tuple[int, int]] | None = None,
    min_dominance: float = DEFAULT_MIN_DOMINANCE,
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
    collapse_clonal: bool = True,
) -> list:
    """
    Pair one well's annotated contigs with the chosen rule and write the paired files.

    Pairs are written (and returned) in natural barcode order, so the output row order does not
    depend on the order of ``sequences``.
    """
    if mode == "exact":
        pairs = natsorted(exact_pairs(sequences), key=lambda p: p.name)
        # identical to v2-base output
        abutils.io.to_airr(pairs, str(paired_airr_file), drop_na_columns=False)
        abutils.io.to_parquet(pairs, str(paired_parquet_file), drop_na_columns=False)
        return pairs
    if mode != "scored":
        raise ValueError(f"unknown pairing mode {mode!r}; expected one of {PAIRING_MODES}")
    pairs, _ = scored_pairs(sequences, support or {}, min_dominance, pseudocount, collapse_clonal)
    pairs = natsorted(pairs, key=lambda p: p.name)
    write_paired(pairs, paired_airr_file, paired_parquet_file, scored=True)
    return pairs


def well_number(name: str) -> int | None:
    """Well index from a file stem (first number: ``well007.fastq`` -> 7, ``VDJ_74`` -> 74)."""
    m = _WELL_RE.search(name)
    return int(m.group(1)) if m else None


def contract_frame(paired_parquet: str | Path, well: int | None) -> pl.DataFrame | None:
    """One well's paired file in the ``pairs/pairs.parquet`` contract schema."""
    p = Path(paired_parquet)
    if not p.exists() or p.stat().st_size == 0:
        return None
    df = pl.read_parquet(p)
    if df.height == 0:
        return None
    cols = [pl.col("name").alias("barcode"), pl.lit(well, pl.Int64).alias("well"), pl.col("name")]
    for c in ("confidence", "margin"):
        cols.append(pl.col(c).cast(pl.Float64) if c in df.columns else pl.lit(None, pl.Float64).alias(c))
    for s in (0, 1):
        for c in _CHAIN_COLS:
            k = f"{c}:{s}"
            cols.append(pl.col(k) if k in df.columns else pl.lit(None).alias(k))
    return df.select(cols).with_columns(pl.col(f"{c}:{i}").cast(pl.Int64) for c in _INT_COLS for i in (0, 1))


def write_contract_pairs(frames: list[pl.DataFrame], output_file: str | Path) -> int:
    """Concatenate per-well contract frames into ``pairs/pairs.parquet``; returns the row count."""
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = [f for f in frames if f is not None and f.height]
    if frames:
        df = pl.concat(frames, how="diagonal_relaxed")
    else:
        schema = {"barcode": pl.Utf8, "well": pl.Int64, "name": pl.Utf8, "confidence": pl.Float64, "margin": pl.Float64}
        for s in (0, 1):
            schema.update({f"{c}:{s}": (pl.Int64 if c in _INT_COLS else pl.Utf8) for c in _CHAIN_COLS})
        df = pl.DataFrame(schema=schema)
    df.write_parquet(out)
    return df.height
