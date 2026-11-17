# Copyright (c) 2025 Benjamin Nemoz
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""
Cross-well processing mode: ``pairplex.run(..., pooling="across_wells")``.

Why
---
Barcoded cDNA leaks out of fixed cells before the 96-well split. A released molecule keeps its
droplet barcode and UMI and lands in a random well; PCR happens after the split, so the reads of
one molecule normally sit in one well. Processing each well on its own therefore sees a cell's
chains scattered over wells (in SimPlex, 69-73% of cells have one chain absent from their home
well). ``pair-pooled`` pools finished per-well contigs; this mode pools at READ level.

Pipeline
--------
1. Parse every well file with :func:`pairplex.utils.parse_barcodes` (well = last integer in the
   file stem) and stream the reads into on-disk shards by barcode hash, so memory stays bounded
   by one shard (about ``SHARD_TARGET_GZ_BYTES`` of gzipped input) rather than the reaction.
2. Per shard, per barcode (in the process pool):
   - collapse UMIs across all wells with UMI-tools' directional adjacency
     (b is absorbed by a if Hamming(a, b) == 1 and count(a) >= 2 * count(b) - 1, counts = reads
     per raw UMI over all wells); a molecule is (barcode, corrected UMI);
   - cluster all the barcode's reads with the legacy greedy clustering and threshold, and make one
     POA consensus per cluster (as legacy);
   - inside each contig, assign each molecule to the well holding most of its reads (ties: lowest
     well number); reads in other wells are counted as hopped. ``n_w`` = molecules assigned to w.
3. Annotate passing contigs per shard with ``abstar.run`` (``mmseqs_threads=1``).
4. Estimate the leakage f over the whole reaction, then pair each shard with
   :func:`pairplex.pooled_pairing.pair_pooled` on the per-contig ``n_w`` (home-well posterior,
   reciprocal best pairs, confidence and margin).

Outputs: ``pairs/pairs.parquet``, ``contigs/contigs.parquet``, ``contigs/contig_wells.parquet``.
"""

from __future__ import annotations

import math
import os
import re
import resource
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path

import abstar
import abutils
import polars as pl
import pyarrow.parquet as pq
from natsort import natsorted
from spoa import poa

from .pooled_pairing import SPECIES, leakage_from_totals, pair_pooled
from .utils import available_cpus, greedy_clustering, parse_barcodes

SHARD_TARGET_GZ_BYTES = 160_000_000  # ~1M merged reads of gzipped FASTQ per shard
PARSE_CHUNK_READS = 10_000  # reads per parse_barcodes task
MAX_READS_IN_FLIGHT = 4_000_000  # split-but-unparsed reads on temp disk
FLUSH_READS = 1_000_000  # parsed reads buffered before they are written to the shards
_FASTX_EXT = re.compile(r"\.(fastq|fq|fasta|fa)(\.gz)?$", re.IGNORECASE)
_SHARD_SEED = 20261117
_READ_SCHEMA = {"barcode": pl.String, "umi": pl.String, "well": pl.Int32, "sequence": pl.String}

PAIR_EXTRA_COLS = ("umis_home", "umis_pooled", "n_wells", "p_home")


# ---------------------------------------------------------------------------------------------
#  wells and shards
# ---------------------------------------------------------------------------------------------


def well_from_path(path: str | Path) -> int:
    """Well number of a per-well input file: the last integer in its stem (``VDJ_123.fastq.gz``
    -> 123, ``well042.fq`` -> 42)."""
    stem = _FASTX_EXT.sub("", Path(path).name)
    nums = re.findall(r"\d+", stem)
    if not nums:
        raise ValueError(f"cannot derive a well number from file name {Path(path).name!r}")
    return int(nums[-1])


def auto_shard_count(input_files: list[str | Path], target_gz_bytes: int = SHARD_TARGET_GZ_BYTES) -> int:
    """Number of barcode shards so that one shard holds about ``target_gz_bytes`` of gzipped input
    (uncompressed files count for a quarter of their size)."""
    eq = 0.0
    for f in input_files:
        s = os.path.getsize(f)
        eq += s if str(f).endswith(".gz") else s / 4
    return int(min(4096, max(1, math.ceil(eq / target_gz_bytes))))


def shard_of(barcodes: pl.Expr | pl.Series, n_shards: int):
    """Shard index of each barcode (stable within one polars version)."""
    return (barcodes.hash(seed=_SHARD_SEED) % n_shards).cast(pl.Int32)


class ShardWriter:
    """Append-only writer of read batches to ``n_shards`` parquet files (one per shard)."""

    def __init__(self, shard_dir: Path, n_shards: int):
        self.dir = Path(shard_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = n_shards
        self._w: dict[int, pq.ParquetWriter] = {}
        self.rows = Counter()

    def path(self, i: int) -> Path:
        return self.dir / f"shard{i:04d}.parquet"

    def write(self, df: pl.DataFrame) -> None:
        if df.height == 0:
            return
        df = df.select([pl.col(c).cast(t) for c, t in _READ_SCHEMA.items()])
        for (i,), part in df.with_columns(shard_of(pl.col("barcode"), self.n).alias("_s")).partition_by(
                "_s", as_dict=True, include_key=False).items():
            tbl = part.to_arrow()
            if i not in self._w:
                self._w[i] = pq.ParquetWriter(self.path(i), tbl.schema)
            self._w[i].write_table(tbl.cast(self._w[i].schema))
            self.rows[i] += part.height

    def close(self) -> list[int]:
        for w in self._w.values():
            w.close()
        return sorted(self._w)


def _count_reads(path: str | Path) -> int:
    """Records in an uncompressed FASTA/Q chunk (4-line FASTQ, or '>' headers)."""
    with open(path, "rb") as fh:
        data = fh.read()
    if not data:
        return 0
    return data.count(b"\n>") + 1 if data[:1] == b">" else (data.count(b"\n") + (not data.endswith(b"\n"))) // 4


def parse_to_shards(
    input_files: list[str],
    writer: ShardWriter,
    executor: ProcessPoolExecutor,
    temp_directory: Path,
    whitelist_path: str | Path | None = None,
    chunk_reads: int = PARSE_CHUNK_READS,
    max_in_flight: int = MAX_READS_IN_FLIGHT,
    flush_reads: int = FLUSH_READS,
) -> dict:
    """Parse barcodes/UMIs of every well file in the pool and stream the reads into shards.

    Never holds more than ``flush_reads`` parsed reads (plus the chunks being parsed) in memory,
    and never more than ~``max_in_flight`` split reads on the temp disk."""
    chunk_dir = Path(temp_directory) / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pending: dict = {}  # future -> (well, fastq chunk, reads)
    buf: list[pl.DataFrame] = []
    stats = {"wells": set(), "input_reads": 0, "valid_reads": 0, "skipped_files": []}

    def flush():
        if buf:
            writer.write(pl.concat(buf))
            buf.clear()

    def collect(done):
        for fut in done:
            well, chunk, _ = pending.pop(fut)
            res = fut.result()
            os.remove(chunk)
            if res is None:
                continue
            df = pl.read_parquet(res, columns=["barcode", "umi", "sequence"]).with_columns(pl.lit(well, pl.Int32).alias("well"))
            os.remove(res)
            buf.append(df)
            stats["valid_reads"] += df.height
            stats["wells"].add(well)
        if sum(d.height for d in buf) >= flush_reads:
            flush()

    for f in natsorted(input_files):
        if os.path.getsize(f) == 0 or abutils.io.determine_fastx_format(str(f)) is None:
            stats["skipped_files"].append(os.path.basename(f))
            continue
        well = well_from_path(f)
        sub = chunk_dir / f"w{well}"
        sub.mkdir(exist_ok=True)
        chunks = abutils.io.split_fastx(fastx_file=str(f), output_directory=str(sub), chunksize=chunk_reads)
        for c in chunks:
            n = _count_reads(c)
            stats["input_reads"] += n
            pending[executor.submit(parse_barcodes, c, sub / "parsed", whitelist_path=whitelist_path)] = (well, c, n)
            while sum(v[2] for v in pending.values()) > max_in_flight:
                done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                collect(done)
    while pending:
        done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
        collect(done)
    flush()
    shutil.rmtree(chunk_dir, ignore_errors=True)
    stats["wells"] = sorted(stats["wells"])
    return stats


# ---------------------------------------------------------------------------------------------
#  UMIs and molecules
# ---------------------------------------------------------------------------------------------


def _neighbors(umi: str):
    for i, c in enumerate(umi):
        for b in "ACGTN":
            if b != c:
                yield umi[:i] + b + umi[i + 1:]


def directional_collapse(counts: dict[str, int]) -> dict[str, str]:
    """UMI-tools directional adjacency. ``counts``: reads per raw UMI. Returns raw -> corrected.

    Node a absorbs b when Hamming(a, b) == 1 and count(a) >= 2 * count(b) - 1; each group is the
    set reachable from its highest-count UMI along such edges (processed by decreasing count,
    ties broken by sequence), and every member maps to that UMI."""
    parent: dict[str, str] = {}
    for root in sorted(counts, key=lambda u: (-counts[u], u)):
        if root in parent:
            continue
        parent[root] = root
        stack = [root]
        while stack:
            a = stack.pop()
            ca = counts[a]
            for nb in _neighbors(a):
                if nb in counts and nb not in parent and ca >= 2 * counts[nb] - 1:
                    parent[nb] = root
                    stack.append(nb)
    return parent


def assign_molecule_wells(umis: list[str], wells: list[int]) -> dict[str, tuple[int, int, int]]:
    """For reads given as parallel lists of corrected UMI and well, return per molecule
    ``(main well, reads in main well, reads elsewhere)``; main well = most reads, ties -> lowest."""
    per = defaultdict(Counter)
    for u, w in zip(umis, wells):
        per[u][w] += 1
    out = {}
    for u, c in per.items():
        w, n = min(c.items(), key=lambda kv: (-kv[1], kv[0]))
        out[u] = (w, n, sum(c.values()) - n)
    return out


def contig_well_counts(umis: list[str], wells: list[int]) -> list[dict]:
    """Per well of one contig: ``umis`` (molecules assigned there), ``reads`` (all reads there,
    hopped copies included) and ``umis_present`` (molecules with >= 1 read there)."""
    mol = assign_molecule_wells(umis, wells)
    reads = Counter(wells)
    present = defaultdict(set)
    for u, w in zip(umis, wells):
        present[w].add(u)
    assigned = Counter(m[0] for m in mol.values())
    return [{"well": int(w), "umis": int(assigned.get(w, 0)), "reads": int(reads[w]), "umis_present": len(present[w])}
            for w in sorted(reads)]


def process_droplet_crosswell(
    barcode: str,
    reads: pl.DataFrame,
    clustering_threshold: float = 0.9,
    min_cluster_reads: int = 3,
    min_cluster_umis: int = 1,
    min_cluster_fraction: float = 0.0,
    umi_collapse: bool = True,
) -> dict:
    """Process all reads of one droplet barcode, pooled across wells (columns umi, well, sequence).

    Returns ``{"contigs": [...], "wells": [...], "stats": {...}}``. Contigs are named
    ``<barcode>_contig-<i>`` by decreasing cluster size, as legacy. ``umis`` counts corrected
    UMIs; ``pass_filters`` applies the legacy thresholds to pooled reads and corrected UMIs."""
    raw = reads["umi"].to_list()
    wells = reads["well"].to_list()
    seqs = reads["sequence"].to_list()
    # umi_collapse=False (diagnostic): a molecule is a raw (barcode, UMI)
    fix = directional_collapse(Counter(raw)) if umi_collapse else {u: u for u in raw}
    umis = [fix[u] for u in raw]

    mol = assign_molecule_wells(umis, wells)
    stats = {
        "reads": len(raw), "raw_umis": len(fix), "molecules": len(mol),
        "molecules_multiwell": sum(1 for m in mol.values() if m[2] > 0),
        "reads_off_main_well": sum(m[2] for m in mol.values()),
    }

    clusters = greedy_clustering(list(enumerate(seqs)), identity_threshold=clustering_threshold)
    clusters = sorted(clusters, key=lambda x: len(x), reverse=True)
    contigs, cw = [], []
    umi_contigs = defaultdict(set)
    for i, clust in enumerate(clusters):
        name = f"{barcode}_contig-{i}"
        cu = [umis[j] for j in clust]
        cwl = [wells[j] for j in clust]
        for u in set(cu):
            umi_contigs[u].add(i)
        n_umis = len(set(cu))
        frac = len(clust) / len(seqs)
        passed = len(clust) >= min_cluster_reads and n_umis >= min_cluster_umis and frac >= min_cluster_fraction
        rows = contig_well_counts(cu, cwl)
        consensus, _ = poa([seqs[j] for j in clust])
        nw = {r["well"]: r["umis"] for r in rows}
        main = min(nw.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        contigs.append({
            "name": name, "barcode": barcode, "reads": len(clust), "umis": n_umis,
            "raw_umis": len({raw[j] for j in clust}), "cluster_fraction": frac, "consensus": consensus,
            "pass_filters": passed, "n_wells": sum(1 for v in nw.values() if v > 0), "main_well": int(main),
        })
        cw.extend({"name": name, **r} for r in rows)
    stats["molecules_multicontig"] = sum(1 for s in umi_contigs.values() if len(s) > 1)
    return {"contigs": contigs, "wells": cw, "stats": stats}


# ---------------------------------------------------------------------------------------------
#  per-shard annotation and pairing
# ---------------------------------------------------------------------------------------------

_ANN_KEEP = ["sequence_id", "locus", "productive", "v_gene", "j_gene", "junction", "cdr3_aa", "sequence"]


def annotate_contigs(contigs: pl.DataFrame, fasta: Path, germline_database: str, receptor: str) -> pl.DataFrame | None:
    """Annotate passing contig consensus sequences with abstar (``mmseqs_threads=1``)."""
    ok = contigs.filter(pl.col("pass_filters"))
    if ok.height == 0:
        return None
    with open(fasta, "w") as f:
        for n, s in zip(ok["name"], ok["consensus"]):
            f.write(f">{n}\n{s}\n")
    ann = abstar.run(sequences=str(fasta), germline_database=germline_database, receptor=receptor,
                     mmseqs_threads=1, n_processes=available_cpus(), as_dataframe=True)
    os.remove(fasta)
    if ann is None or ann.height == 0:
        return None
    return ann


def chain_occurrences(contigs: pl.DataFrame, wells: pl.DataFrame) -> pl.DataFrame:
    """``pair_pooled`` input: one row per (productive IG contig, well with >= 1 assigned molecule),
    ``umis`` = molecules of the contig assigned to that well."""
    c = (contigs.filter(pl.col("pass_filters") & pl.col("productive").fill_null(False)
                        & pl.col("locus").is_in(["IGH", "IGK", "IGL"]) & pl.col("junction").is_not_null())
         .select("name", "barcode", "locus", "v_gene", "j_gene", "junction", "cdr3_aa",
                 pl.col("name").alias("sequence_id"), "sequence"))
    w = wells.filter(pl.col("umis") > 0).select("name", "well", pl.col("umis").cast(pl.Int64))
    return c.join(w, on="name").drop("name")


def species_totals(occ: pl.DataFrame) -> pl.DataFrame:
    """Per chain species: pooled molecules ``N`` and molecules in its modal well ``nmax``."""
    sw = occ.group_by(*SPECIES, "well").agg(pl.col("umis").sum().alias("n"))
    return sw.group_by(SPECIES).agg(pl.col("n").sum().alias("N"), pl.col("n").max().alias("nmax")).select("N", "nmax")


def finalize_pairs(pairs: pl.DataFrame) -> pl.DataFrame:
    """Add the contract's per-chain ``umis:0/1`` (= molecules at the home well)."""
    return pairs.with_columns(pl.col("umis_home:0").alias("umis:0"), pl.col("umis_home:1").alias("umis:1"))


# ---------------------------------------------------------------------------------------------
#  driver
# ---------------------------------------------------------------------------------------------


def _peak_rss_gb() -> dict:
    s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    return {"peak_rss_main_gb": round(s, 3), "peak_rss_largest_child_gb": round(c, 3)}


def run_crosswell(
    input_files: list[str],
    output_directory: Path,
    temp_directory: Path,
    run_log,
    whitelist_path: str | Path | None = None,
    clustering_threshold: float = 0.9,
    min_cluster_reads: int = 3,
    min_cluster_umis: int = 1,
    min_cluster_fraction: float = 0.0,
    receptor: str = "bcr",
    germline_database: str = "human",
    n_shards: int | None = None,
    umi_collapse: bool = True,
    quiet: bool = False,
    debug: bool = False,
) -> dict:
    """Run the cross-well pipeline (see module docstring) and finish ``run_log`` with its stats."""
    t0 = time.time()
    timings = {}
    work = Path(temp_directory) / "crosswell"
    shard_dir, ann_dir = work / "shards", work / "annotated"
    for d in (shard_dir, ann_dir):
        d.mkdir(parents=True, exist_ok=True)
    pairs_dir, contigs_dir = output_directory / "pairs", output_directory / "contigs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    contigs_dir.mkdir(parents=True, exist_ok=True)

    def say(msg):
        print(f"[crosswell {time.time() - t0:7.0f}s] {msg}", flush=True)

    n_shards = int(n_shards or auto_shard_count(input_files))
    say(f"{len(input_files)} input files, {n_shards} shards, {available_cpus()} workers")
    tot = Counter()
    leak_totals = []
    kwargs = dict(clustering_threshold=clustering_threshold, min_cluster_reads=min_cluster_reads,
                  min_cluster_umis=min_cluster_umis, min_cluster_fraction=min_cluster_fraction,
                  umi_collapse=umi_collapse)
    shard_files = []
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=available_cpus(), mp_context=mp.get_context("spawn")) as ex:
        # 1-2. parse and shard
        writer = ShardWriter(shard_dir, n_shards)
        try:
            pstats = parse_to_shards(input_files, writer, ex, work, whitelist_path)
        finally:
            shards = writer.close()
        W = len(pstats["wells"])
        timings["parse_s"] = round(time.time() - t0)
        say(f"parsed {pstats['valid_reads']}/{pstats['input_reads']} reads with valid barcodes from {W} wells")

        # 3-6. per shard: collapse, cluster, consensus, count, annotate
        t1 = time.time()
        for i in shards:
            df = pl.read_parquet(writer.path(i))
            df = df.filter(pl.len().over("barcode") >= min_cluster_reads)
            futures = [ex.submit(process_droplet_crosswell, bc, g, **kwargs)
                       for (bc,), g in df.partition_by("barcode", as_dict=True, include_key=False).items()]
            del df
            contigs, cwells = [], []
            for fut in as_completed(futures):
                r = fut.result()
                contigs.extend(r["contigs"])
                cwells.extend(r["wells"])
                tot.update(r["stats"])
                tot["barcodes"] += 1
            if not debug:
                os.remove(writer.path(i))
            if not contigs:
                continue
            cdf = pl.DataFrame(contigs)
            wdf = pl.DataFrame(cwells, schema={"name": pl.String, "well": pl.Int64, "umis": pl.Int64,
                                               "reads": pl.Int64, "umis_present": pl.Int64})
            ann = annotate_contigs(cdf, ann_dir / f"shard{i:04d}.fasta", germline_database, receptor)
            if ann is not None:
                cdf = cdf.join(ann.rename({c: f"{c}_contig" for c in ann.columns if c in cdf.columns and c != "sequence_id"}),
                               left_on="name", right_on="sequence_id", how="left", coalesce=False)
                cdf = cdf.drop("sequence_id")
            else:
                cdf = cdf.with_columns(*[pl.lit(None, pl.String).alias(c) for c in _ANN_KEEP if c not in ("sequence_id", "productive")],
                                       pl.lit(None, pl.Boolean).alias("productive"))
            cpath, wpath = ann_dir / f"contigs{i:04d}.parquet", ann_dir / f"wells{i:04d}.parquet"
            cdf.write_parquet(cpath)
            wdf.write_parquet(wpath)
            shard_files.append((cpath, wpath))
            tot["contigs"] += cdf.height
            tot["contigs_pass"] += int(cdf["pass_filters"].sum())
            tot["contigs_annotated"] += 0 if ann is None else ann.height
            occ = chain_occurrences(cdf, wdf)
            tot["chain_contigs"] += occ["sequence_id"].n_unique()
            leak_totals.append(species_totals(occ))
            say(f"shard {i}: {len(futures)} barcodes, {cdf.height} contigs")
        timings["contigs_s"] = round(time.time() - t1)

    # 7-8. leakage over the reaction, then pair shard by shard (a barcode lives in one shard)
    t2 = time.time()
    f_est = leakage_from_totals(pl.concat(leak_totals), W) if leak_totals else None
    f = 0.8 if f_est is None else f_est  # same documented fallback as pair_pooled
    pairs = []
    for cpath, wpath in shard_files:
        occ = chain_occurrences(pl.read_parquet(cpath), pl.read_parquet(wpath))
        if occ.height == 0:
            continue
        p, info = pair_pooled(occ, n_wells=W, leakage=f)
        tot["species"] += info["species"]
        if p.height:
            pairs.append(p)
    pairs_df = finalize_pairs(pl.concat(pairs, how="diagonal_relaxed").sort("confidence", descending=True)) if pairs else None
    pairs_file = pairs_dir / "pairs.parquet"
    if pairs_df is None:
        pairs_df = pl.DataFrame()
    pairs_df.write_parquet(pairs_file)

    # 9. outputs
    if shard_files:
        pl.concat([pl.scan_parquet(c) for c, _ in shard_files], how="diagonal_relaxed").sink_parquet(contigs_dir / "contigs.parquet")
        pl.concat([pl.scan_parquet(w) for _, w in shard_files]).sink_parquet(contigs_dir / "contig_wells.parquet")
    timings["pairing_s"] = round(time.time() - t2)
    if not debug:
        shutil.rmtree(work, ignore_errors=True)
        if temp_directory.is_dir() and not os.listdir(temp_directory):
            os.rmdir(temp_directory)

    reads = max(tot["reads"], 1)
    stats = {
        "pooling": "across_wells",
        "umi_collapse": umi_collapse,
        "input_files": len(input_files),
        "wells": W,
        "shards": n_shards,
        "input_reads": pstats["input_reads"],
        "valid_barcodes": pstats["valid_reads"],
        "skipped_files": len(pstats["skipped_files"]),
        "barcodes_processed": tot["barcodes"],
        "reads_processed": tot["reads"],
        "raw_umis": tot["raw_umis"],
        "molecules": tot["molecules"],
        "molecules_multiwell_frac": round(tot["molecules_multiwell"] / max(tot["molecules"], 1), 4),
        "reads_off_main_well_frac": round(tot["reads_off_main_well"] / reads, 4),
        "molecules_multicontig_frac": round(tot["molecules_multicontig"] / max(tot["molecules"], 1), 4),
        "consensus_sequences": tot["contigs_pass"],
        "contigs": tot["contigs"],
        "contigs_annotated": tot["contigs_annotated"],
        "chain_contigs": tot["chain_contigs"],
        "species": tot["species"],
        "leakage": f,
        "leakage_estimated": f_est is not None,
        "pairs": pairs_df.height,
        "pairs_conf_ge_0.9": int((pairs_df["confidence"] >= 0.9).sum()) if pairs_df.height else 0,
        **timings,
        "total_s": round(time.time() - t0),
        **_peak_rss_gb(),
    }
    say(" ".join(f"{k}={v}" for k, v in stats.items()))
    run_log.finish(stats=stats)
    return stats
