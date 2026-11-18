# Copyright (c) 2025 Benjamin Nemoz
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

# This file is part of PairPlex.
# PairPlex is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# PairPlex is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with PairPlex. If not, see <http://www.gnu.org/licenses/>.


import multiprocessing as mp
import os
import time
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import abstar
import abutils
import polars as pl
from natsort import natsorted
from tqdm.auto import tqdm

from .pairing import (
    DEFAULT_MIN_DOMINANCE,
    PAIRING_MODES,
    contract_frame,
    pair_well,
    support_from_metadata,
    well_number,
    write_contract_pairs,
)
from . import qc as qc_mod
from .provenance import RunLog, run_log_path
from .utils import available_cpus, parse_barcodes, print_splash, process_droplet

######################################################
##                Main function                     ##
######################################################


def run(
    sequences: str | Path | list[str | Path],
    output_directory: str | Path,
    temp_directory: str | Path | None = None,
    whitelist_path: str | Path | None = None,
    platform: str = "illumina",
    clustering_threshold: float = 0.9,
    min_cluster_reads: int = 3,
    min_cluster_umis: int = 1,
    min_cluster_fraction: float = 0.0,
    consensus_downsample: int = 100,
    merge_paired_reads: bool = False,
    receptor: str = "bcr",
    germline_database: str = "human",
    mmseqs_threads: int = 1,
    tso_g_max: int = 3,
    barcode_prefer_exact: bool = True,
    umi_collapse: str = "directional",
    pairing: str = "scored",
    min_dominance: float = DEFAULT_MIN_DOMINANCE,
    collapse_clonal: bool = True,
    legacy: bool = False,
    quiet: bool = False,
    debug: bool = False,
    support_columns: bool = True,
    qc: bool = True,
    pooling: str = "per_well",
    crosswell_shards: int | None = None,
) -> list:
    """
    Run PairPlex.

    Parameters
    ----------
    sequences : str | Path | list[str | Path]
        Path to the sequencing data or a list of paths. If a directory, all files in the directory will be processed
        (recursively). If a file, only that file will be processed. If a list of paths, each path will be processed.

    output_directory : str | Path
        Path to the output directory. If it does not exist, it will be created.

    temp_directory : str | Path | None
        Path to the temporary directory. If it does not exist, it will be created. If None, a temporary sub-directory will
        be created in the output directory.

    whitelist_path : str | Path | None
        Path to the whitelist file. If None, the default whitelist will be used.

    platform : str
        The sequencing platform used to generate the data. Options are "illumina" or "element".

    clustering_threshold : float
        The clustering threshold for the consensus sequences. For each droplet barcode (which potentially contains both
        heavy and light chains), sequences are clustered by sequence similarity prior to consensus sequence generation
        so that consensus sequences are separately generated for each antibody chain.

    min_cluster_reads : int
        The minimum number of reads to consider a cluster for consensus sequence generation. Clusters containing fewer
        reads are discarded.

    min_cluster_umis : int
        The minimum number of UMIs to consider a cluster for consensus sequence generation. Clusters containing fewer
        UMIs are discarded. Defaults to ``1``. For datasets with heavy ambient contamination, a value of ``2`` is
        recommended.

    min_cluster_fraction : float
        The minimum fraction of reads to consider a cluster for consensus sequence generation. Clusters containing fewer
        reads than this fraction are discarded.

    consensus_downsample : int
        The number of sequences to downsample to for consensus sequence generation.

    merge_paired_reads : bool
        Whether to merge paired reads. If ``True``, i

    receptor : str
        The receptor to use. Options are "bcr" or "tcr".

    germline_database : str
        The germline database to use. Options are anything accepted by `abstar.run`, including
        custom databases if available.

    mmseqs_threads : int
        Number of threads MMseqs2 uses inside ``abstar.run``. Defaults to ``1``. Always passed explicitly: with
        ``None`` MMseqs2 uses every online CPU of the node (ignoring ``OMP_NUM_THREADS`` and CPU affinity) and
        crashes when its thread count exceeds the size of an abstar sub-query batch. Annotation is not the
        bottleneck of a run, so ``1`` costs little. Does not change results.

    tso_g_max : int
        Maximum number of Gs removed from the start of the insert, after the fixed TSO prefix
        ``TTTCTTATAT`` (read positions 26-35). The 10x TSO ends in rGrGrG, so the default is ``3``;
        genuine 5' Gs of the transcript beyond the TSO's run are kept. v2-base removed every leading
        G (``lstrip("G")``).

    barcode_prefer_exact : bool
        Read orientation choice. If ``True`` (default), an exact barcode whitelist hit in either orientation
        (forward or reverse complement) wins over a Hamming-1 correction in the other; corrections are used only
        when neither orientation is exact. Ties keep the forward orientation. Orientation counts are written to
        the run log (``stats.barcode_orientation``). v2-base took the first orientation whose barcode corrected.

    umi_collapse : str
        UMI error correction, applied within each (well, droplet barcode) before any UMI counting.
        ``"directional"`` (default) uses the UMI-tools directional adjacency method: UMI ``b`` is
        absorbed by UMI ``a`` when they differ at one position and ``reads(a) >= 2 * reads(b) - 1``,
        transitively, processing UMIs by descending read count (see :mod:`pairplex.umi`, which also
        documents how UMIs containing ``N`` are handled). The metadata ``umis`` column is then the
        corrected count, a ``umis_raw`` column holds the count before collapsing, and
        ``min_cluster_umis`` filters on corrected counts. ``"none"`` counts raw UMIs (v2-base
        behaviour, no ``umis_raw`` column).

    pairing : str
        How heavy and light chains of a (well, barcode) are paired. ``"scored"`` (default): rank
        the productive heavy chains and the productive light chains (IGK and IGL together) by
        UMIs (then reads) and emit the top heavy + top light when each group passes
        ``min_dominance``; every pair gets a ``confidence`` and a ``margin`` (see
        ``pairplex.pairing``), and all wells are also written to ``pairs/pairs.parquet``.
        ``"exact"``: the ``v2-base`` rule, keep only barcodes with exactly one heavy and one
        light chain; no ``pairs/pairs.parquet``.

    min_dominance : float
        Scored pairing only. The top chain of each locus group must have at least this many
        times the UMIs of the second-best chain. Defaults to ``2.0`` (top chain holds >= 2/3 of
        a 2-chain group). Values <= 1 accept any split (ties broken by reads).

    collapse_clonal : bool
        Scored pairing only. Before ranking, merge same-locus contigs of a (well, barcode) that
        share V gene, J gene and CDR3 aa (one chain split into several consensus clusters, or
        clonal relatives), summing their UMIs, so they do not compete. Defaults to ``True``.

    legacy : bool
        If ``True``, reproduce v2-base exactly: forces ``tso_g_max`` to strip every leading G
        (``lstrip("G")``), ``barcode_prefer_exact=False``, ``umi_collapse="none"``, ``support_columns=False``,
        ``qc=False``, ``pairing="exact"`` and ``pooling="per_well"``.

    quiet : bool
        Whether to suppress the progress bar and other update messages.

    debug : bool
        If ``True``, all temporary files will be retained to ease debugging.

    support_columns : bool
        If ``True`` (default), the annotated outputs carry the cluster support of each contig:
        ``umis``, ``reads``, ``cluster_fraction``, ``barcode`` and ``contig`` in the unpaired
        outputs, ``umis:0/1``, ``reads:0/1``, ``cluster_fraction:0/1`` (etc.) in the paired outputs.
        abstar's ``umi`` column, which is always null for PairPlex consensus contigs, is dropped.
        Pairing is unchanged.

    qc : bool
        If ``True`` (default), write ``qc/stage_counts.csv`` (drop counts at every stage, per
        well and in total), ``qc/annotation_failures.csv`` (abstar failure and exclusion modes)
        and ``qc/report.json`` / ``qc/report.md`` (fragment fraction, UMI error signature,
        ambient proxy, leakage by chain, pair confidence, clonal light-chain coherence).
        Pairing is unchanged.

    pooling : str
        ``"per_well"`` (default): process each input file (one well) on its own, as before.
        ``"across_wells"``: cross-well mode (see ``pairplex.crosswell``). Reads of all wells of a
        reaction are pooled by droplet barcode (well = last integer in each file stem), UMIs are
        collapsed per barcode across wells (UMI-tools directional adjacency), each barcode's reads
        are clustered once with the legacy greedy clustering, each molecule is assigned to the
        well holding most of its reads, and heavy/light are paired by shared home well with
        ``pooled_pairing.pair_pooled`` (leakage estimated from the data). In this mode
        ``min_cluster_umis`` counts corrected UMIs and ``min_cluster_reads`` /
        ``min_cluster_fraction`` apply to pooled reads. Writes ``pairs/pairs.parquet``,
        ``contigs/contigs.parquet`` and ``contigs/contig_wells.parquet`` instead of the per-well
        ``parsed/``, ``metadata/``, ``consensus/`` and ``annotated/`` files. The per-well options
        ``umi_collapse``, ``pairing``, ``min_dominance``, ``collapse_clonal``, ``support_columns``
        and ``qc`` do not apply in this mode.

    crosswell_shards : int | None
        Cross-well mode only: number of on-disk barcode shards (memory scales with one shard).
        ``None`` picks about one shard per 160 MB of gzipped input (~1M reads).


    Returns
    -------
    list
        A list of paths to the output files.

    """
    parameters = dict(locals())  # captured first: exactly the arguments of this call
    if not isinstance(mmseqs_threads, int) or mmseqs_threads < 1:
        raise ValueError(f"mmseqs_threads must be a positive integer, got {mmseqs_threads!r}")
    if legacy:
        tso_g_max = None  # v2-base: lstrip("G")
        barcode_prefer_exact = False  # v2-base: first orientation that corrects
    elif not isinstance(tso_g_max, int) or tso_g_max < 0:
        raise ValueError(f"tso_g_max must be a non-negative integer, got {tso_g_max!r}")

    # legacy mode forces every v2 behaviour option to its v2-base value
    if legacy:
        umi_collapse = "none"
        support_columns = False
        qc = False
        pairing = "exact"
        pooling = "per_well"
    if umi_collapse not in ("directional", "none"):
        raise ValueError(
            f"umi_collapse must be 'directional' or 'none', got {umi_collapse!r}"
        )
    parameters["umi_collapse_effective"] = umi_collapse
    if pairing not in PAIRING_MODES:
        raise ValueError(f"pairing must be one of {PAIRING_MODES}, got {pairing!r}")
    contract_frames = []
    if pooling not in ("per_well", "across_wells"):
        raise ValueError(f"pooling must be 'per_well' or 'across_wells', not {pooling!r}")
    per_well = pooling == "per_well"

    # print splash screen
    if not quiet:
        print_splash(include_version=True)

    # setup directories
    output_directory = Path(output_directory).resolve()
    temp_directory = (
        Path(temp_directory).resolve()
        if temp_directory is not None
        else output_directory / "temp"
    )
    log_directory = output_directory / "logs"
    parsed_directory = output_directory / "parsed"
    consensus_directory = output_directory / "consensus"
    metadata_directory = output_directory / "metadata"
    annotated_directory = output_directory / "annotated"
    output_directory.mkdir(parents=True, exist_ok=True)
    temp_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    if per_well:
        parsed_directory.mkdir(parents=True, exist_ok=True)
        consensus_directory.mkdir(parents=True, exist_ok=True)
        metadata_directory.mkdir(parents=True, exist_ok=True)
        annotated_directory.mkdir(parents=True, exist_ok=True)

    # process input files
    if isinstance(sequences, str | Path):
        sequences = Path(sequences).resolve()
        if sequences.is_dir():
            input_files = abutils.io.list_files(
                str(sequences),
                recursive=True,
                extension=[
                    "fastq.gz",
                    "fq.gz",
                    "fastq",
                    "fq",
                    "fasta.gz",
                    "fa.gz",
                    "fasta",
                    "fa",
                ],
            )
        elif sequences.is_file():
            input_files = [str(sequences)]
        else:
            raise FileNotFoundError(
                f"string/path input must be a directory or file: {sequences}"
            )
    elif isinstance(sequences, list):
        input_files = [str(Path(f).resolve()) for f in sequences]
    else:
        raise ValueError(f"Invalid input type: {type(sequences)}")
    input_files = [f for f in input_files if "Unassigned" not in f]

    # provenance: version, git commit, dependencies, parameters and inputs of this run
    parameters["max_workers"] = available_cpus()
    run_log = RunLog(
        run_log_path(log_directory),
        tool="pairplex run",
        parameters=parameters,
        inputs=input_files,
    )

    # merge paired reads
    if merge_paired_reads:
        merge_directory = output_directory / "merged"
        merge_log_directory = log_directory / "merge"
        merge_directory.mkdir(parents=True, exist_ok=True)
        merge_log_directory.mkdir(parents=True, exist_ok=True)
        input_files = abstar.pp.merge_fastqs(
            files=input_files,
            output_directory=merge_directory,
            log_directory=merge_log_directory,
            schema=platform.lower(),
            debug=debug,
            show_progress=True,
        )
        print("\n")

    if not per_well:
        from .crosswell import run_crosswell

        run_crosswell(
            input_files,
            output_directory,
            temp_directory,
            run_log,
            whitelist_path=whitelist_path,
            clustering_threshold=clustering_threshold,
            min_cluster_reads=min_cluster_reads,
            min_cluster_umis=min_cluster_umis,
            min_cluster_fraction=min_cluster_fraction,
            receptor=receptor,
            germline_database=germline_database,
            n_shards=crosswell_shards,
            quiet=quiet,
            debug=debug,
            tso_g_max=tso_g_max,
            barcode_prefer_exact=barcode_prefer_exact,
            mmseqs_threads=mmseqs_threads,
        )
        return

    # setup the main progress bar (tracks input file completion)
    main_pbar = tqdm(
        total=len(input_files),
        desc="pairplex",
        position=0,
        leave=True,
        # dynamic_ncols=True,
        bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}",
    )
    blank1_printer = tqdm(total=0, bar_format=" ", position=1, leave=True)
    running_total_printer = tqdm(total=0, bar_format="{desc}", position=2, leave=True)
    blank2_printer = tqdm(total=0, bar_format=" ", position=3, leave=True)
    all_valid_barcodes = 0
    orientation_counts = {}
    all_consensus_sequences = 0
    all_pairs = 0
    running_total_printer.set_description_str(
        f"valid barcodes: {all_valid_barcodes} | consensus sequences: {all_consensus_sequences} | pairs: {all_pairs}"
    )

    # QC accumulators (qc=True): per-well stage counts, abstar failure modes, fragments, UMI stats
    qc_rows, qc_failures, qc_fragments, qc_umis, qc_wells = [], [], {}, [], []

    # initialize the Process Pool
    with ProcessPoolExecutor(
        max_workers=available_cpus(), mp_context=mp.get_context("spawn")
    ) as executor:
        for input_file in natsorted(input_files):
            # Checking for empty file, as this will break later steps
            if os.path.getsize(input_file) == 0:
                main_pbar.set_postfix_str("partitioning barcodes", refresh=True)
                continue

            to_delete = []

            # setup text printers (using tqdm so they get cleared once file is processed)
            name_printer = tqdm(total=0, bar_format="{desc}", position=4, leave=False)
            seqs_printer = tqdm(total=0, bar_format="{desc}", position=5, leave=False)
            valids_printer = tqdm(total=0, bar_format="{desc}", position=6, leave=False)

            # process the input file
            input_file = Path(input_file)
            name = input_file.stem
            name_printer.set_description_str(f"---- {name} ----")
            if qc:
                wc = qc_mod.new_counts(name)
                qc_rows.append(wc)
                qc_wells.append(name)

            # check to make sure the input file is FASTA/Q (and not empty)
            if abutils.io.determine_fastx_format(str(input_file)) is None:
                # file is either empty or not FASTA/Q, so we skip it
                seqs_printer.set_description_str("0 input sequences")
                # close out progress bars
                time.sleep(2)
                name_printer.close()
                seqs_printer.close()
                valids_printer.close()
                main_pbar.update(1)
                continue

            # count sequences
            input_count = 0
            for s in abutils.io.parse_fastx(str(input_file)):
                input_count += 1
            seqs_printer.set_description_str(f"{input_count} input sequences")
            if qc:
                wc["input_reads"] = wc["dropped_orientation_or_barcode"] = input_count

            # split input file into chunks
            main_pbar.set_postfix_str("splitting input file", refresh=True)
            fastq_chunks = abutils.io.split_fastx(
                fastx_file=str(input_file),
                output_directory=str(temp_directory),
                chunksize=1000,
            )
            to_delete.extend(fastq_chunks)

            # --------------------
            #      barcodes
            # --------------------

            main_pbar.set_postfix_str("parsing barcodes", refresh=True)
            parquet_chunks = []
            parse_kwargs = {"whitelist_path": whitelist_path, "tso_g_max": tso_g_max}
            if barcode_prefer_exact:
                parse_kwargs.update(prefer_exact=True, return_counts=True)
            else:
                parse_kwargs.update(prefer_exact=False)
            futures = [
                executor.submit(parse_barcodes, chunk, temp_directory, **parse_kwargs)
                for chunk in fastq_chunks
            ]
            for future in as_completed(futures):
                res = future.result()
                if barcode_prefer_exact:
                    res, counts = res
                    for k, v in counts.items():
                        orientation_counts[k] = orientation_counts.get(k, 0) + v
                if res is not None:
                    parquet_chunks.append(res)
            to_delete.extend(parquet_chunks)

            # concatenate parsed data into a single dataframe
            if parquet_chunks == []:
                valids_printer.set_description_str("0 sequences with valid barcodes")
                running_total_printer.set_description_str(
                    f"valid barcodes: {all_valid_barcodes} | consensus sequences: {all_consensus_sequences} | pairs: {all_pairs}"
                )
                continue
            concat_parquet = abutils.io.concatenate_parquet(
                parquet_chunks, parsed_directory / f"{name}.parquet"
            )
            df = pl.read_parquet(concat_parquet)
            seqs_with_barcodes = df.shape[0]
            if qc:
                wc.update(qc_mod.read_stage_counts(input_count, df))
                wc.update(qc_mod.barcode_stage_counts(df, min_cluster_reads))
                qc_umis.append(qc_mod.umi_hamming1_stats(df))
            all_valid_barcodes += seqs_with_barcodes
            valids_printer.set_description_str(
                f"{seqs_with_barcodes} sequences with valid barcodes"
            )
            running_total_printer.set_description_str(
                f"valid barcodes: {all_valid_barcodes} | consensus sequences: {all_consensus_sequences} | pairs: {all_pairs}"
            )

            # partition into separate parquet files by barcode
            main_pbar.set_postfix_str("partitioning barcodes", refresh=True)
            partitions = df.partition_by("barcode", as_dict=True)

            # filter partitinos by size
            partitions = {
                k: v for k, v in partitions.items() if v.shape[0] >= min_cluster_reads
            }

            # filter barcode groups by size before partitioning
            # df = df.filter(
            #     pl.col("barcode").count().over("barcode") >= min_cluster_reads
            # )
            # partitions = df.partition_by("barcode", as_dict=True)

            # --------------------
            #      consensus
            # --------------------

            # setup the consensus progress bar and printer
            consensus_pbar = tqdm(
                total=len(partitions),
                # desc="consensus sequences",
                position=7,
                leave=False,
                # dynamic_ncols=True,
                bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}",
            )
            consensus_printer = tqdm(
                total=0, bar_format="{desc}", position=8, leave=False
            )

            # make consensus sequences for each droplet
            futures = []
            consensus_kwargs = {
                "temp_directory": temp_directory,
                "min_cluster_reads": min_cluster_reads,
                "min_cluster_umis": min_cluster_umis,
                "min_cluster_fraction": min_cluster_fraction,
                "consensus_downsample": consensus_downsample,
                "clustering_threshold": clustering_threshold,
                "umi_collapse": umi_collapse,
                "quiet": quiet,
                "debug": debug,
            }
            for bc, bc_df in partitions.items():
                if isinstance(bc, tuple):
                    bc = bc[0]
                futures.append(
                    executor.submit(
                        process_droplet,
                        name=bc,
                        partition_df=bc_df,
                        **consensus_kwargs,
                    )
                )

            # collect metadata
            metadata = []
            droplet_count = 0
            consensus_count = 0
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    metadata.extend(res)
                    droplet_count += 1
                    consensus_count += len([r for r in res if r["pass_filters"]])
                consensus_pbar.update(1)
                consensus_printer.set_description_str(
                    f"{droplet_count} droplet barcodes | {consensus_count} consensus sequences"
                )
            metadata_df = pl.DataFrame(metadata)
            consensus_pbar.close()
            if qc:
                wc.update(
                    qc_mod.cluster_stage_counts(
                        metadata_df, min_cluster_reads, min_cluster_umis, min_cluster_fraction
                    )
                )

            # write metadata to file
            metadata_file = metadata_directory / f"{name}.csv"
            metadata_df.write_csv(metadata_file)

            # write consensus sequences to file
            consensus_file = consensus_directory / f"{name}.fasta"
            if metadata_df.is_empty():
                consensus_printer.set_description_str(
                    f"{droplet_count} droplet barcodes | 0 consensus sequences"
                )
                consensus_file.touch()
                continue
            filtered_df = metadata_df.filter(pl.col("pass_filters"))
            with consensus_file.open("w") as f:
                for _name, _consensus in zip(
                    filtered_df["name"], filtered_df["consensus"]
                ):
                    f.write(f">{_name}\n{_consensus}\n")
            consensus_printer.set_description_str(
                f"{droplet_count} droplet barcodes | {filtered_df.shape[0]} consensus sequences"
            )

            # --------------------
            #     annotation
            # --------------------

            pairs_printer = None
            main_pbar.set_postfix_str("annotating sequences", refresh=True)
            # failsafe for empty consensus files
            if consensus_count <= 1 or os.path.getsize(consensus_file) == 0: # Changed consensus_count to include 1 (a single sequence can't be paired)
                continue
            consensus_count = filtered_df.shape[0]
            # run abstar (always with an explicit MMseqs2 thread count, see docstring)
            with qc_mod.capture_abstar_failures() if qc else nullcontext([]) as record_failures:
                sequences = abstar.run(
                    sequences=str(consensus_file),
                    germline_database=germline_database,
                    receptor=receptor,
                    mmseqs_threads=mmseqs_threads,
                )
            if qc:
                ann_df = qc_mod.annotation_frame(sequences)
                n_rec = len(record_failures)
                wc.update(qc_mod.annotation_stage_counts(ann_df, consensus_count, n_rec))
                qc_failures.extend(qc_mod.annotation_failure_rows(name, ann_df, record_failures))
                qc_mod.merge_fragment_counts(qc_fragments, qc_mod.fragment_counts(ann_df, n_rec))
            if support_columns:
                qc_mod.add_support_columns(sequences, metadata_df)

            if len(sequences) > 1: # A single sequence cannot be paired and will cause issues
                
                # unpaired sequences
                unpaired_airr_file = annotated_directory / f"{name}_unpaired.tsv"
                unpaired_parquet_file = annotated_directory / f"{name}_unpaired.parquet"
                abutils.io.to_airr(
                    sequences, str(unpaired_airr_file), drop_na_columns=False
                )
                abutils.io.to_parquet(
                    sequences, str(unpaired_parquet_file), drop_na_columns=False
                )
                all_consensus_sequences += len(sequences)
                running_total_printer.set_description_str(
                    f"valid barcodes: {all_valid_barcodes} | consensus sequences: {all_consensus_sequences} | pairs: {all_pairs}"
                )

                # paired sequences
                main_pbar.set_postfix_str("identifying pairs", refresh=True)
                pairs_printer = tqdm(total=0, bar_format="{desc}", position=9, leave=False)
                paired_airr_file = annotated_directory / f"{name}_paired.tsv"
                paired_parquet_file = annotated_directory / f"{name}_paired.parquet"
                if qc:  # chain composition before any pairing rule
                    wc.update(qc_mod.chain_composition_counts(
                        abutils.tl.assign_pairs(sequences, delim="_", delim_occurance=-1)))
                pairs = pair_well(
                    sequences,
                    paired_airr_file,
                    paired_parquet_file,
                    mode=pairing,
                    support=support_from_metadata(metadata_df),
                    min_dominance=min_dominance,
                    collapse_clonal=collapse_clonal,
                )
                if qc:
                    wc["emitted_pairs"] = len(pairs)
                pairs_printer.set_description_str(f"{len(pairs)} paired sequences")
                if pairing == "scored":
                    contract_frames.append(
                        contract_frame(paired_parquet_file, well_number(name))
                    )

                all_pairs += len(pairs)
                running_total_printer.set_description_str(
                    f"valid barcodes: {all_valid_barcodes} | consensus sequences: {all_consensus_sequences} | pairs: {all_pairs}"
                )

            # --------------------
            #      cleanup
            # --------------------

            if not debug:
                main_pbar.set_postfix_str("cleaning up", refresh=True)
                for f in to_delete:
                    if f is not None:
                        if os.path.exists(f):
                            os.remove(f)
                if os.path.isdir(temp_directory) and not os.listdir(temp_directory):
                    os.rmdir(temp_directory)

            # close out sub-progress bars
            time.sleep(2)
            name_printer.close()
            seqs_printer.close()
            valids_printer.close()
            consensus_printer.close()
            if pairs_printer is not None:
                pairs_printer.close()

            # update the main progress bar
            main_pbar.update(1)

    n_contract_pairs = None
    if pairing == "scored":
        n_contract_pairs = write_contract_pairs(
            contract_frames, output_directory / "pairs" / "pairs.parquet"
        )
    if qc:
        qc_mod.write_qc(
            output_directory, qc_rows, qc_failures, qc_fragments, qc_umis, qc_wells
        )

    run_log.finish(
        stats={
            "input_files": len(input_files),
            "valid_barcodes": all_valid_barcodes,
            "consensus_sequences": all_consensus_sequences,
            "pairs": all_pairs,
            **({"pairs_parquet_rows": n_contract_pairs} if n_contract_pairs is not None else {}),
            **({"barcode_orientation": orientation_counts} if barcode_prefer_exact else {}),
        }
    )

    # return partition_files
