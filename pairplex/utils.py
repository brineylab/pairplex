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


import logging
import multiprocessing as mp
import os
import re

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Set

import abutils
import pandas as pd
import polars as pl
from abstar.preprocess import merging
from abutils import Sequence

from abutils.io import from_pandas, from_polars, make_dir, parse_fastx
from abutils.tools import cluster
from natsort import natsorted
from tqdm.auto import tqdm

BARCODE_DIR = Path("./barcodes/")
DEFAULT_WHITELIST = BARCODE_DIR / "737K-august-2016.txt"


def setup_logger(output_folder: str, verbose: bool, debug: bool) -> logging.Logger:
    """Set up a logger with proper formatting and both file and console handlers."""

    logger = logging.getLogger("PairPlex")
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    log_path = os.path.join(output_folder, "pairplex.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def merge(
    files: list,
    output_folder: str,
    log_folder: str,
    schema: str,
    verbose: bool,
    debug: bool,
) -> list:
    """Quick and dirty adapter to leverage merging wrapper from AbStar. Uses Fastp."""

    global logger

    assert isinstance(files, list), "Incorrect list of files to merge. Aborting."
    assert files != [], "List of files to merge is empty. Aborting"

    merge_dir = os.path.join(output_folder, "01_merged")
    log = os.path.join(log_folder, "merging")
    make_dir(merge_dir)

    logging.info(f"Merging of {len(files)} files into {merge_dir}")

    merged_files = merging.merge_fastqs(
        files=files,
        output_folder=merge_dir,
        output_format="fastq",
        log_directory=log,
        schema=schema,
        algo="fastp",
        binary_path=None,
        merge_args=None,
        minimum_overlap=30,
        allowed_mismatches=5,
        allowed_mismatch_percent=20.0,
        trim_adapters=True,
        adapter_file=None,
        quality_trim=True,
        window_size=4,
        quality_cutoff=20,
        interleaved=False,
        compress_output=False,
        debug=False,
        show_progress=debug,
    )
    return merged_files


def list_wells(merged_files: list, verbose: bool, debug: bool) -> dict:
    """Match input files to wells (e.g., A1, B9) and return mapping."""
    global logger

    well_to_files = {}

    for f in merged_files:
        match = re.search(r"VDJ_([A-H][0-9]{1,2})\.fastq", f)
        if match:
            well = match.group(1)
            well_to_files[well] = f

    if any([verbose, debug]):
        logger.info(f"Found {len(well_to_files)} wells")
    if debug:
        sorted_wells = natsorted(well_to_files)
        for well in sorted_wells:
            logger.debug(f"Well {well}: {well_to_files[well]}")

    return well_to_files


def get_builtin_whitelist(whitelist_name: str) -> str:
    """Get the path to a builtin barcode whitelist."""
    builtin_whitelists = {
        "v2": BARCODE_DIR / "737K-august-2016.txt",
        "v3": BARCODE_DIR / "3M-5pgex-jan-2023.txt",
        "5v2": BARCODE_DIR / "737K-august-2016.txt",
        "5v3": BARCODE_DIR / "3M-5pgex-jan-2023.txt",
        "5pv2": BARCODE_DIR / "737K-august-2016.txt",
        "5pv3": BARCODE_DIR / "3M-5pgex-jan-2023.txt",
        "5primev2": BARCODE_DIR / "737K-august-2016.txt",
        "5primev3": BARCODE_DIR / "3M-5pgex-jan-2023.txt",
        "5'v2": BARCODE_DIR / "737K-august-2016.txt",
        "5'v3": BARCODE_DIR / "3M-5pgex-jan-2023.txt",
        "737k": BARCODE_DIR / "737K-august-2016.txt",
        "3m": BARCODE_DIR / "3M-5pgex-jan-2023.txt",
        "nextgem": BARCODE_DIR / "737K-august-2016.txt",
        "gemx": BARCODE_DIR / "3M-5pgex-jan-2023.txt",
    }
    if whitelist_name.lower() not in builtin_whitelists:
        raise ValueError(f"Invalid whitelist name: {whitelist_name}")
    return builtin_whitelists[whitelist_name.lower()]


def load_barcode_whitelist(whitelist_path: str) -> Set[str]:
    """
    Load a barcode whitelist from a file and return a set of barcodes.

    Parameters
    ----------
    whitelist_path : str
        The path to the barcode whitelist file.

    Returns
    -------
    set[str]
        A set of barcodes.

    Raises
    ------
    FileNotFoundError
        If the barcode whitelist file does not exist.

    """
    whitelist_path = Path(whitelist_path)
    if not whitelist_path.exists():
        raise FileNotFoundError(f"Barcode whitelist file not found: {whitelist_path}")
    with whitelist_path.open() as f:
        return set(line.strip() for line in f)


def correct_barcode(
    barcode: str,
    valid_barcodes: set[str],
) -> str | None:
    """
    Correct a barcode by checking against a set of valid barcodes.

    Parameters
    ----------
    barcode : str
        The barcode to correct.

    valid_barcodes : set[str]
        The set of valid barcodes.

    Returns
    -------
    str | None
        The corrected barcode or None if no correction is possible.

    """
    if barcode in valid_barcodes:
        return barcode
    # check every single-nucleotide variant (Hamming distance == 1)
    matches = []
    for i in range(len(barcode)):
        for mut in "ATGC":
            if mut == barcode[i]:  # skip the original barcode
                continue
            corrected = barcode[:i] + mut + barcode[i + 1 :]
            if corrected in valid_barcodes:
                matches.append(corrected)
    # return the corrected barcode only if a single matching mutant is found
    if len(matches) == 1:
        return matches[0]


def parse_barcodes(
    input_file: str | Path,
    output_directory: str | Path,
    whitelist_path: str | Path | None = None,
    check_rc: bool = True,
) -> str | None:
    """
    Process a chunk of fastq file to extract barcodes, UMIs and TSO sequences.

    Parameters
    ----------
    input_file : list
        The input file, in FASTA/Q format (optionally gzip-compressed).

    chunk_id : str | int
        The ID of the chunk.

    output_directory : str | Path
        The directory to save the output.

    barcodes_path : str | Path | None = None
        The path to the barcode file.

    tso_pattern : str = r"TTTCTTATATG{1,5}"
        The pattern to search for the TSO sequence.

    check_rc : bool = True
        Whether to check the reverse complement of the sequences.

    enforce_bc_whitelist : bool = True
        Whether to enforce the barcode whitelist.

    Returns
    -------
    str | None
        The path to the output file (or None if no records are found).
    """

    input_file = Path(input_file)
    output_name = input_file.stem
    if whitelist_path is None:
        whitelist_path = DEFAULT_WHITELIST
    whitelist = load_barcode_whitelist(whitelist_path)

    records = []
    for seq in abutils.io.parse_fastx(str(input_file)):
        seqs = [seq.sequence]
        if check_rc:
            seqs.append(abutils.tl.reverse_complement(seq.sequence))
        for s in seqs:
            # # parse barcode and UMI
            sequence = s[36:].lstrip("G")  # remove any remaining Gs from the TSO
            barcode = s[:16]
            umi = s[16:26]
            corrected = correct_barcode(barcode, whitelist, allowed_mismatches=1)
            if corrected is None:
                continue
            # build the record
            records.append(
                {
                    "umi": umi,
                    "barcode": corrected,
                    "seq_id": seq.id,
                    "sequence": sequence,
                }
            )
            # all done!
            break

    if records:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_file = output_directory / f"bcdf_{output_name}.parquet"
        df = pl.DataFrame(records)
        df.write_parquet(output_file)
        return str(output_file)


def process_droplets(
    partition_files: list[str | Path],
    output_directory: str | Path,
    temp_directory: str | Path,
    clustering_threshold: float = 0.85,
    consensus_downsample: int = 100,
    min_cluster_reads: int = 3,
    min_cluster_umis: int = 2,
    n_processes: int | None = None,
    quiet: bool = False,
    debug: bool = False,
) -> dict:
    """Process PairPlex droplets to cluster sequences and generate contigs."""

    output_directory = Path(output_directory).resolve()
    temp_directory = Path(temp_directory).resolve()
    temp_directory.mkdir(parents=True, exist_ok=True)

    output_directory = output_directory / "metadata"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_dict = defaultdict(list)

    if n_processes is None:
        n_processes = mp.cpu_count()
    n_processes = min(len(partition_files), max(n_processes, 1))

    # process droplets
    pbar = tqdm(
        total=len(partition_files),
        desc="Processing droplets",
        disable=quiet,
    )
    droplet_kwargs = {
        "temp_directory": temp_directory,
        "clustering_threshold": clustering_threshold,
        "consensus_downsample": consensus_downsample,
        "min_cluster_reads": min_cluster_reads,
        "min_cluster_umis": min_cluster_umis,
        "quiet": quiet,
        "debug": debug,
    }
    with ProcessPoolExecutor(
        max_workers=n_processes,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(process_droplet, partition_file, **droplet_kwargs)
            for partition_file in partition_files
        ]

    # collect results
    for future in as_completed(futures):
        sample_name, metadata = future.result()
        output_dict[sample_name].extend(metadata)
        pbar.update(1)
    pbar.close()

    # write results
    output_files = []
    for sample_name, metadata in output_dict.items():
        metadata_file = output_directory / f"{sample_name}.csv"
        df = pl.DataFrame(metadata)
        df.write_csv(metadata_file)
        output_files.append(metadata_file)

    return output_files


def process_droplet(
    name: str,
    partition_df: pl.DataFrame,
    temp_directory: str | Path = "/tmp",
    clustering_threshold: float = 0.9,
    consensus_downsample: int = 100,
    min_cluster_reads: int = 3,
    min_cluster_umis: int = 2,
    min_cluster_fraction: float = 0.0,
    quiet: bool = False,
    debug: bool = False,
) -> tuple[str, list[dict]]:
    """
    Process a single droplet to generate contigs.

    Parameters
    ----------
    partition_file : str | Path
        The path to the partition file.

    output_directory : str | Path
        The path to the output directory.

    temp_directory : str | Path
        The path to the temporary directory.

    clustering_threshold : float
        The clustering threshold.

    consensus_downsample : int
        Number of reads to downsample for consensus sequence generation.

    min_cluster_reads : int
        Minimum number of reads in a cluster.

    min_cluster_umis : int
        Minimum number of UMIs in a cluster.

    min_cluster_fraction : float
        Minimum fraction of reads in a cluster.

    quiet : bool
        Whether to suppress output.

    debug : bool
        Whether to enable debug mode.

    Returns
    -------
    tuple[str, list[dict]]
        A tuple containing the sample name and metadata (including consensus sequence).

    """

    temp_directory = Path(temp_directory)
    barcode = name.split("_")[-1]
    metadata = []

    sequences = abutils.io.from_polars(
        partition_df, id_key="seq_id", sequence_key="sequence"
    )
    clusters = abutils.tl.cluster(
        sequences, threshold=clustering_threshold, temp_dir=temp_directory, debug=debug
    )
    for i, clust in enumerate(clusters):
        contig_name = f"{barcode}_contig-{i}"
        cluster_df = partition_df.filter(pl.col("seq_id").is_in(clust.seq_ids))
        n_umis = len(cluster_df["umi"].unique())
        cluster_fraction = clust.size / len(sequences)

        # consensus
        if cluster.size > 1:
            consensus = abutils.tl.make_consensus(
                clust.sequences,
                downsample_to=consensus_downsample,
                name=contig_name,
            )
        else:
            consensus = clust.sequences[0]

        # metadata
        meta = {
            "name": contig_name,
            "reads": clust.size,
            "umis": n_umis,
            "cluster_fraction": cluster_fraction,
            "consensus": consensus.sequence,
        }

        # filters
        if all(
            [
                clust.size >= min_cluster_reads,
                n_umis >= min_cluster_umis,
                cluster_fraction >= min_cluster_fraction,
            ]
        ):
            meta["pass_filters"] = True
        else:
            meta["pass_filters"] = False

        metadata.append(meta)

    return metadata
