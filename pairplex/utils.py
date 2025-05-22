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
import os
from pathlib import Path
from typing import Set

import abutils
import polars as pl

from .version import __version__

BARCODE_DIR = Path(__file__).resolve().parent / "barcodes"
DEFAULT_WHITELIST = BARCODE_DIR / "737K-august-2016.txt"


# def setup_logger(output_folder: str, verbose: bool, debug: bool) -> logging.Logger:
#     """Set up a logger with proper formatting and both file and console handlers."""

#     logger = logging.getLogger("PairPlex")
#     if debug:
#         logger.setLevel(logging.DEBUG)
#     else:
#         logger.setLevel(logging.INFO)

#     # Clear existing handlers
#     if logger.hasHandlers():
#         logger.handlers.clear()

#     log_path = os.path.join(output_folder, "pairplex.log")
#     file_handler = logging.FileHandler(log_path)
#     file_handler.setLevel(logging.DEBUG)

#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)

#     formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
#     file_handler.setFormatter(formatter)
#     console_handler.setFormatter(formatter)

#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)

#     return logger


def get_whitelist_path(whitelist_name: str | Path) -> str | Path:
    """Get the path to a builtin barcode whitelist."""
    builtin_whitelists = {
        "v2": BARCODE_DIR / "737K-august-2016.txt",
        "v3": BARCODE_DIR / "3M-5pgex-jan-2023.txt.gz",
        "5v2": BARCODE_DIR / "737K-august-2016.txt",
        "5v3": BARCODE_DIR / "3M-5pgex-jan-2023.txt.gz",
        "5pv2": BARCODE_DIR / "737K-august-2016.txt",
        "5pv3": BARCODE_DIR / "3M-5pgex-jan-2023.txt.gz",
        "5primev2": BARCODE_DIR / "737K-august-2016.txt",
        "5primev3": BARCODE_DIR / "3M-5pgex-jan-2023.txt.gz",
        "5'v2": BARCODE_DIR / "737K-august-2016.txt",
        "5'v3": BARCODE_DIR / "3M-5pgex-jan-2023.txt.gz",
        "737k": BARCODE_DIR / "737K-august-2016.txt",
        "3m": BARCODE_DIR / "3M-5pgex-jan-2023.txt.gz",
        "nextgem": BARCODE_DIR / "737K-august-2016.txt",
        "gemx": BARCODE_DIR / "3M-5pgex-jan-2023.txt.gz",
    }
    # Check if it's a string that corresponds to a built-in whitelist
    if isinstance(whitelist_name, str) and whitelist_name.lower() in builtin_whitelists:
        return builtin_whitelists[whitelist_name.lower()]
    # Check if it's a path or a string path that exists
    elif Path(whitelist_name).exists():
        return Path(whitelist_name)
    else:
        raise ValueError(f"Invalid whitelist name or path: {whitelist_name}")


def load_barcode_whitelist(whitelist_path: str | Path) -> Set[str]:
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
    with open(whitelist_path) as f:
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
    return None  # Explicitly return None if no single correction is found


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
    else:
        whitelist_path = get_whitelist_path(whitelist_path)
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
            corrected = correct_barcode(barcode, whitelist)
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
) -> list[dict]:
    """
    Process a single droplet to generate contigs.

    Parameters
    ----------
    name : str
        Name of the droplet, typically containing the barcode.

    partition_df : pl.DataFrame
        DataFrame containing sequences from a single partition (droplet).

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
    list[dict]
        A list of dictionaries containing metadata for each cluster/contig.
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
        if clust.size > 1:
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


def print_splash(include_version: bool = True) -> None:
    """Print the splash screen."""

    print(PAIRPLEX_LOGO)
    if include_version:
        print(f"v{__version__}")
    print("Copyright (c) 2025 Benjamin Nemoz")
    print("Distributed under the terms of the MIT License.")
    print("")


PAIRPLEX_LOGO = """
 ______       _       ______  _              
(_____ \     (_)     (_____ \| |             
 _____) )____ _  ____ _____) ) | _____ _   _ 
|  ____(____ | |/ ___)  ____/| || ___ ( \ / )
| |    / ___ | | |   | |     | || ____|) X ( 
|_|    \_____|_|_|   |_|      \_)_____|_/ \_)
"""
