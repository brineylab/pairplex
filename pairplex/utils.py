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

# import subprocess as sp
# import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Set

import abutils
import pandas as pd
import polars as pl
from abstar.preprocess import merging
from abutils import Sequence

# from abutils.core.sequence import reverse_complement
from abutils.io import from_pandas, from_polars, make_dir, parse_fastx
from abutils.tools import cluster
from natsort import natsorted
from tqdm.auto import tqdm

BARCODE_DIR = Path(__file__).resolve().parent / "barcodes"
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
        output_directory=merge_dir,
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


# def reverse_complement(seq: str) -> str:
#     complement = str.maketrans("ACGTN", "TGCAN")
#     return seq.translate(complement)[::-1]


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


# def partition_by_barcode(
#     input_files: list[str | Path],
#     output_directory: str | Path,
#     temp_directory: str | Path = "/tmp",
#     whitelist_path: str | Path | None = None,
#     enforce_whitelist: bool = True,
#     tso_pattern: str = r"TTTCTTATATG{1,5}",
#     check_rc: bool = True,
#     chunksize: int = 1000,
#     quiet: bool = False,
#     debug: bool = False,
# ) -> list:
#     """
#     Partition fastq files into separate parquet files by barcode.

#     Parameters
#     ----------
#     input_files : list[str | Path]
#         The input files to partition. Should be either FASTQ or FASTA files,
#         optionally gzipped.

#     output_directory : str | Path
#         The output directory to save the partitioned files.

#     temp_directory : str | Path = "/tmp"
#         The temporary directory to save the partitioned files.

#     whitelist_path : str | Path | None = None
#         The path to the barcode whitelist file. If not provided, the default
#         whitelist will be used, which is the NextGEM 5'v2 barcode whitelist.

#     enforce_whitelist : bool = True
#         Whether to enforce the barcode whitelist. If True, only barcodes that
#         match the whitelist will be considered valid. If False, all barcodes
#         will be considered valid even if they do not match the whitelist.

#     tso_pattern : str = r"TTTCTTATATG{1,5}"
#         The pattern to search for the TSO sequence.

#     check_rc : bool = True
#         Whether to check the reverse complement of the sequences.

#     chunksize : int = 1000
#         Chunksize (in sequences) for parsing barcodes using multiprocessing.

#     quiet : bool = False
#         If True, the progress bar will be suppressed.

#     debug : bool = False
#         If True, all intermediate/temporary files will be kept to assist with
#         debugging.

#     Returns
#     -------
#     list
#         A list of paths to the partitioned files.

#     Raises
#     ------
#     FileNotFoundError
#         If the barcode whitelist file does not exist.

#     FileNotFoundError
#         If any of the input files do not exist.

#     """

#     pbar = tqdm(total=len(input_files), disable=quiet)
#     parsing_kwargs = {
#         "temp_directory": temp_directory,
#         "whitelist_path": whitelist_path,
#         "tso_pattern": tso_pattern,
#         "check_rc": check_rc,
#         "enforce_whitelist": enforce_whitelist,
#     }

#     partitioned_files = []
#     with ProcessPoolExecutor(
#         max_workers=mp.cpu_count(), mp_context=mp.get_context("spawn")
#     ) as executor:
#         for input_file in input_files:
#             input_file = Path(input_file)
#             name = input_file.stem
#             to_delete = []

#             # split input file into chunks
#             fastq_chunks = abutils.io.split_fastx(
#                 fastx_file=str(input_file),
#                 output_directory=str(temp_directory),
#                 chunksize=chunksize,
#             )
#             to_delete.extend(fastq_chunks)

#             # parse barcodes
#             pbar.set_postfix_str("parsing barcodes", refresh=True)
#             futures = [
#                 executor.submit(parse_barcodes, chunk, **parsing_kwargs)
#                 for chunk in fastq_chunks
#             ]

#             parquet_chunks = []
#             for future in as_completed(futures):
#                 res = future.result()
#                 if res is not None:
#                     parquet_chunks.append(res)
#             to_delete.extend(parquet_chunks)

#             # concatenate the chunked parquet files
#             concat_parquet = abutils.io.concatenate_parquet(
#                 parquet_chunks, temp_directory / f"{name}.parquet"
#             )
#             to_delete.append(concat_parquet)

#             # partition into separate parquet files by barcode
#             pbar.set_postfix_str("partitioning barcodes", refresh=True)
#             df = pl.read_parquet(concat_parquet)
#             partitions = df.partition_by("barcode", as_dict=True)

#             for bc, bc_df in partitions.items():
#                 if isinstance(bc, tuple):
#                     bc = bc[0]
#                 partitioned_file = output_directory / f"{name}_{bc}.parquet"
#                 bc_df.write_parquet(partitioned_file)
#                 partitioned_files.append(partitioned_file)

#             # cleanup
#             if debug:
#                 pbar.set_postfix_str("cleaning up", refresh=True)
#                 for f in to_delete:
#                     if f is not None:
#                         if os.path.exists(f):
#                             os.remove(f)
#                 if os.path.isdir(temp_directory) and not os.listdir(temp_directory):
#                     os.rmdir(temp_directory)

#             pbar.update(1)

#     return partitioned_files


# def split_fastq(
#     prefix: str, input_file: str, output_dir: Path, lines_per_chunk: int = 400_000
# ) -> list[str]:
#     if prefix is None:
#         lead = output_dir / "chunk_"
#     else:
#         lead = output_dir / f"{prefix}_chunk_"
#     subprocess.run(
#         [
#             "split",
#             "-l",
#             str(lines_per_chunk),
#             "--numeric-suffixes=1",
#             "--additional-suffix=.fastq",
#             input_file,
#             str(lead),
#         ],
#         check=True,
#     )
#     return sorted(str(f) for f in output_dir.glob("*chunk_*.fastq"))


# def assign_bc_unparalleled(
#     well: str = None,
#     chunks: list = [],
#     barcodes_path: str = None,
#     output_folder: str = "./pairplexed/",
#     temp_folder: str = "/tmp/",
#     enforce_bc_whitelist: bool = True,
#     check_rc: bool = True,
#     verbose: bool = False,
#     debug: bool = False,
# ) -> str:
#     global logger

#     if barcodes_path is None:
#         logger.error("No barcode file provided. Aborting.")
#         raise AssertionError("No barcode file provided. Aborting.")

#     if logger:
#         logger.info(
#             f"[{well}] Starting single-thread cell barcode assignment with {len(chunks)} chunks using 1 threads."
#         )

#     chunk_out_paths = [Path(temp_folder) / Path(c).stem for c in chunks]

#     results = []
#     for chunk, chunk_out_path in zip(chunks, chunk_out_paths):
#         results.append(
#             process_chunk(
#                 chunk,
#                 barcodes_path,
#                 check_rc,
#                 str(chunk_out_path),
#                 enforce_bc_whitelist,
#             )
#         )

#     match_csvs = [csv for csv, _ in results]
#     record_parquets = [pq for _, pq in results]

#     if logger:
#         logger.debug(
#             f"[{well}] Merging {len(match_csvs)} match files and {len(record_parquets)} record files..."
#         )

#     match_df = pd.concat(
#         [
#             pd.read_csv(csv, header=None, names=["cell_barcode", "count"])
#             for csv in match_csvs
#         ]
#     )
#     merged_matches = match_df.groupby("cell_barcode", as_index=False)["count"].sum()

#     out_dir = os.path.join(output_folder, "02_barcoded")
#     make_dir(out_dir)

#     final_csv = os.path.join(out_dir, f"barcoded_{well}_matches.csv")
#     final_parquet = os.path.join(out_dir, f"barcoded_{well}_records.parquet")

#     merged_matches.to_csv(final_csv, index=False)

#     dfs = []
#     for pq in record_parquets:
#         try:
#             df = pl.read_parquet(pq)
#             if df.shape[0] > 0:
#                 dfs.append(df)
#             else:
#                 if logger:
#                     logger.warning(f"[{well}] Skipped empty Parquet file: {pq}")
#         except Exception as e:
#             if logger:
#                 logger.error(f"[{well}] Failed to read {pq}: {e}")

#     logger.debug(f"[{well}] Merging {len(dfs)} record files...")

#     if len(dfs) == 0:
#         if logger:
#             logger.warning(f"[{well}] No valid records found. Returning empty outputs.")
#         # Optionally write empty placeholders
#         empty_df = pl.DataFrame(
#             schema={
#                 "barcode": pl.Utf8,
#                 "UMI": pl.Utf8,
#                 "TSO": pl.Utf8,
#                 "seq_id": pl.Utf8,
#                 "sequence": pl.Utf8,
#             }
#         )
#         empty_df.write_parquet(final_parquet)
#         pd.DataFrame(columns=["cell_barcode", "count"]).to_csv(final_csv, index=False)

#         return {well: {"matches": final_csv, "records": final_parquet}}

#     # If we got here, we have non-empty DataFrames to merge
#     df_records = pl.concat(dfs)
#     df_records.write_parquet(final_parquet)

#     if logger:
#         logger.debug(f"[{well}] Wrote merged match file → {final_csv}")
#         logger.debug(f"[{well}] Wrote merged record file → {final_parquet}")
#         logger.debug(
#             f"[{well}] Total barcodes: {len(merged_matches)} | Total records: {df_records.shape[0]}"
#         )

#     time.sleep(1)  # Ensure all is done before removing files

#     # Clean up temporary chunk files
#     for chunk in chunks:
#         try:
#             os.remove(chunk)
#         except FileNotFoundError:
#             pass
#     for csv in match_csvs:
#         try:
#             os.remove(csv)
#         except FileNotFoundError:
#             pass
#     for pq in record_parquets:
#         try:
#             os.remove(pq)
#         except FileNotFoundError:
#             pass

#     time.sleep(1)  # Ensure all is done before returning

#     return {"matches": final_csv, "records": final_parquet}


# def assign_bc_paralleled(
#     well: str = None,
#     chunks: list = [],
#     barcodes_path: str = None,
#     threads: int = 10,
#     output_folder: str = "./pairplexed/",
#     temp_folder: str = "/tmp/",
#     enforce_bc_whitelist: bool = True,
#     check_rc: bool = True,
#     verbose: bool = False,
#     debug: bool = False,
# ) -> str:
#     global logger

#     if barcodes_path is None:
#         logger.error("No barcode file provided. Aborting.")
#         raise AssertionError("No barcode file provided. Aborting.")

#     logger.info(
#         f"[{well}] Starting multi-threaded cell barcode assignment with {len(chunks)} chunks using {threads} threads."
#     )

#     chunk_out_paths = [
#         Path(temp_folder) / f"{well}_chunk_{i:02d}"
#         for i, _ in enumerate(chunks, start=1)
#     ]

#     results = []
#     # context = multiprocessing.get_context("spawn") # /!\ Make sure to use "spawn" when using polars
#     with mp.Pool(
#         threads,
#     ) as pool:
#         async_results = []
#         for chunk, chunk_out_path in zip(chunks, chunk_out_paths):
#             async_results.append(
#                 pool.apply_async(
#                     process_chunk,
#                     (
#                         chunk,
#                         barcodes_path,
#                         check_rc,
#                         str(chunk_out_path),
#                         enforce_bc_whitelist,
#                     ),
#                 )
#             )
#         results = [async_result.get() for async_result in async_results]

#     logger.debug(f"[{well}] Finished processing all chunks.")

#     match_csvs = [csv for csv, _ in results]
#     record_parquets = [pq for _, pq in results]

#     logger.debug(
#         f"[{well}] Merging {len(match_csvs)} match files and {len(record_parquets)} record files..."
#     )

#     match_df = pd.concat(
#         [
#             pd.read_csv(csv, header=None, names=["cell_barcode", "count"])
#             for csv in match_csvs
#         ]
#     )
#     merged_matches = match_df.groupby("cell_barcode", as_index=False)["count"].sum()

#     out_dir = os.path.join(output_folder, "02_barcoded")
#     make_dir(out_dir)

#     final_csv = os.path.join(out_dir, f"barcoded_{well}_matches.csv")
#     final_parquet = os.path.join(out_dir, f"barcoded_{well}_records.parquet")

#     merged_matches.to_csv(final_csv, index=False)

#     dfs = []
#     for pq in record_parquets:
#         try:
#             df = pl.read_parquet(pq)
#             if df.shape[0] > 0:
#                 dfs.append(df)
#             else:
#                 if logger:
#                     logger.warning(f"[{well}] Skipped empty Parquet file: {pq}")
#         except Exception as e:
#             if logger:
#                 logger.error(f"[{well}] Failed to read {pq}: {e}")

#     logger.debug(f"[{well}] Merging {len(dfs)} record files...")

#     if len(dfs) == 0:
#         if logger:
#             logger.warning(f"[{well}] No valid records found. Returning empty outputs.")
#         # Optionally write empty placeholders
#         empty_df = pl.DataFrame(
#             schema={
#                 "barcode": pl.Utf8,
#                 "UMI": pl.Utf8,
#                 "TSO": pl.Utf8,
#                 "seq_id": pl.Utf8,
#                 "sequence": pl.Utf8,
#             }
#         )
#         empty_df.write_parquet(final_parquet)
#         pd.DataFrame(columns=["cell_barcode", "count"]).to_csv(final_csv, index=False)

#         return {well: {"matches": final_csv, "records": final_parquet}}

#     # If we got here, we have non-empty DataFrames to merge
#     df_records = pl.concat(dfs)
#     df_records.write_parquet(final_parquet)

#     if logger:
#         logger.debug(f"[{well}] Wrote merged match file → {final_csv}")
#         logger.debug(f"[{well}] Wrote merged record file → {final_parquet}")
#         logger.debug(
#             f"[{well}] Total barcodes: {len(merged_matches)} | Total records: {df_records.shape[0]}"
#         )

#     time.sleep(1)  # Ensure all is done before removing files

#     # Clean up temporary chunk files
#     for chunk in chunks:
#         try:
#             os.remove(chunk)
#         except FileNotFoundError:
#             pass
#     for csv in match_csvs:
#         try:
#             os.remove(csv)
#         except FileNotFoundError:
#             pass
#     for pq in record_parquets:
#         try:
#             os.remove(pq)
#         except FileNotFoundError:
#             pass

#     time.sleep(1)  # Ensure all is done before returning

#     return {"matches": final_csv, "records": final_parquet}


# def process_chunk(
#     chunk: list,
#     chunk_id: str | int,
#     output_directory: str | Path,
#     barcodes_path: str | Path | None = None,
#     tso_pattern: str = r"TTTCTTATATG{1,5}",
#     check_rc: bool = True,
#     enforce_bc_whitelist: bool = True,
# ) -> str | None:
#     """
#     Process a chunk of fastq file to extract barcodes, UMIs and TSO sequences.

#     Parameters
#     ----------
#     chunk : list
#         A list of sequences to process.

#     chunk_id : str | int
#         The ID of the chunk.

#     output_directory : str | Path
#         The directory to save the output.

#     barcodes_path : str | Path | None = None
#         The path to the barcode file.

#     tso_pattern : str = r"TTTCTTATATG{1,5}"
#         The pattern to search for the TSO sequence.

#     check_rc : bool = True
#         Whether to check the reverse complement of the sequences.

#     enforce_bc_whitelist : bool = True
#         Whether to enforce the barcode whitelist.

#     Returns
#     -------
#     str | None
#         The path to the output file (or None if no records are found).
#     """

#     if enforce_bc_whitelist:
#         whitelist = load_barcode_whitelist(barcodes_path)

#     tso_pattern = re.compile(tso_pattern)

#     records = []
#     for seq in parse_fastx(chunk):
#         seqs = [seq.sequence]
#         if check_rc:
#             seqs.append(reverse_complement(seq.sequence))
#         for s in seqs:
#             # find the TSO
#             m = tso_pattern.search(s)
#             if not m:
#                 continue

#             # parse barcode and UMI
#             tso = m.group(0)
#             sequence = s[m.start() :]
#             leader = s[: m.start()]
#             barcode = leader[:-10]
#             umi = leader[-10:]

#             # check whitelist (if necessary)
#             if enforce_bc_whitelist:
#                 if barcode not in whitelist:
#                     continue
#             records.append(
#                 {
#                     "umi": umi,
#                     "barcode": barcode,
#                     "tso": tso,
#                     "seq_id": seq.id,
#                     "sequence": sequence,  # we want the sequence with TSO, barcode and UMI stripped off
#                 }
#             )
#             break

#     if records:
#         output_file = Path(output_directory) / f"bcdf_{chunk_id}.parquet"
#         df = pl.DataFrame(records)
#         df.write_parquet(output_file)
#         return str(output_file)


# def write_matches(matches: Counter, records: dict, outpath: Path):
#     """Write the matches and records to CSV and Parquet files respectively."""

#     global logger

#     csv_file = Path(str(outpath) + "_matches.csv")
#     parquet_file = Path(str(outpath) + "_records.parquet")

#     if logger:
#         logger.debug(f"Writing matches to {csv_file}")

#     df = pd.DataFrame(matches.items(), columns=["cell_barcode", "count"])
#     df.to_csv(csv_file, index=False, header=False)

#     if records == {}:
#         if logger:
#             logger.warning(
#                 f"No records found for {csv_file}. Returning empty Parquet file."
#             )
#         ## /!\ use Polars only with 'spawn' context manager
#         # empty_df = pl.DataFrame(schema={"barcode": pl.Utf8, "UMI": pl.Utf8, "TSO": pl.Utf8, "seq_id": pl.Utf8, "sequence": pl.Utf8})
#         # empty_df.write_parquet(parquet_file)
#         ## /!\ use Pandas with 'fork' context manager (default)
#         empty_df = pd.DataFrame(
#             {
#                 "barcode": pd.Series(dtype="str"),
#                 "UMI": pd.Series(dtype="str"),
#                 "TSO": pd.Series(dtype="str"),
#                 "seq_id": pd.Series(dtype="str"),
#                 "sequence": pd.Series(dtype="str"),
#             }
#         )
#         empty_df.to_parquet(parquet_file, index=False)
#         return csv_file, parquet_file

#     if logger:
#         logger.debug(f"Writing records to {parquet_file}")

#     flattened = [
#         {"barcode": barcode, **record}
#         for barcode, recs in records.items()
#         for record in recs
#     ]

#     ## /!\ use Polars only with 'spawn' context manager
#     # df = pl.DataFrame(flattened)
#     # df.write_parquet(parquet_file)
#     ## /!\ use Pandas with 'fork' context manager (default)
#     df = pd.DataFrame(flattened)
#     df.to_parquet(parquet_file, index=False)

#     time.sleep(0.1)  # Ensure the file is written before returning

#     if logger:
#         logger.debug(f"Successfully finished writing matches and records")

#     return csv_file, parquet_file


# def get_barcode_file(name: str) -> str:
#     """Get the barcode file path from the name."""
#     if name == "5prime":
#         return "./barcodes/3M-5pgex-jan-2023.txt"
#     elif name == "3prime":
#         return "./barcodes/3M-3pgex-may-2023.txt"
#     elif name == "v2":
#         return "./barcodes/737K-august-2016.txt"
#     else:
#         raise ValueError(f"Unknown barcode file: {name}")


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


# def process_cell(
#     well: str,
#     cell: str,
#     sequence_bin: pl.DataFrame | pd.DataFrame,
#     cluster_folder: str,
#     min_cluster_size: int,
#     clustering_threshold: float,
#     min_umi_count: int,
#     consentroid: bool,
#     debug: bool,
# ) -> dict:
#     """Process a single cell to cluster sequences and generate contigs."""

#     global logger

#     if debug:
#         logger.debug(
#             f"[{well}] Processing cell {cell} with {len(sequence_bin)} sequences (clustering threshold: {clustering_threshold})"
#         )

#     # Cluster the sequences
#     if isinstance(sequence_bin, pl.DataFrame):
#         sequences = from_polars(sequence_bin, id_key="seq_id", sequence_key="sequence")
#     elif isinstance(sequence_bin, pd.DataFrame):
#         sequences = from_pandas(sequence_bin, id_key="seq_id", sequence_key="sequence")
#     chain_bins = cluster.cluster(
#         sequences, threshold=clustering_threshold, temp_dir=cluster_folder, debug=False
#     )

#     if debug:
#         logger.debug(f"[{well}] Cell {cell} → {len(chain_bins)} clusters")

#     # Generate contigs
#     contigs = []
#     metadata = []

#     for i, chain_bin in enumerate(chain_bins):
#         if chain_bin.size < min_cluster_size:
#             if debug:
#                 logger.debug(
#                     f"[{well}] Skipping cluster {i} of cell {cell} (size={chain_bin.size})"
#                 )
#             continue

#         seq_ids = chain_bin.seq_ids
#         if isinstance(sequence_bin, pl.DataFrame):
#             umi_count = (
#                 sequence_bin.filter(
#                     (pl.col("barcode") == cell) & (pl.col("seq_id").is_in(seq_ids))
#                 )
#                 .select("UMI")
#                 .unique()
#                 .height
#             )
#         elif isinstance(sequence_bin, pd.DataFrame):
#             umi_count = (
#                 sequence_bin.query(
#                     f"barcode == '{cell}' and seq_id in {tuple(seq_ids)}"
#                 )
#                 .drop_duplicates(subset=["UMI"])
#                 .shape[0]
#             )

#         if umi_count < min_umi_count:
#             if debug:
#                 logger.debug(
#                     f"[{well}] Skipping cluster {i} of cell {cell} due to low UMI count ({umi_count})"
#                 )
#             continue

#         if consentroid == "consensus":
#             sequence = chain_bin.make_consensus().sequence
#             if debug:
#                 logger.debug(
#                     f"[{well}] Cell {cell}, cluster {i} → consensus built (UMIs={umi_count}, reads={chain_bin.size})"
#                 )
#         elif consentroid == "centroid":
#             sequence = chain_bin.centroid.sequence
#             if debug:
#                 logger.debug(
#                     f"[{well}] Cell {cell}, cluster {i} → centroid selected (UMIs={umi_count}, reads={chain_bin.size})"
#                 )

#         name = f"{cell}_contig-{i}"
#         chain = Sequence(sequence, id=name)
#         contigs.append(chain)
#         metadata.append(
#             {"sequence_id": name, "UMI_count": umi_count, "reads": chain_bin.size}
#         )

#     return {"contigs": contigs, "metadata": metadata}


# def process_cell_pair(
#     cell: str, cell_df: pd.DataFrame, metadata_path: str, only_pairs: bool = False
# ) -> dict | None:
#     # Convert back to Polars
#     cell_df = pl.DataFrame(cell_df)

#     if len(cell_df) == 1:
#         if only_pairs:
#             return None
#         else:
#             return None  # TODO: Handle single chain logic

#     elif len(cell_df) == 2:
#         chain1 = cell_df.filter(pl.col("locus") == "IGH")
#         chain2 = cell_df.filter(pl.col("locus") != "IGH")

#         if len(chain1) == 1 and len(chain2) == 1:
#             # Pair the chains
#             heavy = from_polars(chain1)[0]
#             light = from_polars(chain2)[0]
#             for k in [k for k in light.annotations.keys() if k.startswith("d")]:
#                 light.annotations.pop(k)

#             # Gather the corresponding metadata
#             heavy_umi, heavy_reads = (
#                 pl.read_csv(well_to_files[well])
#                 .filter((pl.col("sequence_id") == heavy["sequence_id"]))[
#                     ["UMI_count", "reads"]
#                 ]
#                 .row(0)
#             )
#             light_umi, light_reads = (
#                 pl.read_csv(well_to_files[well])
#                 .filter((pl.col("sequence_id") == light["sequence_id"]))[
#                     ["UMI_count", "reads"]
#                 ]
#                 .row(0)
#             )

#             # Prepare the dictionary for the pair
#             pair_dict = {}
#             pair_dict["index"] = cell
#             for k, v in heavy.annotations.items():
#                 pair_dict[k + ":1"] = v
#             pair_dict["umi:1"] = heavy_umi
#             pair_dict["reads:1"] = heavy_reads
#             for k, v in light.annotations.items():
#                 pair_dict[k + ":2"] = v
#             pair_dict["umi:2"] = light_umi
#             pair_dict["reads:2"] = light_reads

#             return pair_dict

#         else:
#             # Handle the case where we have two chains but they are not paired
#             if only_pairs:
#                 return None
#             else:
#                 return None
#     elif len(cell_df) > 2:
#         # Handle the case where we have more than two chains
#         if only_pairs:
#             return None
#         else:
#             return None

#     return None
