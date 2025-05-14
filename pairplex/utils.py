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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with PairPlex.  If not, see <http://www.gnu.org/licenses/>.


from abutils.io import make_dir, from_polars, parse_fastx
from abutils.tools import cluster
from abutils import Sequence
from abstar.preprocess import merging
import re, os, subprocess, logging, time
from collections import defaultdict, Counter
from natsort import natsorted
from pathlib import Path
from typing import Set
import polars as pl
import pandas as pd



def setup_logger(output_folder: str, debug: bool) -> logging.Logger:
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

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def merge(files: list, output_folder: str, log_folder: str, schema: str, verbose: bool, debug: bool) -> list:
    """Quick and dirty adapter to leverage merging wrapper from AbStar. Uses Fastp."""
    
    assert isinstance(files, list), "Incorrect list of files to merge. Aborting."
    assert files != [], "List of files to merge is empty. Aborting"

    merge_dir = os.path.join(output_folder, '01_merged')
    log = os.path.join(log_folder, 'merging')
    make_dir(merge_dir)

    merged_files = merging.merge_fastqs(files=files,
                                        output_directory=merge_dir,
                                        output_format='fastq',
                                        log_directory=log,
                                        schema=schema,
                                        algo= 'fastp',
                                        binary_path= None,
                                        merge_args= None,
                                        minimum_overlap= 30,
                                        allowed_mismatches= 5,
                                        allowed_mismatch_percent= 20.0,
                                        trim_adapters= True,
                                        adapter_file= None,
                                        quality_trim= True,
                                        window_size= 4,
                                        quality_cutoff= 20,
                                        interleaved= False,
                                        compress_output= False,
                                        debug=False,
                                        show_progress=verbose,)
    return merged_files



def list_wells(merged_files: list, verbose: bool, debug: bool) -> dict:
    """Match input files to wells (e.g., A1, B9) and return mapping."""
    global logger

    well_to_files = {}

    for f in merged_files:
        match = re.search(r'VDJ_([A-H][0-9]{1,2})\.fastq', f)
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



def reverse_complement(seq: str) -> str:
    complement = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(complement)[::-1]



def load_barcode_whitelist(path: str) -> Set[str]:
    return set(line.strip() for line in open(path))



def split_fastq(input_file: str, output_dir: Path, lines_per_chunk: int = 400_000) -> list[str]:
    prefix = output_dir / "chunk_"
    subprocess.run([
        "split",
        "-l", str(lines_per_chunk),
        "--numeric-suffixes=1",
        "--additional-suffix=.fastq",
        input_file,
        str(prefix)
    ], check=True)
    return sorted(str(f) for f in output_dir.glob("chunk_*.fastq"))



def assign_bc_parallel(well: str = None,
                       chunks: list = [], 
                       barcodes_path: str = None, 
                       threads: int = 10, 
                       output_folder: str = './pairplexed/',
                       temp_folder: str = '/tmp/',
                       enforce_bc_whitelist: bool = True,
                       check_rc: bool = True, 
                       verbose: bool = False, 
                       debug: bool = False) -> str:
    global logger

    assert barcodes_path is not None, logger.error("No barcode file provided. Aborting.")

    if logger: logger.info(f"[{well}] Starting parallel assignment with {len(chunks)} chunks using {threads} threads.")
    
    chunk_out_paths = [Path(temp_folder) / Path(c).stem for c in chunks]

    # to be parallelized
    results = []
    for chunk, chun_ou_path in zip(chunks, chunk_out_paths):
        results.append(
            process_chunk(chunk, barcodes_path, check_rc, str(chun_ou_path), enforce_bc_whitelist)
        )
    
    match_csvs = [csv for csv, _ in results]
    record_parquets = [pq for _, pq in results]

    if logger: logger.debug(f"[{well}] Merging {len(match_csvs)} match files and {len(record_parquets)} record files...")

    match_df = pd.concat([
        pd.read_csv(csv, header=None, names=["cell_barcode", "count"])
        for csv in match_csvs
    ])
    merged_matches = match_df.groupby("cell_barcode", as_index=False)["count"].sum()

    out_dir = os.path.join(output_folder, '02_barcoded')
    make_dir(out_dir)

    final_csv = os.path.join(out_dir, f"barcoded_{well}_matches.csv")
    final_parquet = os.path.join(out_dir, f"barcoded_{well}_records.parquet")

    merged_matches.to_csv(final_csv, index=False)

    dfs = []
    for pq in record_parquets:
        try:
            df = pl.read_parquet(pq)
            if df.shape[0] > 0:
                dfs.append(df)
            else:
                if logger:
                    logger.warning(f"[{well}] Skipped empty Parquet file: {pq}")
        except Exception as e:
            if logger:
                logger.error(f"[{well}] Failed to read {pq}: {e}")

    logger.debug(f"[{well}] Merging {len(dfs)} record files...")

    if len(dfs) == 0:
        if logger:
            logger.warning(f"[{well}] No valid records found. Returning empty outputs.")
        # Optionally write empty placeholders
        empty_df = pl.DataFrame(schema={"barcode": pl.Utf8, "UMI": pl.Utf8, "TSO": pl.Utf8, "seq_id": pl.Utf8, "sequence": pl.Utf8})
        empty_df.write_parquet(final_parquet)
        pd.DataFrame(columns=["cell_barcode", "count"]).to_csv(final_csv, index=False)

        return {well: {"matches": final_csv, "records": final_parquet}}

    # If we got here, we have non-empty DataFrames to merge
    df_records = pl.concat(dfs)
    df_records.write_parquet(final_parquet)

    if logger:
        logger.debug(f"[{well}] Wrote merged match file → {final_csv}")
        logger.debug(f"[{well}] Wrote merged record file → {final_parquet}")
        logger.debug(f"[{well}] Total barcodes: {len(merged_matches)} | Total records: {df_records.shape[0]}")

    time.sleep(1)  # Ensure all is done before removing files

    # Clean up temporary chunk files
    for chunk in chunks:
        os.remove(chunk)
    for csv in match_csvs:
        os.remove(csv)
    for pq in record_parquets:
        os.remove(pq)
 
    time.sleep(1)  # Ensure all is done before returning

    return {"matches": final_csv, "records": final_parquet}



def process_chunk(chunk, barcodes_path, check_rc, outpath, enforce_bc_whitelist):
    """Process a chunk of fastq file to extract barcodes, UMIs and TSO sequences."""
    
    barcodes = load_barcode_whitelist(barcodes_path)

    # The TSO sequence is defined as TTTCTTATATG{1,5} in the 5' end of the read (5'RACE protocol). Change sequence here if needed.
    tso_re = re.compile(r"TTTCTTATATG{1,5}")
    matches = Counter()
    records = defaultdict(list)

    while True:
        try:
            for sequence in parse_fastx(chunk, ):
                for s in (sequence, reverse_complement(sequence)) if check_rc else (seq,):
                    # We first check that we have a match for the TSO
                    m = tso_re.search(s) # we're using search instead of match to allow for diffrent positions of the TSO

                    if not m:
                        continue
                    tso = m.group(0)
                    leader = s[:m.start()]
                    barcode = leader[:-10]
                    umi = leader[-10:]

                    # Then, if enabled, we verify that the barcode is on the whitelist
                    if enforce_bc_whitelist:
                        if barcode not in barcodes:
                            continue
                    
                    # If all concurs, we add the record to the matches and increment counters
                    matches[barcode] += 1
                    if barcode not in records:
                        records[barcode] = []
                    records[barcode].append({"UMI":umi,"TSO":tso,"seq_id":header,"sequence":seq})

                    break  # stop once match is found for this orientation (don't do more positions)
                    break  # stop once match is found for first orientation (don't do reverse complement)

        except StopIteration:
            break

    matches_file, records_file = write_matches(matches, records, Path(outpath))

    return matches_file, records_file


# def process_chunk(chunk, barcodes_path, check_rc, outpath, enforce_bc_whitelist):
#     """Process a chunk of fastq file to extract barcodes, UMIs and TSO sequences."""
    
#     barcodes = load_barcode_whitelist(barcodes_path)

#     # The TSO sequence is defined as TTTCTTATATG{1,5} in the 5' end of the read (5'RACE protocol). Change sequence here if needed.
#     tso_re = re.compile(r"TTTCTTATATG{1,5}")
#     matches = Counter()
#     records = defaultdict(list)

#     with open(chunk, "r") as handle:
#         while True:
#             try:
#                 header = next(handle).strip()[1:].split(' ')[0]     # remove the '@' at the beginning and the trailing info 
#                 seq = next(handle).strip()                          # plain sequence
#                 next(handle)                                        # skip the '+' line
#                 next(handle)                                        # skip the quality line

#                 for s in (seq, reverse_complement(seq)) if check_rc else (seq,):
                    
#                     # We first check that we have a match for the TSO
#                     m = tso_re.search(s) # we're using search instead of match to allow for diffrent positions of the TSO

#                     if not m:
#                         continue
#                     tso = m.group(0)
#                     leader = s[:m.start()]
#                     barcode = leader[:-10]
#                     umi = leader[-10:]

#                     # Then, if enabled, we verify that the barcode is on the whitelist
#                     if enforce_bc_whitelist:
#                         if barcode not in barcodes:
#                             continue
                    
#                     # If all concurs, we add the record to the matches and increment counters
#                     matches[barcode] += 1
#                     if barcode not in records:
#                         records[barcode] = []
#                     records[barcode].append({"UMI":umi,"TSO":tso,"seq_id":header,"sequence":seq})

#                     break  # stop once match is found for this orientation (don't do more positions)
#                     break  # stop once match is found for first orientation (don't do reverse complement)

#             except StopIteration:
#                 break

#     matches_file, records_file = write_matches(matches, records, Path(outpath))

#     return matches_file, records_file


def write_matches(matches: Counter, records: dict, outpath: Path):
    """Write the matches and records to CSV and Parquet files respectively."""

    global logger

    csv_file = Path(str(outpath) + "_matches.csv")
    parquet_file = Path(str(outpath) + "_records.parquet")

    if logger:
        logger.debug(f"Writing matches to {csv_file}")

    df = pd.DataFrame(matches.items(), columns=["cell_barcode", "count"])
    df.to_csv(csv_file, index=False, header=False)

    if logger:
        logger.debug(f"Writing records to {parquet_file}")

    flattened = [
        {"barcode": barcode, **record}
        for barcode, recs in records.items()
        for record in recs
    ]

    df = pl.DataFrame(flattened)
    df.write_parquet(parquet_file)

    time.sleep(0.1)  # Ensure the file is written before returning
    if logger:
        logger.debug(f"Successfully finished writing matches and records")

    return csv_file, parquet_file


def get_barcode_file(name: str) -> str:
    """Get the barcode file path from the name."""
    if name == "5prime":
        return "./barcodes/3M-5pgex-jan-2023.txt"
    elif name == "3prime":
        return "./barcodes/3M-3pgex-jan-2023.txt"
    elif name == "v2":
        return "./barcodes/737K-august-2016.txt"
    else:
        raise ValueError(f"Unknown barcode file: {name}")
    


def process_cell(well: str, 
                 cell: str, 
                 sequence_bin: dict, 
                 cluster_folder: str, 
                 min_cluster_size: int, 
                 clustering_threshold: float,
                 min_umi_count: int, 
                 consentroid: bool, 
                 debug: bool) -> dict:
    """Process a single cell to cluster sequences and generate contigs."""
    
    global logger

    if debug:
        logger.debug(f"[{well}] Processing cell {cell} with {len(sequence_bin)} sequences (clustering threshold: {clustering_threshold})")

    # Cluster the sequences
    sequences = from_polars(sequence_bin, id_key="seq_id", sequence_key="sequence")
    chain_bins = cluster.cluster(sequences, threshold=clustering_threshold, temp_dir=cluster_folder, debug=False)

    if debug:
        logger.debug(f"[{well}] Cell {cell} → {len(chain_bins)} clusters")

    # Generate contigs
    contigs = []
    metadata = []

    for i, chain_bin in enumerate(chain_bins):
        if chain_bin.size < min_cluster_size:
            if debug:
                logger.debug(f"[{well}] Skipping cluster {i} of cell {cell} (size={chain_bin.size})")
            continue

        seq_ids = chain_bin.seq_ids
        umi_count = (
            sequence_bin.filter(
                (pl.col('barcode') == cell) & 
                (pl.col('seq_id').is_in(seq_ids))
            )
            .select('UMI')
            .unique()
            .height
        )

        if umi_count < min_umi_count:
            if debug:
                logger.debug(f"[{well}] Skipping cluster {i} of cell {cell} due to low UMI count ({umi_count})")
            continue

        if consentroid == "consensus":
            sequence = chain_bin.make_consensus().sequence
            if debug:
                logger.debug(f"[{well}] Cell {cell}, cluster {i} → consensus built (UMIs={umi_count}, reads={chain_bin.size})")
        elif consentroid == "centroid":
            sequence = chain_bin.centroid.sequence
            if debug:
                logger.debug(f"[{well}] Cell {cell}, cluster {i} → centroid selected (UMIs={umi_count}, reads={chain_bin.size})")

        name = f"{cell}_contig-{i}"
        chain = Sequence(sequence, id=name)
        contigs.append(chain)
        metadata.append({
            "sequence_id": name,
            "UMI_count": umi_count,
            "reads": chain_bin.size
        })

    return {
        "contigs": contigs,
        "metadata": metadata
    }
                