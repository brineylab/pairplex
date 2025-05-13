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



from abutils.io import list_files, make_dir
from abstar.preprocess import merging
import re, os, subprocess, shutil, tempfile, sys, multiprocessing
from collections import defaultdict, Counter
from natsort import natsorted
from pathlib import Path
from typing import Set, Tuple
import polars as pl
import pandas as pd



def merge(files: list, output_folder: str, log_folder: str, verbose: bool, debug: bool) -> list:
    """Quick and dirty adapter to leverage merging wrapper from AbStar. Uses Fastp."""
    
    assert isinstance(files, list), "Incorrect list of files to merge. Aborting."
    assert files != [], "List of files to merge is empty. Aborting"

    merge_dir = os.path.join(output_folder, '01_merged')
    make_dir(merge_dir)

    merged_files = merging.merge_fastqs(files=files,
                                        output_directory=merge_dir,
                                        output_format='fastq',
                                        log_directory=log_folder,
                                        schema= 'element',
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
                                        debug=debug,
                                        show_progress=verbose,)
    return merged_files



def list_wells(merged_files: list, verbose: bool, debug: bool) -> dict:
    """Match input files to wells (e.g., A1, B9) and return mapping."""
    well_to_files = {}

    for f in merged_files:
        match = re.search(r'VDJ_([A-H][0-9]{1,2})\.fastq', f)
        if match:
            well = match.group(1)
            well_to_files[well] = f

    if any([verbose, debug]):
        sorted_wells = natsorted(well_to_files)
        print(f"Found {len(sorted_wells)} wells:")
        for w in sorted_wells:
            print(f"  {w} → {well_to_files[w]}")

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
                       barcodes_path: str = "./data/3M-5pgex-jan-2023.txt", 
                       threads: int = 10, 
                       output_folder: str = './pairplexed/',
                       temp_folder: str = '/tmp/',
                       enforce_bc_whitelist: bool = True,
                       check_rc: bool = True, 
                       verbose: bool = False, 
                       debug: bool = False) -> str:
    """
    Process FASTQ chunks in parallel and merge output files.
    Returns a dict with the well name and merged output file paths.
    """

    barcodes = load_barcode_whitelist(barcodes_path)
    chunk_out_paths = [Path(temp_folder) / Path(c).stem for c in chunks]

    with multiprocessing.Pool(threads) as pool:
        results = pool.starmap(
            process_chunk,
            [
                (chunk, barcodes, check_rc, str(outpath), enforce_bc_whitelist)
                for chunk, outpath in zip(chunks, chunk_out_paths)
            ]
        )

    # Collect outputs
    match_csvs = [csv for csv, _ in results]
    record_parquets = [pq for _, pq in results]

    # Merge CSVs (matches)
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

    # Merge Parquet files (records)
    df_records = pl.concat([pl.read_parquet(pq) for pq in record_parquets])
    df_records.write_parquet(final_parquet)

    return {well: {"matches": final_csv, "records": final_parquet}}


def process_chunk(chunk, barcodes, check_rc, outpath, enforce_bc_whitelist):
    """Process a chunk of fastq file to extract barcodes, UMIs and TSO sequences."""
    
    # The TSO sequence is defined as TTTCTTATATG{1,5} in the 5' end of the read (5'RACE protocol). Change sequence here if needed.
    tso_re = re.compile(r"TTTCTTATATG{1,5}")
    matches = Counter()
    records = defaultdict(list)

    with open(chunk, "r") as handle:
        while True:
            try:
                header = next(handle).strip()[1:]   # remove the '@' at the beginning
                seq = next(handle).strip()          # plain sequence
                next(handle)                        # skip the '+' line
                next(handle)                        # skip the quality line

                for s in (seq, reverse_complement(seq)) if check_rc else (seq,):
                    for i in range(0, len(s) - 40):
                        segment = s[i:i+46]
                        if len(segment) < 46:
                            continue
                        barcode = segment[:16]
                        umi = segment[16:26]
                        tail = segment[26:]

                        # We first check that we have a match for the TSO
                        m = tso_re.search(tail) # we're using search instead of match to allow for diffrent positions of the TSO

                        if not m:
                            continue
                        tso = m.group(0)

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

    matches_file, records_file = write_matches(matches, records, outpath)

    return matches_file, records_file


def write_matches(matches: Counter, records: dict, outpath: Path):
    """Write the matches and records to csv and parquet files respectively"""

    csv_file = outpath + "_matches.csv"
    parquet_file = outpath + "_records.parquet"

    df = pd.DataFrame(matches.items(), columns=["cell_barcode", "count"])
    df.to_csv(csv_file, index=False, header=False)

    flattened = [
                {"barcode": barcode, **record}
                    for barcode, recs in records.items()
                    for record in recs
                ]

    df = pl.DataFrame(flattened)
    df.write_parquet(parquet_file, )

    return csv_file, parquet_file


######################################################
##                Main function                     ##
######################################################

def main(sequencing_folder: str = "./", 
         output_folder: str = "./pairplexed/",
         barcodes_path: str = "./data/3M-5pgex-jan-2023.txt",
         chunk_size: int = 100_000,
         threads: int = 32,
         verbose: bool = False,
         debug: bool = False
        ):
    """PairPlex main routine"""

    ########### Pre-flight ##########
    make_dir(output_folder)
    temp_folder = os.path.join(output_folder, 'temp')
    log_folder = os.path.join(output_folder, '00_logs')
    make_dir(log_folder)
    make_dir(temp_folder)

    
    ########### Pre-processing data ###########
    files = list_files(sequencing_folder, recursive=True, extension="fastq.gz")
    files = [f for f in files if 'Unassigned' not in f]
    if any(("R2" in f) for f in files):
        # Paired-end sequencing, requires merging
        merged_files = merge(files=files, output_folder=output_folder, log_folder=log_folder, verbose=verbose, debug=debug)
    else:
        merged_files = files

        
    ########### Assigning barcodes in wells ###########
    wells = list_wells(merged_files, )
    barcoded_wells = []
    
    for well in natsorted(wells):
        fastq = merged_files[well]
        
        # First, we split into chunks to parallelize
        fastq_chunks = split_fastq(input_file=fastq, output_dir=temp_folder, lines_per_chunk=4*chunk_size)

        # Then, we assign barcodes/UMI and TSO for every chunk and concatenate results in a single file
        barcoded = assign_bc_parallel(well=well,
                                      chunks=fastq_chunks, 
                                      barcodes_path=barcodes_path, 
                                      threads=min(len(fastq_chunks), threads),
                                      output_folder=output_folder,
                                      temp_folder=temp_folder,
                                      check_rc=True, 
                                      verbose=verbose, 
                                      debug=debug)

        barcoded_wells.append(barcoded)

    
    ########### Processing individual cells/droplets ###########
    
    for well in barcoded_wells:
        df = pl.read_parquet(barcoded_wells[well]['records'])
        break
    
    
    return df