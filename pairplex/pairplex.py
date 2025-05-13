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

"""
PairPlex: A tool for the analysis of pairwise cross-linking mass spectrometry data.
"""

from abutils.io import list_files, make_dir
from abstar.preprocess import merging
import re, os, subprocess, shutil, tempfile, sys, multiprocessing
from collections import defaultdict, Counter
from natsort import natsorted
from pathlib import Path
from typing import Set, Tuple
import polars as pl



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
                       check_rc: bool = True, 
                       verbose: bool = False, 
                       debug: bool = False) -> str:
    """ """

    barcodes = load_barcode_whitelist(barcodes_path)
    chunk_out_paths = [Path(temp_folder) / (Path(c).stem + ".parquet") for c in chunks]
    
    with multiprocessing.Pool(threads) as pool:
        pool.starmap(
            process_chunk,
            [(chunk, barcodes, check_rc, outpath)
             for chunk, outpath in zip(chunks, chunk_out_paths)]
        )

    df = pl.concat(chunk_out_paths)

    outpath = os.path.join(output_folder, '02_barcoded')
    make_dir(outpath)
    path = os.path.join(outpath, f"barcoded_{well}.parquet")
    df.write_parquet(path, )
    
    return {well: path}


def process_chunk(chunk: str, barcodes: Set[str], check_rc: bool, outpath: str) -> None:
    # To-do
    return



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
        df = pl.read_parquet(barcoded_wells[well])

    
    
    return