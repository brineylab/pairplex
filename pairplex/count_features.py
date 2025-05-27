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

from pathlib import Path
from natsort import natsorted
import abutils
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import polars as pl

from .utils import parse_fbc

def count_features(
    sequences: str | Path,
    output_directory: str | Path,
    whitelist_path: str | Path | None = None,
    antigen_barcodes: str | Path | None = None,
    debug: bool = False,
):
    """
    Count features in the sequencing folder and save the results to the output directory.
    
    Args:
        sequences (str | Path): Path to the sequencing folder.
        output_directory (str | Path): Path to the output directory.
        whitelist_path (str | Path | None): Path to the cell barcode whitelist file or name of a built-in whitelist.
            If None, default whitelist will be used.
        antigen_barcodes (str | Path): Path to the antigen barcodes file.
            If None, default antigen barcodes (x50) will be used.
        debug (bool): Whether to enable debug mode, which saves all temporary files to ease troubleshooting.
            Default is False.
    """

    # sequence_data = Path(sequence_data)
    output_directory = Path(output_directory)
    # antigen_barcodes = Path(antigen_barcodes)  # no need, this is handled in parse_fbc in utils.py
    temp_directory = output_directory / "temp"
    parsed_directory = output_directory / "parsed"
    output_directory.mkdir(parents=True, exist_ok=True)
    temp_directory.mkdir(parents=True, exist_ok=True)
    parsed_directory.mkdir(parents=True, exist_ok=True)
    if debug:
        print(f"Debug mode is enabled. All temporary files will be saved in {output_directory}")

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
    input_files = [f for f in input_files if "_R2_" in f] # as discussed, there might be a better way to do this, but we only need to go through read 2 


    main_pbar = tqdm(
        total=len(input_files),
        desc="pairplex",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )

    blank1_printer = tqdm(total=0, bar_format=" ", position=1, leave=True)
    running_total_printer = tqdm(total=0, bar_format="{desc}", position=2, leave=True)
    blank2_printer = tqdm(total=0, bar_format=" ", position=3, leave=True)
    all_valid_barcodes = 0

     # initialize the Process Pool
    with ProcessPoolExecutor(
        max_workers=mp.cpu_count(), mp_context=mp.get_context("spawn")
    ) as executor:
        for input_file in natsorted(input_files):
            to_delete = []

            name_printer = tqdm(total=0, bar_format="{desc}", position=4, leave=False)
            seqs_printer = tqdm(total=0, bar_format="{desc}", position=5, leave=False)
            valids_printer = tqdm(total=0, bar_format="{desc}", position=6, leave=False)

            input_file = Path(input_file)
            name = input_file.stem
            name_printer.set_description_str(f"---- {name} ----")
            
            # count sequences
            input_count = 0
            for s in abutils.io.parse_fastx(str(input_file)):
                input_count += 1
            seqs_printer.set_description_str(f"{input_count} input sequences")

            # split input file into chunks
            main_pbar.set_postfix_str("splitting input file", refresh=True)
            fastq_chunks = abutils.io.split_fastx(
                fastx_file=str(input_file),
                output_directory=str(temp_directory),
                chunksize=1000,
            )
            to_delete.extend(fastq_chunks)

            ########################
            #  Parse barcodes
            ########################

            main_pbar.set_postfix_str("parsing barcodes", refresh=True)
            parquet_chunks = []
            futures = [
                executor.submit(
                    parse_fbc, chunk, temp_directory, whitelist_cell_bc=whitelist_path, whitelist_feature_bc=antigen_barcodes, strict=False
                )
                for chunk in fastq_chunks
            ]
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    parquet_chunks.append(res)
            to_delete.extend(parquet_chunks)

            # concatenate parsed data into a single dataframe
            if parquet_chunks:
                concat_parquet = abutils.io.concatenate_parquet(
                    parquet_chunks, parsed_directory / f"{name}.parquet"
                )
                df = pl.read_parquet(concat_parquet)
            else:
                raise ValueError(
                    f"No valid barcodes found in {input_file}. Please check the input file."
                )


            ########################
            # count valid barcodes
            ########################

            # partition into separate parquet files by barcode
            main_pbar.set_postfix_str("partitioning barcodes", refresh=True)
            partitions = df.partition_by("barcode", as_dict=True)

            # we don't need to filter by size, we're counting all of them

            # count valid barcodes
            main_pbar.set_postfix_str("counting valid barcodes", refresh=True)



    return df