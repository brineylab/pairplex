# Copyright (c) 2025 brineylab
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import abstar
import abutils
import polars as pl
from natsort import natsorted
from tqdm.auto import tqdm

from .utils import parse_barcodes, print_splash


def run(
    sequences: str | Path,
    output_directory: str | Path,
    temp_directory: str | Path | None = None,
    whitelist_path: str | Path | None = None,
    check_rc: bool = True,
    annotate: bool = True,
    receptor: str = "bcr",
    germline_database: str = "human",
    merge_paired_reads: bool = False,
    platform: str = "illumina",
    debug: bool = False,
    quiet: bool = False,
) -> str | None:
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
    corrected_barcodes_directory = output_directory / "corrected_barcodes"
    output_directory.mkdir(parents=True, exist_ok=True)
    temp_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    corrected_barcodes_directory.mkdir(parents=True, exist_ok=True)
    if annotate:
        annotated_directory = output_directory / "annotated"
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

    main_pbar = tqdm(
        total=len(input_files),
        desc="correcting barcodes: ",
        position=0,
        leave=True,
        bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}",
    )

    with ProcessPoolExecutor(
        max_workers=mp.cpu_count(), mp_context=mp.get_context("spawn")
    ) as executor:
        for input_file in natsorted(input_files):
            to_delete = []

            name_printer = tqdm(total=0, bar_format="{desc}", position=2, leave=False)
            seqs_printer = tqdm(total=0, bar_format="{desc}", position=3, leave=False)

            input_file = Path(input_file)
            name = input_file.stem
            name_printer.set_description_str(f"---- {name} ----")
            # count sequences
            input_count = 0
            for s in abutils.io.parse_fastx(str(input_file)):
                input_count += 1
            seqs_printer.set_description_str(f"{input_count} input sequences")

            # break input into chunks for parallel processing
            fastq_chunks = abutils.io.split_fastx(
                fastx_file=str(input_file),
                output_directory=str(temp_directory),
                chunksize=1000,
            )
            to_delete.extend(fastq_chunks)

            # parse/correct barcodes for each chunk
            parquet_chunks = []
            futures = [
                executor.submit(
                    parse_barcodes, chunk, temp_directory, whitelist_path=whitelist_path
                )
                for chunk in fastq_chunks
            ]

            parse_pbar = tqdm(
                total=len(futures),
                desc="parse barcodes: ",
                position=4,
                leave=False,
                bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}",
            )

            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    parquet_chunks.append(res)
                parse_pbar.update(1)
            to_delete.extend(parquet_chunks)

            concat_parquet = abutils.io.concatenate_parquet(
                parquet_chunks, corrected_barcodes_directory / f"{name}.parquet"
            )

            if annotate:
                annotation_printer = tqdm(
                    total=0, bar_format="{desc}", position=5, leave=False
                )
                mmseqs_threads = None
                if input_count < 1000:
                    mmseqs_threads = 1
                # run abstar
                annotation_printer.set_description_str("annotating with abstar...")
                sequences = abstar.run(
                    sequences=str(input_file),
                    germline_database=germline_database,
                    receptor=receptor,
                    mmseqs_threads=mmseqs_threads,
                )
                annotation_printer.set_description_str("annotating with abstar...done!")

                # merge annotations and corrected_barcodes
                merging_printer = tqdm(
                    total=0, bar_format="{desc}", position=6, leave=False
                )
                merging_printer.set_description_str(
                    "merging annotations and corrected barcodes..."
                )
                annot_df = abutils.io.to_polars(sequences)
                bc_df = (
                    pl.read_parquet(concat_parquet)
                    .rename({"seq_id": "sequence_id"})
                    .select(["sequence_id", "barcode", "umi"])
                )
                merged_df = bc_df.join(annot_df, on="sequence_id")

                # write merged dataframe to parquet
                merged_df.write_parquet(annotated_directory / f"{name}.parquet")
                merging_printer.set_description_str(
                    "merging annotations and corrected barcodes...done!"
                )

            # clean up temp files
            for f in to_delete:
                if f is not None:
                    if os.path.exists(f):
                        os.remove(f)

            main_pbar.update(1)

            # close out sub-progress bars
            time.sleep(2)
            name_printer.close()
            seqs_printer.close()
            parse_pbar.close()
            annotation_printer.close()
            merging_printer.close()

    print("\n")
    main_pbar.close()
