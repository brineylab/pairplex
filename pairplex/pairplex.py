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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import abstar
import abutils
import polars as pl
from natsort import natsorted
from tqdm.auto import tqdm

from .utils import parse_barcodes, process_droplet

######################################################
##                Main function                     ##
######################################################


def run(
    sequencing_directory: list[str | Path],
    output_directory: str | Path,
    temp_directory: str | Path = "/tmp",
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
    quiet: bool = False,
    debug: bool = False,
) -> list:
    """ """
    # setup directories
    sequencing_directory = Path(sequencing_directory).resolve()
    output_directory = Path(output_directory).resolve()
    temp_directory = Path(temp_directory).resolve()
    log_directory = output_directory / "logs"
    parsed_directory = output_directory / "parsed"
    consensus_directory = output_directory / "consensus"
    metadata_directory = output_directory / "metadata"
    annotated_directory = output_directory / "annotated"
    output_directory.mkdir(parents=True, exist_ok=True)
    temp_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    parsed_directory.mkdir(parents=True, exist_ok=True)
    consensus_directory.mkdir(parents=True, exist_ok=True)
    metadata_directory.mkdir(parents=True, exist_ok=True)
    annotated_directory.mkdir(parents=True, exist_ok=True)

    # process input files
    input_files = abutils.io.list_files(
        str(sequencing_directory),
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
    input_files = natsorted([f for f in input_files if "Unassigned" not in f])

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
            interleaved=False,
            debug=debug,
            show_progress=False,
        )

    # setup the main progress bar (tracks input file completion)
    main_pbar = tqdm(
        total=len(input_files),
        desc="input files",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )

    # initialize the Process Pool
    with ProcessPoolExecutor(
        max_workers=mp.cpu_count(), mp_context=mp.get_context("spawn")
    ) as executor:
        for input_file in natsorted(input_files):
            to_delete = []

            # setup text printers (using tqdm so they get cleared once file is processed)
            name_printer = tqdm(total=0, bar_format="{desc}", position=3, leave=False)
            seqs_printer = tqdm(total=0, bar_format="{desc}", position=4, leave=False)
            valids_printer = tqdm(total=0, bar_format="{desc}", position=5, leave=False)
            contig_printer = tqdm(total=0, bar_format="{desc}", position=7, leave=False)
            pairs_printer = tqdm(total=0, bar_format="{desc}", position=8, leave=False)

            # process the input file
            input_file = Path(input_file)
            name = input_file.stem
            name_printer.set_description(f"---- {name} ----")
            # count sequences
            input_count = 0
            for s in abutils.io.parse_fastx(str(input_file)):
                input_count += 1
            seqs_printer.set_description(f"{input_count} input sequences")

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
            futures = [
                executor.submit(
                    parse_barcodes, chunk, temp_directory, whitelist_path=whitelist_path
                )
                for chunk in fastq_chunks
            ]
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    parquet_chunks.append(res)
            to_delete.extend(parquet_chunks)

            # concatenate parsed data into a single dataframe
            concat_parquet = abutils.io.concatenate_parquet(
                parquet_chunks, parsed_directory / f"{name}.parquet"
            )
            df = pl.read_parquet(concat_parquet)
            seqs_with_barcodes = df.shape[0]
            valids_printer.set_description(
                f"{seqs_with_barcodes} sequences with valid barcodes"
            )

            # partition into separate parquet files by barcode
            main_pbar.set_postfix_str("partitioning barcodes", refresh=True)
            partitions = df.partition_by("barcode", as_dict=True)

            # filter partitinos by size
            partitions = {
                k: v for k, v in partitions.items() if v.shape[0] >= min_cluster_reads
            }

            # --------------------
            #      consensus
            # --------------------

            # setup the consensus progress bar
            consensus_pbar = tqdm(
                total=len(partitions),
                desc="consensus sequences",
                position=6,
                leave=False,
                dynamic_ncols=True,
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
                "quiet": quiet,
                "debug": debug,
            }
            for bc, bc_df in partitions.items():
                if isinstance(bc, tuple):
                    bc = bc[0]
                futures.append(
                    executor.submit(
                        process_droplet,
                        name=name,
                        partition_df=bc_df,
                        **consensus_kwargs,
                    )
                )

            # collect metadata
            metadata = []
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    metadata.append(res)
                consensus_pbar.update(1)
            metadata_df = pl.DataFrame(metadata)

            # write metadata to file
            metadata_file = metadata_directory / f"{name}.csv"
            metadata_df.write_csv(metadata_file)

            # write consensus sequences to file
            consensus_file = consensus_directory / f"{name}.fasta"
            filtered_df = metadata_df.filter(pl.col("pass_filters"))
            with consensus_file.open("w") as f:
                for name, consensus in zip(
                    filtered_df["name"], filtered_df["consensus"]
                ):
                    f.write(f">{name}\n{consensus}\n")
            contig_printer.set_description(
                f"{filtered_df.shape[0]} consensus sequences"
            )

            # --------------------
            #     annotation
            # --------------------

            sequences = abstar.run(
                sequences=consensus_file,
                germline_database=germline_database,
                receptor=receptor,
            )

            # unpaired sequences
            unpaired_airr_file = annotated_directory / f"{name}_unpaired.tsv"
            unpaired_parquet_file = annotated_directory / f"{name}_unpaired.parquet"
            abutils.io.to_airr(sequences, str(unpaired_airr_file))
            abutils.io.to_parquet(sequences, str(unpaired_parquet_file))

            # paired sequences
            paired_airr_file = annotated_directory / f"{name}_paired.tsv"
            paired_parquet_file = annotated_directory / f"{name}_paired.parquet"
            pairs = abutils.tl.assign_pairs(sequences, delim="_", delim_occurance=-1)
            pairs = [p for p in pairs if len(p.heavies) == 1 and len(p.lights) == 1]
            pairs_printer.set_description(f"{len(pairs)} paired sequences")
            abutils.io.to_airr(pairs, str(paired_airr_file))
            abutils.io.to_parquet(pairs, str(paired_parquet_file))

            # --------------------
            #      cleanup
            # --------------------

            if debug:
                main_pbar.set_postfix_str("cleaning up", refresh=True)
                for f in to_delete:
                    if f is not None:
                        if os.path.exists(f):
                            os.remove(f)
                if os.path.isdir(temp_directory) and not os.listdir(temp_directory):
                    os.rmdir(temp_directory)

            # close out sub-progress bars
            name_printer.close()
            seqs_printer.close()
            valids_printer.close()
            contig_printer.close()
            pairs_printer.close()

            # update the main progress bar
            main_pbar.update(1)

    # return partition_files
