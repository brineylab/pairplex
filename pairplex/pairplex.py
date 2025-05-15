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
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import abstar
import abutils
import pandas as pd
import polars as pl
from abutils.io import from_polars, list_files, make_dir, split_fastq, to_fasta
from natsort import natsorted
from tqdm.auto import tqdm

from pairplex.utils import (
    assign_bc_paralleled,
    assign_bc_unparalleled,
    get_barcode_file,
    list_wells,
    merge,
    process_cell,
    process_cell_pair,
    setup_logger,
)

######################################################
##                Main function                     ##
######################################################


def main(
    input_directory: str,
    output_directory: str,
    barcodes: str = "5prime",
    enforce_bc_whitelist: bool = True,
    sequencer: str = "element",
    chunk_size: int = 100_000,
    n_processes: int | None = None,
    min_cluster_size: int = 3,
    min_umi_count: int = 2,
    consentroid: str = "consensus",
    only_pairs: bool = True,
    output_fmt: str = "tsv",
    verbose: bool = False,
    debug: bool = False,
):
    """PairPlex: DemultiPLEXing and PAIRing BCR sequences from combinatorial single-cellRNA sequencing experiments.

    Parameters
    ----------
    input_directory : str
        Path to the folder containing the sequencing data.

    output_directory : str
        Path to the folder where the output will be saved.

    barcodes : str
        Name of the barcode file to use. Default is "5prime". Options are "5prime", "3prime", or 'v2'.

    enforce_bc_whitelist : bool
        Whether to enforce the barcode whitelist. Default is True.

    sequencer : str
        Sequencer type. Default is "element". Options are "element" or "illumina".

    chunk_size : int
        Number of reads per chunk for parallel processing. Default is 100_000.

    n_processes : int | None
        Number of processes to use for parallel processing. Default is None (use all available cores).

    min_cluster_size : int
        Minimum number of reads to consider a cluster. Default is 3.

    min_umi_count : int
        Minimum UMI count to consider a chain as valid in a cluster. Default is 2.

    consentroid : str
        Type of consensus sequence to generate. Default is "consensus". Options are "consensus" or "centroid".

    only_pairs : bool
        Whether to only keep paired chains. Default is True.

    output_fmt : str
        Format of the output files. Default is "tsv". Options are "tsv" or "parquet".

    verbose : bool
        Whether to print verbose output. Default is False.

    debug : bool
        Whether to print debug output. Default is False.
    """

    ###################### Pre-flight ######################

    output_directory = Path(output_directory).resolve()
    temp_directory = output_directory / "temp"
    log_directory = output_directory / "00_logs"
    output_directory.mkdir(parents=True, exist_ok=True)
    temp_directory.mkdir(exist_ok=True)
    log_directory.mkdir(exist_ok=True)

    global logger
    logger = setup_logger(log_directory, verbose, debug)
    logger.info("====== Starting PairPlex pipeline ======")

    if n_processes is None:
        n_processes = mp.cpu_count()
    if n_processes > mp.cpu_count():
        # logger.warning(  # warning is the same as info in this case
        logger.info(
            f"Requested {n_processes} processes, but only {mp.cpu_count()} are available. Using {mp.cpu_count()} processes instead."
        )
        n_processes = mp.cpu_count()

    ###################### Pre-processing data ######################
    logger.info("=== Pre-processing data ===")

    files = abutils.io.list_files(
        str(input_directory),
        recursive=True,
        extension=["fastq.gz", "fq.gz", "fastq", "fq"],
    )
    files = [f for f in files if "Unassigned" not in f]

    # TODO: this is a potential problem -- what if R2 is present in the sample name?
    # example: DONOR2.fastq would trigger a merging attempt
    # we should either make read merging an explicit argument or require paired-end reads (no processing of already-merged files)
    # the first is probably better, because what if a new platform with 600bp+ single-end reads comes out?
    # could make read merging `True` by default, so that the default behavior is compatible with current Illumina/Element read profiles
    if any(("R2" in f) for f in files):
        # Paired-end sequencing, requires merging
        merged_files = merge(
            files=files,
            output_folder=output_directory,
            log_folder=log_directory,
            schema=sequencer,
            verbose=verbose,
            debug=debug,
        )
    else:
        merged_files = files

    ###################### Assigning barcodes in wells ######################
    wells = list_wells(merged_files, verbose=verbose, debug=debug)

    barcoded_wells = {}

    barcodes_path = get_barcode_file(barcodes)
    if not os.path.exists(barcodes_path):
        raise FileNotFoundError(f"Barcode file not found: {barcodes_path}")
    logger.debug(f"Using barcode file: {barcodes_path}")

    for well in natsorted(wells):
        fastq = wells[well]

        # First, we split into chunks to parallelize
        fastq_chunks = split_fastq(
            prefix=well,
            input_file=fastq,
            output_dir=Path(temp_directory),
            lines_per_chunk=4 * chunk_size,
        )

        # Then, we assign barcodes/UMI and TSO for every chunk and concatenate results in a single file
        if n_processes > 1:
            n_processes_to_use = min(n_processes, len(fastq_chunks))
            barcoded = assign_bc_paralleled(
                well=well,
                chunks=fastq_chunks,
                barcodes_path=barcodes_path,
                threads=n_processes_to_use,
                output_folder=output_directory,
                temp_folder=temp_directory,
                enforce_bc_whitelist=enforce_bc_whitelist,
                check_rc=True,
                verbose=verbose,
                debug=debug,
            )
        else:
            barcoded = assign_bc_unparalleled(
                well=well,
                chunks=fastq_chunks,
                barcodes_path=barcodes_path,
                output_folder=output_directory,
                temp_folder=temp_directory,
                enforce_bc_whitelist=enforce_bc_whitelist,
                check_rc=True,
                verbose=verbose,
                debug=debug,
            )

        barcoded_wells[well] = barcoded

    ###################### Processing individual cells/droplets ######################
    logger.info("=== Generating BCR sequences for individual cells/droplets ===")

    for well in barcoded_wells:
        start_time = time.time()

        if verbose or debug:
            logger.info(
                f"[{well}] Processing cells from {barcoded_wells[well]['records']}"
            )

        well_contigs = []
        well_metadata = []

        cluster_folder = temp_directory / well
        cluster_folder.mkdir(exist_ok=True)

        df = pl.read_parquet(barcoded_wells[well]["records"])
        cells = df["barcode"].unique()

        if verbose:
            logger.info(f"[{well}] Found {len(cells)} cells")

        # Change the value of the clustering threshold here if needed
        clustering_threshold = 0.8

        if n_processes == 1:
            for cell in cells:
                sequence_bin = df.filter(pl.col("barcode") == cell)
                results = process_cell(
                    well=well,
                    cell=cell,
                    sequence_bin=sequence_bin,
                    cluster_folder=cluster_folder,
                    clustering_threshold=clustering_threshold,
                    min_cluster_size=min_cluster_size,
                    min_umi_count=min_umi_count,
                    consentroid=consentroid,
                    debug=debug,
                )

                well_contigs.extend(results["contigs"])
                well_metadata.extend(results["metadata"])

        else:
            futures = []
            with ProcessPoolExecutor(
                max_workers=n_processes,
                mp_context=mp.get_context("fork"),
            ) as executor:
                for cell in cells:
                    sequence_bin = df.filter(pl.col("barcode") == cell).to_pandas()
                    futures.append(
                        executor.submit(
                            process_cell,
                            well=well,
                            cell=cell,
                            sequence_bin=sequence_bin,
                            cluster_folder=cluster_folder,
                            clustering_threshold=clustering_threshold,
                            min_cluster_size=min_cluster_size,
                            min_umi_count=min_umi_count,
                            consentroid=consentroid,
                            debug=debug,
                        )
                    )

                for future in tqdm(
                    as_completed(futures),
                    total=len(cells),
                    desc=f"[{well}] Processing cells",
                ):
                    result = future.result()
                    well_contigs.extend(result["contigs"])
                    well_metadata.extend(result["metadata"])

        if not debug:
            # Clean up temporary files
            shutil.rmtree(cluster_folder)

        # Save contigs
        contig_folder = output_directory / "03_contigs"
        contig_folder.mkdir(exist_ok=True)
        contig_path = contig_folder / f"{well}_contigs.fasta"
        to_fasta(sequences=well_contigs, fasta_file=contig_path)
        if verbose:
            logger.debug(f"[{well}] Saved {len(well_contigs)} contigs to {contig_path}")

        # Save metadata
        metadata_folder = output_directory / "04_metadata"
        metadata_folder.mkdir(exist_ok=True)
        metadata_file = metadata_folder / f"{well}_metadata.csv"
        df_metadata = pd.DataFrame(well_metadata)
        df_metadata.to_csv(metadata_file, index=False)
        if verbose:
            logger.debug(f"[{well}] Metadata written to {metadata_file}")

        # Loggin elapsed time
        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        logger.info(f"[{well}] Finished in {minutes:02d}:{seconds:02d} minutes")

    #
    # TODO: remove this section
    # better to just generate the contigs and metadata files and let the user run abstar themselves
    # they may want to run abstar with different parameters (custom germline database, etc.)
    #

    ###################### Running AbStar ######################
    logger.info("=== Running AbStar annotation ===")

    # Pre-flight
    abstar_folder = output_directory / "05_annotated"
    abstar_folder.mkdir(exist_ok=True)

    contig_fastas = list_files(contig_folder, recursive=True, extension="fasta")
    contig_fastas = [f for f in contig_fastas if "checkpoint" not in f]
    logger.info(f"Found {len(contig_fastas)} contig FASTA files to annotate.")

    # Run AbStar
    for file in contig_fastas:
        try:
            if verbose or debug:
                logger.info(f"Annotating {file} with AbStar...")
            abstar.run(
                sequences=file,
                germline_database="human",
                project_path=abstar_folder,
                verbose=verbose,
                debug=False,
            )
            if debug:
                logger.debug(f"Finished annotation for {file}")
        except Exception as e:
            logger.error(f"AbStar failed on file {file}: {e}")
            continue

    logger.info("AbStar annotation completed.")

    ###################### Pairing chains ######################
    logger.info("=== Pairing chains  ===")

    # Create the pairs folder
    pairs_folder = output_directory / "06_pairs"
    pairs_folder.mkdir(exist_ok=True)

    wells_metadata = list_files(
        str(output_directory / "04_metadata"), recursive=True, extension="csv"
    )
    wells_metadata = [f for f in wells_metadata if "checkpoint" not in f]
    well_to_files = {}
    for f in wells_metadata:
        m = re.search(r"([A-H][0-9]{1,2})_metadata\.csv$", f)
        if m:
            well = m.group(1)
            well_to_files[well] = f

    for well in wells:
        df = pl.read_csv(
            os.path.join(abstar_folder, "airr", f"{well}_contigs.tsv"), separator="\t"
        )
        df = df.with_columns(
            [
                pl.col("sequence_id")
                .map_elements(lambda x: x.split("_")[0])
                .alias("cell_barcode"),
                pl.col("sequence_id")
                .map_elements(lambda x: x.split("_")[1])
                .alias("contig_id"),
            ]
        )

        cells = df["cell_barcode"].unique()
        pair_dicts = []

        if n_processes == 1:
            for cell in tqdm(cells):
                cell_df = df.filter(pl.col("cell_barcode") == cell)

                if len(cell_df) == 1:
                    # Only one contig, no pairing needed
                    if only_pairs:
                        # If we only want pairs, we skip this cell
                        continue
                    else:
                        # To-do
                        pass

                elif len(cell_df) == 2:
                    # Two contigs, try to pair them
                    chain1 = cell_df.filter(pl.col("locus") == "IGH")
                    chain2 = cell_df.filter(pl.col("locus") != "IGH")

                    if len(chain1) == 1 and len(chain2) == 1:
                        # Pair the chains
                        heavy = from_polars(chain1)[0]
                        light = from_polars(chain2)[0]
                        for k in [
                            k for k in light.annotations.keys() if k.startswith("d")
                        ]:
                            light.annotations.pop(k)

                        # Gather the corresponding metadata
                        heavy_umi, heavy_reads = (
                            pl.read_csv(well_to_files[well])
                            .filter((pl.col("sequence_id") == heavy["sequence_id"]))[
                                ["UMI_count", "reads"]
                            ]
                            .row(0)
                        )
                        light_umi, light_reads = (
                            pl.read_csv(well_to_files[well])
                            .filter((pl.col("sequence_id") == light["sequence_id"]))[
                                ["UMI_count", "reads"]
                            ]
                            .row(0)
                        )

                        # Prepare the dictionary for the pair
                        pair_dict = {}
                        pair_dict["index"] = cell
                        for k, v in heavy.annotations.items():
                            pair_dict[k + ":1"] = v
                        pair_dict["umi:1"] = heavy_umi
                        pair_dict["reads:1"] = heavy_reads
                        for k, v in light.annotations.items():
                            pair_dict[k + ":2"] = v
                        pair_dict["umi:2"] = light_umi
                        pair_dict["reads:2"] = light_reads

                        pair_dicts.append(pair_dict)

                    else:
                        # If we have two contigs but they are not a pair (two heavy or two light chains), we need to decide what to do
                        if only_pairs:
                            continue
                        else:
                            # To-do
                            pass

                elif len(cell_df) > 2:
                    # More than two contigs (doublets? or secondary recombination). We need to figure out what to do in this case
                    # For now, we will just skip this cell
                    # To-do
                    pass

        elif n_processes > 1:
            pairs_dicts = []
            with ProcessPoolExecutor(
                max_workers=n_processes, mp_context=mp.get_context("spawn")
            ) as executor:
                futures = [
                    executor.submit(
                        process_cell_pair,
                        cell,
                        df.filter(pl.col("cell_barcode") == cell).to_pandas(),
                        well_to_files[well],
                        only_pairs,
                    )
                    for cell in cells
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(cells),
                    desc=f"[{well}] Pairing cells",
                ):
                    result = future.result()
                    if result:
                        pairs_dicts.append(result)

        well_pairs = pl.DataFrame(pair_dicts)

        # Save pairs
        pairs_path = os.path.join(pairs_folder, f"{well}_pairs.tsv")
        well_pairs.write_csv(pairs_path, separator="\t")

        if verbose:
            logger.debug(f"[{well}] Saved {len(well_pairs)} pairs to {pairs_path}")

    ###################### Final generation of output files ######################
    logger.info("=== Generating final output  ===")

    # Create the final output folder
    final_output_folder = output_directory / "07_final"
    final_output_folder.mkdir(exist_ok=True)

    pair_files = list_files(pairs_folder, recursive=True, extension="tsv")
    wells = [os.path.basename(f).split("_")[0] for f in pair_files]

    dfs = []
    for well, file in zip(wells, pair_files):
        _df = pl.read_csv(file, separator="\t")
        _df = _df.with_columns(pl.lit(well).alias("well"))
        dfs.append(_df)

    # Concatenate all dataframes
    final_df = pl.concat(dfs)
    total_pairs = final_df.shape[0]

    if output_fmt == "parquet":
        final_df.write_parquet(os.path.join(final_output_folder, "all_pairs.parquet"))
        if verbose:
            logger.info(
                f"Saved {total_pairs} pairs to {os.path.join(final_output_folder, 'all_pairs.parquet')}"
            )
    else:
        final_df.write_csv(
            os.path.join(final_output_folder, "all_pairs.tsv"), separator="\t"
        )
        if verbose:
            logger.info(
                f"Saved {total_pairs} pairs to {os.path.join(final_output_folder, 'all_pairs.tsv')}"
            )

    return
