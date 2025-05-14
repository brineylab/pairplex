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



from abutils.io import list_files, make_dir, from_polars, to_fasta
from abutils.tools import cluster
from abutils import Sequence
import abstar
from abstar.preprocess import merging
import re, os, subprocess, shutil, tempfile, sys, multiprocessing, logging, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, Counter
from natsort import natsorted
from pathlib import Path
from typing import Set, Tuple
import polars as pl
import pandas as pd

from pairplex.utils import *





######################################################
##                Main function                     ##
######################################################

def main(sequencing_folder: str = "./", 
         output_folder: str = "./pairplexed/",
         barcodes: str = "5prime",
         enforce_bc_whitelist: bool = True,
         chunk_size: int = 100_000,
         threads: int = 32,
         min_cluster_size: int = 3,
         min_umi_count: int = 2,
         consentroid: str = "consensus",
         verbose: bool = False,
         debug: bool = False
        ):
    
    """PairPlex main routine"""


    ###################### Pre-flight ######################
    make_dir(output_folder)
    temp_folder = os.path.join(output_folder, 'temp')
    log_folder = os.path.join(output_folder, '00_logs')
    make_dir(log_folder)
    make_dir(temp_folder)
    global logger 
    logger = setup_logger(log_folder, debug)
    logger.info("=== Starting PairPlex pipeline ===")


    
    ###################### Pre-processing data ######################
    files = list_files(sequencing_folder, recursive=True, extension="fastq.gz")
    files = [f for f in files if 'Unassigned' not in f]
    if any(("R2" in f) for f in files):
        # Paired-end sequencing, requires merging
        merged_files = merge(files=files, output_folder=output_folder, log_folder=log_folder, verbose=verbose, debug=debug)
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
        fastq_chunks = split_fastq(input_file=fastq, output_dir=Path(temp_folder), lines_per_chunk=4*chunk_size)

        # Then, we assign barcodes/UMI and TSO for every chunk and concatenate results in a single file
        barcoded = assign_bc_parallel(well=well,
                                      chunks=fastq_chunks, 
                                      barcodes_path=barcodes_path, 
                                      threads=min(len(fastq_chunks), threads),
                                      output_folder=output_folder,
                                      temp_folder=temp_folder,
                                      enforce_bc_whitelist=enforce_bc_whitelist,
                                      check_rc=True, 
                                      verbose=verbose, 
                                      debug=debug)

        barcoded_wells[well] = barcoded

    
    ###################### Processing individual cells/droplets ######################
    logger.info("=== Processing individual cells/droplets ===")

    for well in barcoded_wells:
        if verbose or debug:
            logger.info(f"[{well}] Processing cells from {barcoded_wells[well]['records']}")

        well_contigs = []
        well_metadata = []

        cluster_folder = os.path.join(temp_folder, well)
        make_dir(cluster_folder)

        df = pl.read_parquet(barcoded_wells[well]['records'])
        cells = df['barcode'].unique()

        if verbose:
            logger.info(f"[{well}] Found {len(cells)} cells")

        # This needs to be parallelized
        for cell in cells:

            sequence_bin = from_polars(df.filter(pl.col('barcode') == cell), id_key="seq_id", sequence_key="sequence")
            chain_bins = cluster.cluster(sequence_bin, threshold=0.80, temp_dir=cluster_folder, debug=False)

            if debug:
                logger.debug(f"[{well}] Cell {cell} → {len(chain_bins)} clusters")

            for i, chain_bin in enumerate(chain_bins):
                if chain_bin.size < min_cluster_size:
                    if debug:
                        logger.debug(f"[{well}] Skipping cluster {i} of cell {cell} (size={chain_bin.size})")
                    continue

                seq_ids = chain_bin.seq_ids
                umi_count = (
                    df.filter(
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
                well_contigs.append(chain)
                well_metadata.append({
                    "sequence_id": name,
                    "UMI_count": umi_count,
                    "reads": chain_bin.size
                })

        if not debug:
            # Clean up temporary files
            shutil.rmtree(cluster_folder)


        # Save contigs
        contig_folder = os.path.join(output_folder, '03_contigs')
        make_dir(contig_folder)
        contig_path = os.path.join(contig_folder, f"{well}_contigs.fasta")
        to_fasta(sequences=well_contigs, fasta_file=contig_path)
        if verbose:
            logger.debug(f"[{well}] Saved {len(well_contigs)} contigs to {contig_path}")


        # Save metadata
        metadata_folder = os.path.join(output_folder, '04_metadata')
        make_dir(metadata_folder)
        metadata_file = os.path.join(metadata_folder, f"{well}_metadata.csv")
        df_metadata = pd.DataFrame(well_metadata)
        df_metadata.to_csv(metadata_file, index=False)
        if verbose:
            logger.debug(f"[{well}] Metadata written to {metadata_file}")


    ###################### Running AbStar ######################
    logger.info("=== Running AbStar annotation ===")

    # Pre-flight
    abstar_folder = os.path.join(output_folder, '05_annotated')
    make_dir(abstar_folder)

    # Files
    contig_fastas = list_files(contig_folder, recursive=True, extension="fasta")
    logger.info(f"Found {len(contig_fastas)} contig FASTA files to annotate.")

    # Run AbStar
    for file in contig_fastas:
        try:
            if verbose or debug:
                logger.info(f"Annotating {file} with AbStar...")
            abstar.run(
                sequences=file,
                germline_database='human',
                project_path=abstar_folder,
                verbose=verbose,
                debug=debug
            )
            if debug:
                logger.debug(f"Finished annotation for {file}")
        except Exception as e:
            logger.error(f"AbStar failed on file {file}: {e}")
            continue

    logger.info("AbStar annotation completed.")


    ###################### Pairing chains ######################
    logger.info("=== Pairing chains  ===")



    return