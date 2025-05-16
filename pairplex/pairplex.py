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



from abutils.io import list_files, make_dir, to_fasta
from abutils import Pair
from tqdm.auto import tqdm
import abstar
import os, shutil, time
from natsort import natsorted
from pathlib import Path
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
         sequencer: str = "element",
         chunk_size: int = 100_000,
         threads: int = 32,
         min_cluster_size: int = 3,
         min_umi_count: int = 2,
         consentroid: str = "consensus",
         only_pairs: bool = True,
         output_fmt: str = "tsv",
         keep_intermediates = False,
         verbose: bool = False,
         debug: bool = False
        ):
    
    """PairPlex: DemultiPLEXing and PAIRing BCR sequences from combinatorial single-cellRNA sequencing experiments.
    Args:
        sequencing_folder (str): Path to the folder containing the sequencing data.
        output_folder (str): Path to the folder where the output will be saved.
        barcodes (str): Name of the barcode file to use. Default is "5prime". Options are "5prime", "3prime", or 'v2'.
        enforce_bc_whitelist (bool): Whether to enforce the barcode whitelist. Default is True.
        sequencer (str): Sequencer type. Default is "element". Options are "element" or "illumina".
        chunk_size (int): Number of reads per chunk for parallel processing. Default is 100_000.
        threads (int): Number of threads to use for parallel processing. Default is 32.
        min_cluster_size (int): Minimum number of reads to consider a cluster. Default is 3.
        min_umi_count (int): Minimum UMI count to consider a chain as valid in a cluster. Default is 2.
        consentroid (str): Type of consensus sequence to generate. Default is "consensus". Options are "consensus" or "centroid".
        only_pairs (bool): Whether to only keep paired chains. Default is True.
        output_fmt (str): Format of the output files. Default is "tsv". Options are "tsv" or "parquet".
        keep_intermediates (bool): Whether to keep intermediate files. Default is False.
        verbose (bool): Whether to print verbose output. Default is False.
        debug (bool): Whether to print debug output. Default is False.
    """


    ###################### Pre-flight ######################

    make_dir(output_folder)
    temp_folder = os.path.join(output_folder, 'temp')
    log_folder = os.path.join(output_folder, '00_logs')
    make_dir(log_folder)
    make_dir(temp_folder)
    global logger 
    logger = setup_logger(log_folder, verbose, debug)
    logger.info("====== Starting PairPlex pipeline ======")

    if threads > os.cpu_count():
        logger.warning(f"Requested {threads} threads, but only {os.cpu_count()} are available. Using {os.cpu_count()} threads instead.")
        threads = os.cpu_count()

    
    ###################### Pre-processing data ######################
    logger.info("=== Pre-processing data ===")

    files = list_files(sequencing_folder, recursive=True, extension="fastq.gz")
    files = [f for f in files if 'Unassigned' not in f]
    if any(("R2" in f) for f in files):
        # Paired-end sequencing, requires merging
        merged_files = merge(files=files, output_folder=output_folder, log_folder=log_folder, schema=sequencer, verbose=verbose, debug=debug)
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
        fastq_chunks = split_fastq(prefix=well, input_file=fastq, output_dir=Path(temp_folder), lines_per_chunk=4*chunk_size)

        # Then, we assign barcodes/UMI and TSO for every chunk and concatenate results in a single file
        if threads > 1:
            threads_to_use = min(threads, len(fastq_chunks))
            barcoded = assign_bc_paralleled(well=well,
                                        chunks=fastq_chunks, 
                                        barcodes_path=barcodes_path, 
                                        threads=threads_to_use,
                                        output_folder=output_folder,
                                        temp_folder=temp_folder,
                                        enforce_bc_whitelist=enforce_bc_whitelist,
                                        check_rc=True, 
                                        verbose=verbose, 
                                        debug=debug)
        else:
            barcoded = assign_bc_unparalleled(well=well,
                                        chunks=fastq_chunks, 
                                        barcodes_path=barcodes_path, 
                                        output_folder=output_folder,
                                        temp_folder=temp_folder,
                                        enforce_bc_whitelist=enforce_bc_whitelist,
                                        check_rc=True, 
                                        verbose=verbose, 
                                        debug=debug)

        barcoded_wells[well] = barcoded

    
    ###################### Processing individual cells/droplets ######################
    logger.info("=== Generating BCR sequences for individual cells/droplets ===")

    for well in barcoded_wells:
        start_time = time.time()

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

        # Change the value of the clustering threshold here if needed
        clustering_threshold = 0.8

        if threads == 1:
            for cell in cells:
                
                sequence_bin = df.filter(pl.col('barcode') == cell)
                results = process_cell(well=well,
                                    cell=cell, 
                                    sequence_bin=sequence_bin, 
                                    cluster_folder=cluster_folder, 
                                    clustering_threshold=clustering_threshold,
                                    min_cluster_size=min_cluster_size, 
                                    min_umi_count=min_umi_count, 
                                    consentroid=consentroid, 
                                    debug=debug)
                
                well_contigs.extend(results['contigs'])
                well_metadata.extend(results['metadata'])
        
        else:
            futures = []
            with ProcessPoolExecutor(
                    max_workers=threads,
                    mp_context=multiprocessing.get_context("fork"),
                ) as executor:
                    for cell in cells:
                        sequence_bin = df.filter(pl.col('barcode') == cell).to_pandas()
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

                    for future in tqdm(as_completed(futures), total=len(cells), desc=f"[{well}] Processing cells"):
                        result = future.result()
                        well_contigs.extend(result['contigs'])
                        well_metadata.extend(result['metadata'])

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

        # Loggin elapsed time
        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        logger.info(f"[{well}] Finished in {minutes:02d}:{seconds:02d} minutes")


    ###################### Running AbStar ######################
    logger.info("=== Running AbStar annotation ===")

    # Pre-flight
    abstar_folder = os.path.join(output_folder, '05_annotated')
    make_dir(abstar_folder)

    contig_fastas = list_files(contig_folder, recursive=True, extension="fasta")
    contig_fastas = [f for f in contig_fastas if 'checkpoint' not in f]
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
                debug=False
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
    pairs_folder = os.path.join(output_folder, '06_pairs')
    make_dir(pairs_folder)

    wells_metadata = list_files('./test/04_metadata/', recursive=True, extension="csv")
    wells_metadata = [f for f in wells_metadata if 'checkpoint' not in f]
    well_to_files = {}
    for f in wells_metadata:
        m = re.search(r'([A-H][0-9]{1,2})_metadata\.csv$', f)
        if m:
            well = m.group(1)
            well_to_files[well] = f

    for well in natsorted(wells):
        df = pl.read_csv(os.path.join(abstar_folder, 'airr', f"{well}_contigs.tsv"), separator="\t")
        df = df.with_columns([
            pl.col("sequence_id").map_elements(lambda x: x.split('_')[0]).alias("cell_barcode"),
            pl.col("sequence_id").map_elements(lambda x: x.split('_')[1]).alias("contig_id"),
        ])

        metadata_df = pl.read_csv(well_to_files[well]).to_pandas()
        pair_dicts = []
        unpaired_dicts = []

        if threads == 1:
            for cell in tqdm(cells, desc=f"[{well}] Pairing cells", total=len(cells)):
                cell_df = df.filter(pl.col('cell_barcode') == cell).to_pandas()
                pair, dicts = process_cell_pair(cell, cell_df, metadata_df, only_pairs)
                
                if pair:
                    pair_dicts.extend(dicts)
                else:
                    if dicts:
                        unpaired_dicts.extend(dicts)
                        
        elif threads > 1:
            futures = []
            with ProcessPoolExecutor(
                    max_workers=threads,
                    mp_context=multiprocessing.get_context("fork"),
                ) as executor:
                    for cell in tqdm(cells, total=len(cells), desc=f"[{well}] Pairing cells"):
                        cell_df = df.filter(pl.col('cell_barcode') == cell).to_pandas()
                        futures.append(
                            executor.submit(
                                process_cell_pair,
                                cell=cell, 
                                cell_df=cell_df, 
                                metadata=metadata_df, 
                                only_pairs=only_pairs
                            )
                        )

                    for future in futures:
                        result = future.result()
                        if result:
                            pair, dicts = result
                            if pair:
                                pair_dicts.extend(dicts)
                            else:
                                if dicts:
                                    unpaired_dicts.extend(dicts)


        well_pairs = pl.DataFrame(pair_dicts)
        if not only_pairs:
            well_unpaired = pl.DataFrame(unpaired_dicts)


        # Save pairs
        pairs_path = os.path.join(pairs_folder, f"{well}_pairs.tsv")
        well_pairs.write_csv(pairs_path, separator="\t")
        if not only_pairs:
            unpaired_path = os.path.join(pairs_folder, f"{well}_unpaired.tsv")
            well_unpaired.write_csv(unpaired_path, separator="\t")

        if verbose:
            logger.debug(f"[{well}] Saved {len(well_pairs)} pairs to {pairs_path}")
            if not only_pairs:
                logger.debug(f"[{well}] Saved {len(well_unpaired)} unpaired to {unpaired_path}")

    ###################### Final generation of output files ######################
    logger.info("=== Generating final output  ===")

    # Create the final output folder
    final_output_folder = os.path.join(output_folder, '07_final')
    make_dir(final_output_folder)

    # Merging all pairs
    pair_files = list_files(pairs_folder, recursive=True, extension="tsv")
    pair_files = [f for f in pair_files if 'checkpoint' not in f]
    pair_files = [f for f in pair_files if 'pairs'  in f]

    wells = [os.path.basename(f).split('_')[0] for f in pair_files]

    dfs = []
    for well, file in zip(wells, pair_files):
        _df = pl.read_csv(file, separator="\t")
        _df = _df.with_columns(pl.lit(well).alias("well"))
        dfs.append(_df)

    # Concatenate all dataframes
    final_df = pl.concat(dfs)
    total_pairs = final_df.shape[0]

    if output_fmt == "parquet":
        final_df.write_parquet(os.path.join(final_output_folder, 'all_pairs.parquet'))
        if verbose:
            logger.info(f"Saved {total_pairs} pairs to {os.path.join(final_output_folder, 'all_pairs.parquet')}")
    else:
        final_df.write_csv(os.path.join(final_output_folder, 'all_pairs.tsv'), separator="\t")
        if verbose:
            logger.info(f"Saved {total_pairs} pairs to {os.path.join(final_output_folder, 'all_pairs.tsv')}")

    # Merging all unpaired, if applicable
    if not only_pairs:
        unpaired_files = list_files(pairs_folder, recursive=True, extension="tsv")
        unpaired_files = [f for f in unpaired_files if 'checkpoint' not in f]
        unpaired_files = [f for f in unpaired_files if 'unpaired'  in f]

        dfs = []
        for well, file in zip(wells, unpaired_files):
            _df = pl.read_csv(file, separator="\t")
            _df = _df.with_columns(pl.lit(well).alias("well"))
            dfs.append(_df)

     # Concatenate all dataframes
    final_df = pl.concat(dfs)
    total_unpaired = final_df.shape[0]

    if output_fmt == "parquet":
        final_df.write_parquet(os.path.join(final_output_folder, 'all_unpaired.parquet'))
        if verbose:
            logger.info(f"Saved {total_unpaired} single sequences to {os.path.join(final_output_folder, 'all_unpaired.parquet')}")
    else:
        final_df.write_csv(os.path.join(final_output_folder, 'all_unpaired.tsv'), separator="\t")
        if verbose:
            logger.info(f"Saved {total_unpaired} single sequences to {os.path.join(final_output_folder, 'all_unpaired.tsv')}")
   


    ###################### Cleaning and losing gracefully ######################
    logger.info("=== PairPlex pipeline completed ===")

    # Clean up temporary files
    if not debug:
        shutil.rmtree(temp_folder)
    
    if not any([debug, keep_intermediates]):
        for folder in ['00_logs', '01_logs', '02_barcoded', '03_contigs', '04_metadata', '05_annotated', '06_pairs']:
            shutil.rmtree(os.path.join(output_folder, folder))
            if any([verbose, debug]):
                logger.debug(f"Removed {os.path.join(output_folder, folder)}")


    logger.info("Completely PairpPlexed!")

    return