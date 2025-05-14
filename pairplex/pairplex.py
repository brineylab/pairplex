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
    logger = setup_logger(log_folder, debug)
    logger.info("====== Starting PairPlex pipeline ======")

    if threads > os.cpu_count():
        logger.warning(f"Requested {threads} threads, but only {os.cpu_count()} are available. Using {os.cpu_count()} threads instead.")
        threads = os.cpu_count()

    
    ###################### Pre-processing data ######################
    logger.info("=== Pre-processing data ===")

    # Check if step already done
    if list_files(os.path.join(output_folder, '01_merged'), recursive=True, extension="fastq") != []:
        logger.info("Merged files already exist. Skipping merging step.")
        merged_files = list_files(os.path.join(output_folder, '01_merged'), recursive=True, extension="fastq")

    else:
        files = list_files(sequencing_folder, recursive=True, extension="fastq.gz")
        files = [f for f in files if 'Unassigned' not in f]
        if any(("R2" in f) for f in files):
            # Paired-end sequencing, requires merging
            merged_files = merge(files=files, output_folder=output_folder, log_folder=log_folder, schema=sequencer, verbose=verbose, debug=debug)
        else:
            merged_files = files

        
    ###################### Assigning barcodes in wells ######################
    wells = list_wells(merged_files, verbose=verbose, debug=debug)

    # Check if step already done
    if list_files(os.path.join(output_folder, '02_barcoded'), recursive=True, extension="parquet") != []:
        logger.info("Barcoded files already exist. Skipping barcoding step.")

        _parquet = list_files(os.path.join(output_folder, '02_barcoded'), recursive=True, extension="parquet")
        _csv = list_files(os.path.join(output_folder, '02_barcoded'), recursive=True, extension="csv")
        _parquet = [f for f in _parquet if 'checkpoint' not in f]
        _csv = [f for f in _csv if 'checkpoint' not in f]
        barcoded_wells = {}
        for c, v in zip(sorted(_parquet), sorted(_csv)):
            m = re.search(r'barcoded_([A-H][0-9]{1,2})_matches\.csv$', c)
            if m:
                barcoded_wells[well] = {
                    'records': c,
                    'metadata': v
                }

    else:    
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

    # Check if step already done
    if (list_files(os.path.join(output_folder, '04_metadata'), recursive=True, extension="csv") != []) and \
       (list_files(os.path.join(output_folder, '03_contigs'), recursive=True, extension="fasta") != []):
        logger.info("Contigs and metadata already exist. Skipping processing step.")

    else: 
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

    # Check if step already done
    if list_files(os.path.join(output_folder, '05_annotated'), recursive=True, extension="tsv") != []:
        logger.info("AbStar annotation already exists. Skipping annotation step.")

    else:
        # Files
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

    # Check if step already done
    if list_files(os.path.join(output_folder, '06_pairs'), recursive=True, extension="tsv") != []:
        logger.info("Pairs already exist. Skipping pairing step.")
        
    else:
        wells_metadata = list_files('./test/04_metadata/', recursive=True, extension="csv")
        wells_metadata = [f for f in wells_metadata if 'checkpoint' not in f]
        well_to_files = {}
        for f in wells_metadata:
            m = re.search(r'([A-H][0-9]{1,2})_metadata\.csv$', f)
            if m:
                well = m.group(1)
                well_to_files[well] = f

        for well in wells:
            df = pl.read_csv(os.path.join(abstar_folder, 'airr', f"{well}_contigs.tsv"), separator="\t")
            df = df.with_columns([
                pl.col("sequence_id").map_elements(lambda x: x.split('_')[0]).alias("cell_barcode"),
                pl.col("sequence_id").map_elements(lambda x: x.split('_')[1]).alias("contig_id"),
            ])

            cells = df['cell_barcode'].unique()
            pair_dicts = []

            for cell in tqdm(cells):
                cell_df = df.filter(pl.col('cell_barcode') == cell)
                
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
                    chain1 = cell_df.filter(pl.col('locus') == 'IGH')
                    chain2 = cell_df.filter(pl.col('locus') != 'IGH')
                    
                    if len(chain1) == 1 and len(chain2) == 1:
                        # Pair the chains
                        heavy = from_polars(chain1)[0]
                        light = from_polars(chain2)[0]
                        for k in [k for k in light.annotations.keys() if k.startswith("d")]:
                            light.annotations.pop(k)
                        
                        # Gather the corresponding metadata
                        heavy_umi, heavy_reads = pl.read_csv(well_to_files[well]).filter((pl.col("sequence_id") == heavy['sequence_id']))[["UMI_count",'reads']].row(0)
                        light_umi, light_reads = pl.read_csv(well_to_files[well]).filter((pl.col("sequence_id") == light['sequence_id']))[["UMI_count",'reads']].row(0)

                        # Prepare the dictionary for the pair
                        pair_dict = {}
                        pair_dict['index'] = cell
                        for k, v in heavy.annotations.items():
                            pair_dict[k+":1"] = v
                        pair_dict['umi:1'] = heavy_umi
                        pair_dict['reads:1'] = heavy_reads
                        for k, v in light.annotations.items():
                            pair_dict[k+":2"] = v
                        pair_dict['umi:2'] = light_umi
                        pair_dict['reads:2'] = light_reads

                        pair_dicts.append(pair_dict)

                elif len(cell_df) > 2:
                    # More than two contigs (doublets? or secondary recombination). We need to figure out what to do in this case
                    # For now, we will just skip this cell
                    # To-do
                    pass

            well_pairs = pl.DataFrame(pair_dicts)

            # Save pairs
            pairs_path = os.path.join(pairs_folder, f"{well}_pairs.tsv")
            well_pairs.write_csv(pairs_path, separator="\t")

            if verbose:
                logger.debug(f"[{well}] Saved {len(well_pairs)} pairs to {pairs_path}")

    ###################### Final generation of output files ######################
    logger.info("=== Generating final output  ===")

    # Create the final output folder
    final_output_folder = os.path.join(output_folder, '07_final')
    make_dir(final_output_folder)

    pair_files = list_files(pairs_folder, recursive=True, extension="tsv")
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


    return