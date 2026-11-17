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

import click

from ..correct_barcodes import run as run_correct_barcodes
from ..make_fastq import run as run_make_fastq
from ..pairplex import run as run_pairplex
from ..pooled_pairing import run_pooled_pairing
from ..version import __version__


@click.group()
def cli():
    pass


@cli.command()
def version():
    print(f"PairPlex v{__version__}")


@cli.command()
@click.argument("sequences", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path())
@click.option(
    "--temp_directory",
    type=click.Path(),
    default="/tmp",
    help="Temporary directory for intermediate files",
)
@click.option(
    "--whitelist_path",
    type=click.Path(),
    default=None,
    help="Path to the whitelist file or name of a built-in whitelist",
)
@click.option(
    "--platform",
    type=click.Choice(["illumina", "element"]),
    default="illumina",
    help="Sequencing platform",
)
@click.option(
    "--clustering_threshold", type=float, default=0.9, help="Clustering threshold"
)
@click.option(
    "--min_cluster_reads",
    type=int,
    default=3,
    help="Minimum number of reads in a cluster",
)
@click.option(
    "--min_cluster_umis",
    type=int,
    default=1,
    help="Minimum number of UMIs in a cluster. Defaults to 1; 2 is recommended for datasets with heavy ambient contamination.",
)
@click.option(
    "--min_cluster_fraction",
    type=float,
    default=0.0,
    help="Minimum fraction of reads in a cluster",
)
@click.option(
    "--consensus_downsample",
    type=int,
    default=100,
    help="Downsample when calculating consensus sequence",
)
@click.option(
    "--merge_paired_reads",
    is_flag=True,
    help="Whether to merge paired reads, requires paired-end reads as input",
)
@click.option(
    "--receptor", type=click.Choice(["bcr", "tcr"]), default="bcr", help="Receptor type"
)
@click.option(
    "--germline_database",
    type=str,
    default="human",
    help="Germline database for V(D)J assignment",
)
@click.option(
    "--mmseqs_threads",
    type=click.IntRange(min=1),
    default=1,
    help="Threads for MMseqs2 during annotation (always explicit; None made MMseqs2 use every node CPU and crash)",
)
@click.option(
    "--tso_g_max",
    type=click.IntRange(min=0),
    default=3,
    help="Maximum number of Gs stripped after the TSO prefix (the TSO's rGrGrG run); genuine 5' Gs are kept",
)
@click.option(
    "--legacy",
    is_flag=True,
    help="Reproduce v2-base behaviour exactly (overrides the options of newer fixes)",
)
@click.option("--quiet", is_flag=True, help="Whether to suppress output")
@click.option(
    "--debug",
    is_flag=True,
    help="Whether to enable debug mode, which saves all temporary files to ease troubleshooting",
)
def run(
    sequences: str | Path,
    output_directory: str | Path,
    temp_directory: str | Path,
    whitelist_path: str | Path | None,
    platform: str,
    clustering_threshold: float,
    min_cluster_reads: int,
    min_cluster_umis: int,
    min_cluster_fraction: float,
    consensus_downsample: int,
    merge_paired_reads: bool,
    receptor: str,
    germline_database: str,
    mmseqs_threads: int,
    tso_g_max: int,
    legacy: bool,
    quiet: bool,
    debug: bool,
):
    run_pairplex(
        sequences=sequences,
        output_directory=output_directory,
        temp_directory=temp_directory,
        whitelist_path=whitelist_path,
        platform=platform,
        clustering_threshold=clustering_threshold,
        min_cluster_reads=min_cluster_reads,
        min_cluster_umis=min_cluster_umis,
        min_cluster_fraction=min_cluster_fraction,
        consensus_downsample=consensus_downsample,
        merge_paired_reads=merge_paired_reads,
        receptor=receptor,
        germline_database=germline_database,
        mmseqs_threads=mmseqs_threads,
        tso_g_max=tso_g_max,
        legacy=legacy,
        quiet=quiet,
        debug=debug,
    )


@cli.command()
@click.argument("sequences", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path())
@click.option(
    "--temp_directory",
    type=click.Path(),
    default="/tmp",
    help="Temporary directory for intermediate files",
)
@click.option(
    "--whitelist_path",
    type=click.Path(),
    default=None,
    help="Path to the whitelist file or name of a built-in whitelist",
)
@click.option(
    "--check_rc",
    is_flag=True,
    help="Whether to check the reverse complement of the barcodes",
)
@click.option(
    "--annotate/--no-annotate",
    default=True,
    help="Whether to annotate the sequences",
)
@click.option(
    "--receptor", type=click.Choice(["bcr", "tcr"]), default="bcr", help="Receptor type"
)
@click.option(
    "--germline_database",
    type=str,
    default="human",
    help="Germline database for V(D)J assignment",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Whether to enable debug mode, which saves all temporary files to ease troubleshooting",
)
@click.option("--quiet", is_flag=True, help="Whether to suppress output")
def correct_barcodes(
    sequences: str | Path,
    output_directory: str | Path,
    temp_directory: str | Path,
    whitelist_path: str | Path | None,
    check_rc: bool,
    annotate: bool,
    receptor: str,
    germline_database: str,
    debug: bool,
    quiet: bool,
):
    run_correct_barcodes(
        sequences=sequences,
        output_directory=output_directory,
        temp_directory=temp_directory,
        whitelist_path=whitelist_path,
        check_rc=check_rc,
        annotate=annotate,
        receptor=receptor,
        germline_database=germline_database,
        debug=debug,
        quiet=quiet,
    )


@cli.command()
@click.argument("sequencing_folder", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path())
@click.option(
    "--samplesheet",
    type=click.Path(),
    default=None,
    help="Path to the samplesheet file (CSV or YAML)",
)
@click.option(
    "--platform",
    type=click.Choice(["illumina", "element"]),
    default="illumina",
    help="Sequencing platform",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Whether to enable debug mode, which saves all temporary files to ease troubleshooting",
)
def make_fastq(
    sequencing_folder: str | Path,
    output_directory: str | Path,
    samplesheet: str | Path | None,
    platform: str,
    debug: bool,
):
    run_make_fastq(
        sequence_dir=sequencing_folder,
        output_directory=output_directory,
        samplesheet=samplesheet,
        platform=platform,
        debug=debug,
    )


@cli.command(name="pair-pooled")
@click.argument("run_directory", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), default=None, help="Output parquet (default: RUN_DIRECTORY/pooled/pooled_pairs.parquet)")
@click.option("--wells", type=str, default=None, help="Well range of one reaction, e.g. 1-96 (default: all wells)")
@click.option("--leakage", type=float, default=None, help="Pre-split leakage f in [0,1); estimated from the data if omitted")
@click.option("--min-confidence", type=float, default=0.0, help="Drop pairs below this confidence (default keeps all, scored)")
@click.option("--ambient-filter/--no-ambient-filter", default=False, help="Exclude likely ambient copies (opt-in; validate first)")
def pair_pooled_cmd(run_directory, output, wells, leakage, min_confidence, ambient_filter):
    """Pool a per-well run by droplet barcode and pair H/L by shared home well, with a confidence per pair."""
    rng = tuple(int(x) for x in wells.split("-")) if wells else None
    pairs, info = run_pooled_pairing(run_directory, output, rng, leakage, min_confidence, ambient_filter)
    print(f"wells={info['wells']} leakage={info['leakage']:.3f} ({'estimated' if info['leakage_estimated'] else 'given'}) "
          f"species={info['species']} pairs={info['pairs']} -> {info['output']}")
    for t in (0.5, 0.9):
        print(f"  pairs with confidence >= {t}: {int((pairs['confidence'] >= t).sum())}")
