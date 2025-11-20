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

from ..make_fastq import make_fastq as make_fastq_pairplex
from ..pairplex import run as run_pairplex
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
    help="Minimum number of UMIs in a cluster",
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
        quiet=quiet,
        debug=debug,
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
    
    make_fastq_pairplex(
        sequencing_folder=sequencing_folder,
        output_directory=output_directory,
        samplesheet=samplesheet,
        platform=platform,
        debug=debug,
    )
