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

from ..pairplex import run as run_pairplex


@click.group()
def cli():
    pass


@cli.command()
@click.argument("sequences", type=click.Path(exists=True))
@click.argument("--output-directory", type=click.Path())
@click.option("--temp-directory", type=click.Path(), default="/tmp")
@click.option("--whitelist-path", type=click.Path(), default=None)
@click.option(
    "--platform", type=click.Choice(["illumina", "element"]), default="illumina"
)
@click.option("--clustering-threshold", type=float, default=0.9)
@click.option("--min-cluster-reads", type=int, default=3)
@click.option("--min-cluster-umis", type=int, default=1)
@click.option("--min-cluster-fraction", type=float, default=0.0)
@click.option("--consensus-downsample", type=int, default=100)
@click.option("--merge-paired-reads", is_flag=True)
@click.option("--receptor", type=click.Choice(["bcr", "tcr"]), default="bcr")
@click.option("--germline-database", type=str, default="human")
@click.option("--quiet", is_flag=True)
@click.option("--debug", is_flag=True)
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
