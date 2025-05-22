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


import logging
from pathlib import Path
from typing import Optional
import subprocess as sp


def make_fastq(
    sequence_dir: str,
    output_directory: str,
    samplesheet: Optional[str] = None,
    platform: str = "illumina",
    debug: Optional[bool] = False,
) -> str:
    """
    Performs basecalling from BCL files to generate FASTQ files.

    Parameters
    ----------
    sequences : str
        Path to the sequences file.
    output_directory : str
        Path to the output directory.
    platform : str, optional
        Sequencing platform, by default "illumina"

    Returns
    -------
    The path of the output_directory
    """

    # Preflight
    seq_dir = Path(sequence_dir)
    if not seq_dir.exists() or not seq_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory does not exist: {seq_dir}")

    supported_platforms = {"illumina", "element"}
    if platform.lower() not in supported_platforms:
        raise ValueError(f"Platform '{platform}' is not supported. Supported: {supported_platforms}")

    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    samplesheet_path = None

    if samplesheet is not None:
        samplesheet_path = Path(samplesheet)
        if samplesheet_path.suffix.lower() in {".yaml", ".yml"}:
            csv_samplesheet = convert_yaml_to_csv(samplesheet_path, platform)
        if samplesheet_path.suffix.lower() == ".csv":
            csv_samplesheet = samplesheet_path
    else:
        for f in seq_dir.iterdir():
            if f.suffix.lower() == ".csv" and any(["samplesheet" in f.name.lower(), "runmanifest" in f.name.lower()]):
                csv_samplesheet = f

    if not csv_samplesheet:
        raise FileNotFoundError("No samplesheet found. Please provide a samplesheet or place it in the sequence directory.")

    # Basecalling
    if platform.lower() == "illumina":
        basecalling_cmd = f"bcl2fastq --runfolder-dir {seq_dir} --output-dir {out_dir} --sample-sheet {csv_samplesheet}"

    elif platform.lower() == "element":
        basecalling_cmd = f"bases2fastq {seq_dir} {out_dir} --run-manifest {csv_samplesheet}"
        if not debug:
            basecalling_cmd += " --skip-qc-report"

    with sp.Popen(basecalling_cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE) as proc:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            logging.error(f"Basecalling failed: {stderr.decode()}")
            raise RuntimeError(f"Basecalling failed: {stderr.decode()}")

    if debug:
        logging.debug(f"Basecalling output: {stdout.decode()}")

    print(f"Basecalling completed. Output files are in: {out_dir}")
    # return str(out_dir)


def convert_yaml_to_csv(yaml_file: Path, platform: str) -> Path:
    """
    Convert a YAML samplesheet to a CSV samplesheet.

    Parameters
    ----------
    yaml_file : Path
        Path to the YAML samplesheet.
    platform : str
        Sequencing platform, either "illumina" or "element".

    Returns
    -------
    Path
        Path to the converted CSV samplesheet.
    """

    if platform is "illumina":
        csv_file = build_illumina_samplesheet(yaml_file)

    elif platform is "element": 
        csv_file = build_elemnt_runmanifest(yaml_file)

    return csv_file


def build_illumina_samplesheet(yaml_file: Path) -> Path:
    """
    Build an Illumina samplesheet from a YAML file.

    Parameters
    ----------
    yaml_file : Path
        Path to the YAML samplesheet.

    Returns
    -------
    Path
        Path to the converted CSV samplesheet.
    """
    # Placeholder for actual implementation
    pass


def build_elemnt_runmanifest(yaml_file: Path) -> Path:
    """
    Build an Element run manifest from a YAML file.

    Parameters
    ----------
    yaml_file : Path
        Path to the YAML samplesheet.

    Returns
    -------
    Path
        Path to the converted CSV samplesheet.
    """
    # Placeholder for actual implementation
    pass