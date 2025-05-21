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
import os
import gzip
from pathlib import Path
from typing import Set

import abutils


def make_fastq(
    sequences: str,
    output_directory: str,
    temp_directory: str = "/tmp",
    platform: str = "illumina",
) -> None:
    """
    Performs basecalling from BCL files to generate FASTQ files.

    Parameters
    ----------
    sequences : str
        Path to the sequences file.
    output_directory : str
        Path to the output directory.
    temp_directory : str, optional
        Path to the temporary directory, by default "/tmp"
    platform : str, optional
        Sequencing platform, by default "illumina"

    Returns
    -------
    None
    """

