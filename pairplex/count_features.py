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

def count_features(
    sequencing_folder: str | Path,
    output_directory: str | Path,
    samplesheet: str | Path | None,
    platform: str,
    debug: bool,
):
    """
    Count features in the sequencing folder and save the results to the output directory.
    
    Args:
        sequencing_folder (str | Path): Path to the sequencing folder.
        output_directory (str | Path): Path to the output directory.
        samplesheet (str | Path | None): Path to the samplesheet file (CSV or YAML).
        platform (str): Sequencing platform.
        debug (bool): Whether to enable debug mode, which saves all temporary files to ease troubleshooting.
    """

    

    return