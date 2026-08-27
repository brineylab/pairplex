![](https://img.shields.io/pypi/v/pairplex.svg?colorB=blue)
[![tests](https://github.com/brineylab/pairplex/actions/workflows/pytest.yml/badge.svg)](https://github.com/brineylab/pairplex/actions/workflows/pytest.yml)
![](https://img.shields.io/pypi/pyversions/pairplex.svg)
![](https://img.shields.io/badge/license-MIT-blue.svg)

# PairPlex: large-scale natively paired antibody sequencing


<p align="center">
  <img src="./pairplex/data/pairplex_logo_borders.png" alt="pairplex Logo" width="400"/>
</p>

<br>

PairPlex uses combinatorial barcoding (inspired by [UDA-seq](https://doi.org/10.1038/s41592-024-02586-y)) to perform cost-effective sequencing of large numbers of natively paired antibodies by massively overloading 10x Genomics reactions.

<br>

## Installation
PairPlex can be installed with `pip`:
``` bash
pip install pairplex
```

Alternatively, you can install from source (which may not be entirely stable, so use at your own risk):
``` bash
git clone https://github.com/brineylab/pairplex
cd pairplex
pip install .
```

Installation can be quickly confirmed by checking the version:
``` bash
pairplex version
```
If the current version is displayed, installation was successful.

<br>

## Usage

##### CLI
``` bash
pairplex run /path/to/sequencing_data /path/to/output_directory
```

Sequencing data can be provided as a single FASTA/Q file or a directory containing one or more FASTA/Q files (optionally gzip compressed). If a directory is provided, all files in the directory will be processed (recursively).

Providing the `merge_paired_reads` option will instruct PairPlex to merge paired-end sequencing reads with [fastp](https://github.com/OpenGene/fastp) prior to processing, so the output of Illumina's `bcl2fastq` or Element's `bases2fastq` can be used directly as input:

``` bash
pairplex run --merge_paired_reads /path/to/paired_reads /path/to/output_directory
```

By default, we assume Illumina-style naming conventions for paired-end read files. For paired FASTQ files produced by Element's `bases2fastq`, the `platform` option should be set to `element`.
``` bash
pairplex run --merge_paired_reads --platform element /path/to/paired_reads /path/to/output_directory
```
> [!NOTE]
>If the sequencing data is already merged (or wasn't paired-end to begin with), the `platform` option is not used, since the file naming conventions are only needed to match file pairs for read merging.

The complete list of CLI options can be displayed by running:
``` bash
pairplex run --help
```

##### API
``` python
import pairplex

pairplex.run(
    sequences="/path/to/sequencing_data",
    output_directory="/path/to/output_directory",
    merge_paired_reads=True,
)
```

<br>

## Choosing thresholds with SimPlex

PairPlex's pairing quality depends on filter thresholds (`min_cluster_reads`, `min_cluster_umis`, `min_cluster_fraction`, `clustering_threshold`). To choose them on evidence rather than by guessing, this repo ships **[SimPlex](./simplex/README.md)** (`import simplex`) — a companion tool that generates **mechanism-faithful synthetic sequencing data with known ground truth**, runs it through PairPlex, and **scores** the result (correct vs mispaired pairs, recall, yield loss). Sweep thresholds against SimPlex to read off the precision/yield trade-off before committing to a production configuration.

See **[`simplex/README.md`](./simplex/README.md)** for what it simulates, the available knobs and their expected effects, and how to interpret the scored results.

<br>

## Citation
If you are using Pairplex or the dataset generated of paired antibody sequences, please cite:




