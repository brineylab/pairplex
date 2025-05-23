![](https://img.shields.io/pypi/v/pairplex.svg?colorB=blue)
[![tests](https://github.com/brineylab/pairplex/actions/workflows/pytest.yml/badge.svg)](https://github.com/brineylab/pairplex/actions/workflows/pytest.yml)
![](https://img.shields.io/pypi/pyversions/pairplex.svg)
![](https://img.shields.io/badge/license-MIT-blue.svg)

# PairPlex
Demultiplex single-cell antibody repertoires with native pairing.
---

<img src="./pairplex/data/pairplex_logo_borders.png" alt="PairPlex Logo" width="400"/>

Paired santibody sequences at high-throughput fr a fraction of the cost leaves you PairPlex? So were we!

**PairPlex** uses combinatorial barcoding and single-cell RNA-seq to obtain **paired antibody sequences** in a **super high-throughput** fashion.

In January 2025, a novel method was unveiled to massively increase the scale of single-cell sequencing by making use of a combinatorial indexing approach [1]. We took on the endeavor to adapt this approach to BCR/antibody repertoire sequencing, largely enhancing available methods to obtain natively-paired anitbody sequences at a high-throughput.
In a nutshell, this method combines 10X-Genomics approach to VDJ sequencing with the throughput of bulkNGS techniques. Thanks to the use of a 5'RACE-based approach, the obtain antibody repertoire is largely unbiased. Maximal length (2x300bp) short reads-based sequencing ensure the hightest possible quality of sequencing. 
Following sequencing, demultiPLEXing and PAIRing of sequences must be performed. PairPlex is a Python-coded pipeline that handles these tasks from sequencing data all the way to fully annotated AIRR-compatible paired sequences tables. 

Full protocol is available here: [Protocols.io][2]  
The python code for PairPlex is available in the present GitHub repository: [GitHub][3]  

Using this approach and PairPlex, we generated a database of XX million natively paired antibody sequences from 8 healthy donors. In addition, we also sequenced the immune loci for these donors and annotated the resulting antibody repertoires using customized donor-matching germline databases, hence providing an outstanding antibody repertoire.

This full dataset is made available here: [XXM-PairedAntibodyRepertoire] [4]  
Antibody sequences will also be integrated to the Observed Antibody Space (OAS) database

Welcome to a whole new antibody dimension! Yes, you too can be **PairPlex**!



[1]: Li, Y., Huang, Z., Xu, L. et al. UDA-seq: universal droplet microfluidics-based combinatorial indexing for massive-scale multimodal single-cell sequencing. Nat Methods (2025). https://doi.org/10.1038/s41592-024-02586-y  
[2]: https://protocols.io/blablabla  
[3]: https://github.com/brineylab/pairplex  
[4]: Link-to-database  



## Requirements and Installation
PairPlex makes extensive use of the following libraries: 
Installation of these should however be automatically handled (with the correct versions) by the install script

To install PairPlex, two options:
##### With Pypi
`pip install pairplex`
##### From this repository
```
git clone https://github.com/brineylab/pairplex
cd pairplex
pip install ./
```
##### Verify installation
Verifying correct installation can be done by checking the version. In the Terminal interface, use:
`pairplex --version`
The version number should be returned

## Usage
PairPlex can be used from the CLI or from the Python API

##### CLI
`pairplex run ...`

##### API
```
pairplex(sequencing_folder='./SequencingRun/', verbose=False)
```

##### Options
Many options are available. Here's a quick overview:


## Reporting bugs


## Citation
If you are using Pairplex or the dataset generated of paired antibody sequences, please cite:

**Large-scale antibody repertoire leaves you PairPlex**  
_some awesome people at the Briney lab_  
soon-to-be-published  



