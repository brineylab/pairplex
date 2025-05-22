PairPlex: High-throughput native pairing of antibody heavy and light chains
===========================================================================


With th advent of large scale and cheaper sequencing technologies, the AIRR community has been looking for ways to obtain paired antibody sequences from high-throughput sequencing data. PairPlex is a software package that provides a solution to this problem by using heuristics to identify and pair antibody heavy and light chains from sequencing data. 

We designed PairPlex to be easy to use and flexible, allowing users to customize the pairing process to their specific needs. The software is designed to work with a variety of sequencing platforms and can handle large datasets efficiently. PairPlex is alse designed to work out of the box with the AIRR community's data standards, making it easy to integrate into existing workflows. We have paid extra attention to extensively test the different thresholds and parameters used in the pairing process, ensuring that the software is robust and reliable.

The PairPlex software is available as a Python package and can be installed using pip. The package includes a command-line interface (CLI) for easy use, as well as a Python API for more advanced users. The software is open source and is released under the MIT license, allowing users to modify and distribute the code as needed.

The resulting paired sequences are made available in AIRR-copliant formats, which grants compatibility with other tools and databases in the AIRR ecosystem. The foloowing sections provide an overview of the PairPlex software, including its features, installation instructions, and usage examples. We also provide a detailed description of the experimental (wetlab) procedure along with algorithms used in the pairing process and the underlying data structures. Finally, we discuss the limitations of the software and future directions for research in this area.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   experimental_procedure
   algorithms
   data_structures
   limitations
   future_directions
   license

