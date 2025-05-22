PairPlex: High-throughput native pairing of antibody heavy and light chains
===========================================================================


.. _installation:

Installation
-------------------

PairPlex is a Python package that can be installed using pip. The package is compatible with Python 3.10 and later versions. While it hasn't been tested with previous version, it might work on decently recent Python engines. 

To install PairPlex, follow these steps:

1. In a terminal, create a new virtual environment (optional but recommended):
   ```bash
   python -m venv pairplex_env
   ```
2. Activate the virtual environment:
   - On Windows:
     ```bash
     pairplex_env\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source pairplex_env/bin/activate
     ```
3. Install PairPlex using pip:
   ```bash
   pip install pairplex
   ```
4. Verify the installation by running the following command:
   ```bash
   pairplex --version
   ```
   This should display the version of PairPlex installed.


If you are willing to install from the source, you can clone the repository and install it using pip:

1. Clone the PairPlex repository:
   ```bash
   git clone https://gihub.com/brineylab/pairplex.git
   ```
2. Navigate to the cloned directory:
   ```bash
    cd pairplex
    ```
3. Install the package using pip:
    ```bash
    pip install ./
    ```
4. Verify the installation by running the following command:
    ```bash
    pairplex --version
    ```
    This should display the version of PairPlex installed.


PairPlex has been tested on the following platforms:
- Windows 11 WSL running Ubuntu 22.04
- macOS ?
- Ubuntu 22.04
- Ubuntu 20.04
The package is designed to be cross-platform and should work on any system with Python 3.10 or later installed. If you encounter any issues during installation or usage, please report them on the GitHub repository's issue tracker. 


Required dependencies
-------------------

PairPlex has the following dependencies:
- `polars`: For high-performance data manipulation.
- `abstar`: For antibody sequence processing (assignment and annotation). [This package needs to meet the following version requirements: >=0.7.3 ]
- `abutils`: Helper package for file processing and formatted-data manipulation. [This package needs to meet the following version requirements: >=0.5.3 ]
For other packages, please refer to the `requirements.txt` file in the repository.