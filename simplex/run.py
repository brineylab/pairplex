"""Top-level pipeline orchestrator: `simplex.run(...)`.

Wires together every stage of the cells -> molecules -> routing -> reads -> truth ->
scoring pipeline (all other modules in this package) into one call that generates a
synthetic dataset, its ground truth, and a run manifest.
"""
import sys, hashlib, json
from pathlib import Path
import numpy, polars
from .cells import load_pairs, assign_droplets_and_barcodes, assign_wells
from .config import SimplexConfig
from .molecules import generate_molecules
from .routing import route_and_amplify
from .reads import apply_sequencing_errors, build_merged
from .truth import build_truth_components, build_truth_cells, build_truth_barcodes
from .io import write_merged_fastq, write_truth
try: from .version import __version__ as _SV
except Exception: _SV="0.0.0"
try: import pairplex; _PPV=getattr(pairplex,"__version__","unknown")
except Exception: _PPV="unknown"

def run(input_data, output_directory, **knobs):
    """Generate one synthetic SimPlex dataset + ground truth under `output_directory`.

    Builds a `SimplexConfig` from `input_data`/`output_directory`/`**knobs`, refuses to
    run into a non-empty `output_directory` (stale FASTQs must not contaminate a new
    experiment), then runs every pipeline stage in order: load & subsample cells ->
    assign droplets/barcodes -> assign wells -> validate the config against the actual
    cell count (the OOM guard needs the real `n_cells` when it was `None`) -> generate
    molecules -> route/amplify into reads -> apply sequencing error -> build truth
    tables -> build merged FASTQ records. Writes the merged per-well FASTQ files and
    truth parquets, plus `simplex_config.json` and `run_manifest.json` (versions, seed,
    input fingerprint, config hash, row counts) for provenance.

    Returns the path to the `reads/` directory containing the written FASTQ files.
    """
    cfg=SimplexConfig(input_data=str(input_data), output_directory=str(output_directory), **knobs)
    out=Path(output_directory)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output dir {out} not empty; refusing to overwrite an experiment")
    out.mkdir(parents=True, exist_ok=True)
    cells=load_pairs(cfg.input_data, cfg.n_cells, cfg.seed); cfg.validate(actual_n_cells=cells.height)
    cells=assign_droplets_and_barcodes(cells, cfg.cells_per_droplet_mean, cfg.cells_per_droplet_sd, cfg.chemistry, cfg.barcode_pool_size, cfg.seed)
    cells=assign_wells(cells, cfg.wells, cfg.seed)
    mols, chain_status=generate_molecules(cells, cfg.recovery_rate, cfg.molecules_per_chain_mean, cfg.release_rate, cfg.umi_length, cfg.rt_sub_rate, cfg.rt_indel_rate, cfg.seed)
    mols, reads=route_and_amplify(mols, cfg.wells, cfg.molecule_survival_rate, cfg.reads_per_molecule_mean, cfg.index_hop_rate, cfg.seed)
    reads=apply_sequencing_errors(reads, cfg.sequencing_sub_rate, cfg.sequencing_indel_rate, cfg.seed)
    comp=build_truth_components(cells, reads)
    tcells=build_truth_cells(cells, chain_status, mols, reads)
    tbar=build_truth_barcodes(cells, tcells, comp)
    built=build_merged(reads, cfg.tso, cfg.rc_fraction, cfg.variable_length, cfg.seed)
    reads_paths=write_merged_fastq(built, out)
    write_truth(out, comp, tcells, tbar, reads if cfg.write_read_truth else None)
    cfg.to_json(out/"simplex_config.json")
    manifest={"simplex_version":_SV,"pairplex_version":_PPV,"python":sys.version.split()[0],
        "polars":polars.__version__,"numpy":numpy.__version__,"seed":cfg.seed,
        "input_fingerprint":hashlib.blake2b(Path(cfg.input_data).read_bytes(),digest_size=16).hexdigest(),
        "config_hash":hashlib.blake2b(json.dumps(cfg.to_dict(),sort_keys=True).encode(),digest_size=16).hexdigest(),
        "rng_scheme":"per-stage blake2b (v1: order-dependent, not chunk-invariant)",
        "counts":{"cells":cells.height,"molecules":mols.height,"reads":reads.height,"components":comp.height,"keys":tbar.height},
        "outputs":[p.name for p in reads_paths]+["truth/truth_components.parquet","truth/truth_cells.parquet","truth/truth_barcodes.parquet"]}
    (out/"run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return out/"reads"
