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
    """Generate one synthetic SimPlex dataset + ground truth, then return the ``reads/`` dir.

    Simulates the PairPlex wet lab from real paired antibody sequences so you can run the
    output through PairPlex and score how well it recovers the correct heavy/light pairs.
    Pipeline: load & subsample cells -> assign overloaded droplets/barcodes -> assign wells
    -> generate molecules (recovery, UMIs, resident/free split, inherited RT error) ->
    route/amplify (free molecules redistribute across wells keeping barcode+UMI; molecule
    survival; per-read index hopping) -> sequencing error -> build ground-truth tables ->
    write merged per-well FASTQ. Refuses a non-empty ``output_directory``.

    Parameters
    ----------
    input_data : str | Path
        Paired parquet with (at least) ``sequence_id:0``, ``sequence:0``, ``sequence_id:1``,
        ``sequence:1`` and ``locus:0`` / ``locus:1``. ``name``, if present, becomes the stable
        ``source_pair_id``. (This is the shape of PairPlex's own ``*_paired.parquet`` output.)
    output_directory : str | Path
        Must be empty/nonexistent. Receives ``reads/`` (one FASTQ per well), ``truth/``
        (``truth_components``/``truth_cells``/``truth_barcodes`` parquet), ``simplex_config.json``
        and ``run_manifest.json``.
    **knobs
        Any :class:`~simplex.config.SimplexConfig` field. Defaults are an **illustrative
        baseline — calibrate the uncertain ones to your assay and sweep ranges**. Key knobs:

        - Sampling: ``n_cells`` (None=all), ``seed``.
        - Overloading: ``cells_per_droplet_mean`` (loading rate lambda = cells per GEM; Poisson
          occupancy over ``round(n_cells/lambda)`` droplets), ``cells_per_droplet_overdispersion``
          (>=0, 0=pure Poisson; >0 adds NB-like clumping), ``wells``, ``barcode_pool_size``.
        - Capture/depth: ``recovery_rate``, ``molecules_per_chain_mean``,
          ``molecule_survival_rate``, ``reads_per_molecule_mean``.
        - Contamination: ``release_rate`` (ambient — fraction of molecules that drift to another
          well keeping barcode+UMI), ``index_hop_rate``.
        - Errors: ``rt_sub_rate``/``rt_indel_rate`` (inherited per read family), and
          ``sequencing_sub_rate``/``sequencing_indel_rate`` (independent per read).
        - Structure/output: ``chemistry``, ``output_mode`` ("merged" only), ``rc_fraction``,
          ``variable_length``; ``barcode_length``/``umi_length``/``tso`` are fixed & validated.
        - Bookkeeping: ``write_read_truth``.

    Returns
    -------
    pathlib.Path
        The ``output_directory/reads`` directory (pass this to ``pairplex.run(sequences=...)``).

    See Also
    --------
    simplex.score : score a PairPlex run against the truth produced here.
    simplex.config.SimplexConfig : the full knob surface, with per-knob documentation.
    """
    cfg=SimplexConfig(input_data=str(input_data), output_directory=str(output_directory), **knobs)
    out=Path(output_directory)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output dir {out} not empty; refusing to overwrite an experiment")
    out.mkdir(parents=True, exist_ok=True)
    cells=load_pairs(cfg.input_data, cfg.n_cells, cfg.seed); cfg.validate(actual_n_cells=cells.height)
    cells=assign_droplets_and_barcodes(cells, cfg.cells_per_droplet_mean, cfg.cells_per_droplet_overdispersion, cfg.chemistry, cfg.barcode_pool_size, cfg.seed)
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
