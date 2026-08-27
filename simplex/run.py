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
from ._contract import BARCODE_LEN, UMI_LEN, TSO
from .molecules import generate_molecules
from .routing import route_and_amplify
from .reads import apply_sequencing_errors, build_merged
from .truth import build_truth_components, build_truth_cells, build_truth_barcodes
from .io import write_merged_fastq, write_truth
from ._log import configure_logging, logger, pbar
try: from .version import __version__ as _SV
except Exception: _SV="0.0.0"
try: import pairplex; _PPV=getattr(pairplex,"__version__","unknown")
except Exception: _PPV="unknown"

def run(
    input_data,
    output_directory,
    n_cells: int | None = None,
    wells: int = 96,
    cells_per_droplet_mean: float = 2.0,
    cells_per_droplet_overdispersion: float = 0.0,
    barcode_pool_size: int | None = None,
    recovery_rate: float = 0.5,
    molecules_per_chain_mean: float = 20.0,
    release_rate: float = 0.02,
    molecule_survival_rate: float = 0.8,
    reads_per_molecule_mean: float = 5.0,
    rt_sub_rate: float = 0.0,
    rt_indel_rate: float = 0.0,
    sequencing_sub_rate: float = 0.001,
    sequencing_indel_rate: float = 0.0,
    index_hop_rate: float = 0.001,
    barcode_length: int = BARCODE_LEN,
    umi_length: int = UMI_LEN,
    tso: str = TSO,
    chemistry: str = "v2",
    output_mode: str = "merged",
    rc_fraction: float = 0.0,
    variable_length: bool = True,
    write_read_truth: bool = False,
    seed: int = 0,
    quiet: bool = False,
    verbose: bool = False,
) -> "Path":
    """
    Generate one synthetic SimPlex dataset with mechanism-faithful ground truth.

    Simulates the PairPlex wet lab from real paired antibody sequences so the output can be
    run through PairPlex and *scored* (with :func:`simplex.score`) to measure how well a given
    PairPlex configuration recovers the correct heavy/light pairs. Pipeline: load & subsample
    cells -> assign overloaded droplets/barcodes -> assign wells -> generate molecules
    (recovery, UMIs, resident/free split, inherited RT error) -> route & amplify (free
    molecules redistribute across wells keeping barcode+UMI; molecule survival; per-read index
    hopping) -> sequencing error -> build ground-truth tables -> write merged per-well FASTQ.

    Defaults are an **illustrative baseline, not a claim about any assay** — calibrate the
    high-leverage ones (`release_rate`, `cells_per_droplet_mean`, `molecules_per_chain_mean`,
    `recovery_rate`) to your real ``metadata/*.csv`` and sweep ranges.

    Parameters
    ----------
    input_data : str | Path
        Paired parquet with (at least) ``sequence_id:0``, ``sequence:0``, ``sequence_id:1``,
        ``sequence:1`` and ``locus:0`` / ``locus:1`` columns. ``name``, if present, becomes the
        stable ``source_pair_id``. (This is the shape of PairPlex's own ``*_paired.parquet``.)
    output_directory : str | Path
        Must be empty or nonexistent (raises otherwise, so stale outputs cannot contaminate a
        run). Receives ``reads/`` (one FASTQ per well), ``truth/`` (component/cell/barcode
        parquet), ``simplex_config.json`` and ``run_manifest.json``.
    n_cells : int | None, default None
        Subsample the input to this many cells (oversamples with replacement if larger).
        ``None`` uses every input pair.
    wells : int, default 96
        Number of plate wells cells are distributed into. More wells -> fewer within-well
        barcode collisions.
    cells_per_droplet_mean : float, default 2.0
        Loading rate lambda = cells per GEM. Cells are randomly loaded into
        ``round(n_cells / lambda)`` droplets -> Poisson occupancy. Higher -> more cells share a
        barcode -> more collision risk. Realistic lambda ~= cells-loaded / GEM-barcodes (~1-3).
    cells_per_droplet_overdispersion : float, default 0.0
        ``0`` = pure Poisson occupancy; ``> 0`` makes droplet capture propensities vary
        (Dirichlet, concentration ``1/overdispersion``) -> Negative-Binomial-like clumping.
    barcode_pool_size : int | None, default None
        ``None`` = one unique barcode per droplet; an int samples droplet barcodes from a pool
        of that size, forcing controlled cross-droplet barcode reuse (a stress knob).
    recovery_rate : float, default 0.5
        Per-cell, per-chain capture probability. Lower -> more chain dropout.
    molecules_per_chain_mean : float, default 20.0
        Mean number of distinct molecules (UMIs) per captured chain. Higher -> more support.
    release_rate : float, default 0.02
        Fraction of molecules that become "free"/ambient and redistribute to a random well,
        keeping barcode+UMI (moves molecules, does not add them). The main contamination knob.
    molecule_survival_rate : float, default 0.8
        Bernoulli survival applied *before* amplification (not read thinning). Lower -> less
        support / more effective dropout.
    reads_per_molecule_mean : float, default 5.0
        Read-family depth per surviving molecule. Higher -> more reads per contig.
    rt_sub_rate, rt_indel_rate : float, default 0.0
        RT substitution / indel error, stamped once per molecule and inherited by its whole read
        family (survives consensus, so it stresses ``clustering_threshold``).
    sequencing_sub_rate : float, default 0.001
        Independent per-read substitution error (consensus removes most of it).
    sequencing_indel_rate : float, default 0.0
        Independent per-read indel error.
    index_hop_rate : float, default 0.001
        Per-read probability of misassignment to a different well (must be 0 when ``wells == 1``).
    barcode_length : int, default 16
        Fixed and validated to match ``pairplex.parse_barcodes`` (changing it raises).
    umi_length : int, default 10
        Fixed and validated to match ``pairplex.parse_barcodes`` (changing it raises).
    tso : str, default "TTTCTTATATGGG"
        Fixed and validated TSO (the parser's ``s[36:].lstrip("G")`` extraction assumes it).
    chemistry : str, default "v2"
        10x barcode whitelist to draw droplet barcodes from ("v2" or "v3").
    output_mode : str, default "merged"
        Only ``"merged"`` is supported in this version (paired-end/fastp is deferred).
    rc_fraction : float, default 0.0
        Fraction of reads written reverse-complemented (exercises orientation handling).
    variable_length : bool, default True
        Randomly truncate cDNA 5'/3' to mimic variable read/insert length.
    write_read_truth : bool, default False
        Also write a per-read provenance parquet (``truth/truth_reads.parquet``; large).
    seed : int, default 0
        Root seed; all per-stage RNG derives from it. Same seed + same input + same layout
        reproduces identical output.
    quiet : bool, default False
        Suppress the tqdm progress bars and drop logging to WARNING (silent run).
    verbose : bool, default False
        Enable DEBUG-level logging (per-stage detail, full config). Ignored if ``quiet``.

    Returns
    -------
    pathlib.Path
        The ``output_directory/reads`` directory (pass it to ``pairplex.run(sequences=...)``).

    See Also
    --------
    simplex.score : score a PairPlex run against the truth produced here.
    simplex.config.SimplexConfig : the same knobs as a validated dataclass.
    """
    # quiet/verbose are run-behaviour, not part of the reproducible data-gen config
    kw = dict(locals())
    for k in ("quiet", "verbose"):
        kw.pop(k, None)
    kw["input_data"] = str(input_data)
    kw["output_directory"] = str(output_directory)
    cfg = SimplexConfig(**kw)
    configure_logging(quiet=quiet, verbose=verbose)
    out=Path(output_directory)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output dir {out} not empty; refusing to overwrite an experiment")
    out.mkdir(parents=True, exist_ok=True)
    logger.info("run starting: input=%s wells=%d seed=%d", cfg.input_data, cfg.wells, cfg.seed)
    logger.debug("config: %s", cfg.to_dict())
    bar = pbar(total=9, desc="simplex", quiet=quiet)

    cells=load_pairs(cfg.input_data, cfg.n_cells, cfg.seed); cfg.validate(actual_n_cells=cells.height)
    logger.info("[1/9] loaded %d cells", cells.height); bar.update(1)
    cells=assign_droplets_and_barcodes(cells, cfg.cells_per_droplet_mean, cfg.cells_per_droplet_overdispersion, cfg.chemistry, cfg.barcode_pool_size, cfg.seed)
    logger.info("[2/9] assigned %d droplets/barcodes (lambda=%.2f)", cells["droplet_id"].n_unique(), cfg.cells_per_droplet_mean); bar.update(1)
    cells=assign_wells(cells, cfg.wells, cfg.seed)
    logger.info("[3/9] assigned cells to %d wells", cfg.wells); bar.update(1)
    mols, chain_status=generate_molecules(cells, cfg.recovery_rate, cfg.molecules_per_chain_mean, cfg.release_rate, cfg.umi_length, cfg.rt_sub_rate, cfg.rt_indel_rate, cfg.seed)
    logger.info("[4/9] generated %d molecules (%d free)", mols.height, int(mols["is_free"].sum()) if mols.height else 0); bar.update(1)
    mols, reads=route_and_amplify(mols, cfg.wells, cfg.molecule_survival_rate, cfg.reads_per_molecule_mean, cfg.index_hop_rate, cfg.seed)
    logger.info("[5/9] amplified to %d reads (%d molecules survived)", reads.height, int(mols["survived"].sum()) if mols.height else 0); bar.update(1)
    reads=apply_sequencing_errors(reads, cfg.sequencing_sub_rate, cfg.sequencing_indel_rate, cfg.seed)
    logger.info("[6/9] applied sequencing errors (%d total)", int(reads["n_seq_errors"].sum()) if reads.height else 0); bar.update(1)
    comp=build_truth_components(cells, reads)
    tcells=build_truth_cells(cells, chain_status, mols, reads)
    tbar=build_truth_barcodes(cells, tcells, comp)
    logger.info("[7/9] built truth: %d components, %d (well,barcode) keys", comp.height, tbar.height); bar.update(1)
    built=build_merged(reads, cfg.tso, cfg.rc_fraction, cfg.variable_length, cfg.seed)
    logger.info("[8/9] assembled %d merged reads", built.height); bar.update(1)
    reads_paths=write_merged_fastq(built, out, show_progress=not quiet)
    write_truth(out, comp, tcells, tbar, reads if cfg.write_read_truth else None)
    logger.info("[9/9] wrote %d FASTQ file(s) + truth to %s", len(reads_paths), out); bar.update(1)
    bar.close()
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
