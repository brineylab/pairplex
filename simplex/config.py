"""Configuration and validation for a SimPlex run.

`SimplexConfig` is the single knob surface for `simplex.run`: it holds every parameter
of the generator (sampling, droplet overloading, capture/depth, contamination, error
rates, fixed read structure, and bookkeeping) and enforces the invariants from the
frozen design spec via `validate()`.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from ._contract import BARCODE_LEN, UMI_LEN, TSO as _TSO

@dataclass
class SimplexConfig:
    """All knobs for one SimPlex generator run, grouped by what they control:

    - **Sampling**: `n_cells` (subsample this many source pairs; `None` = use all).
    - **Overloading** (cells sharing a 10X droplet barcode): `cells_per_droplet_mean` is the
      physical loading rate lambda (cells per GEM); cells are randomly loaded into
      `round(n_cells / lambda)` droplets, giving **Poisson** occupancy (the encapsulation
      process). `cells_per_droplet_overdispersion` (>= 0, default 0 = pure Poisson) makes
      droplet capture propensities vary (Dirichlet weights, concentration `1/overdispersion`)
      to model cell clumping / uneven GEMs (Negative-Binomial-like occupancy). `barcode_pool_size`
      (`None` = one unique barcode per droplet; an int samples droplet barcodes from a smaller
      pool, enabling controlled barcode reuse/collision across droplets).

    **Defaults are an illustrative baseline, not a claim about any assay** — calibrate the
    uncertain ones (especially `release_rate`, `molecules_per_chain_mean`, `recovery_rate`,
    `cells_per_droplet_mean`) to your real `metadata/*.csv` and **sweep ranges**.
    - **Capture & depth**: `recovery_rate` (per-cell, per-chain capture probability),
      `molecules_per_chain_mean` (RT molecule count per captured chain),
      `molecule_survival_rate` (Bernoulli survival applied *before* amplification —
      not read thinning), `reads_per_molecule_mean` (read-family depth per surviving
      molecule).
    - **Contamination**: `release_rate` (fraction of molecules that become "free" and
      are redistributed to a random well, keeping barcode+UMI — this moves molecules,
      it does not add them) and `index_hop_rate` (per-read probability a read's
      `final_well` differs from its `amplification_well`).
    - **Errors**: `rt_sub_rate`/`rt_indel_rate` are stamped once per molecule and
      inherited by its whole read family (RT error); `sequencing_sub_rate`/
      `sequencing_indel_rate` are applied independently per read (sequencing error).
    - **Read structure** (fixed/validated, not free knobs): `barcode_length`/
      `umi_length`/`tso` must match the values `pairplex.parse_barcodes` assumes;
      `chemistry` selects the 10X barcode whitelist; `output_mode` is `"merged"`-only
      in Phase 1-2; `rc_fraction` is the fraction of reads emitted reverse-complemented;
      `variable_length` truncates cDNA ends to mimic variable read/insert length.
    - **Bookkeeping**: `write_read_truth` (also persist a per-read truth parquet —
      expensive, off by default), `seed` (root seed for all per-stage RNG streams).
    """
    input_data: str; output_directory: str
    n_cells: int | None = None; wells: int = 96
    cells_per_droplet_mean: float = 2.0; cells_per_droplet_overdispersion: float = 0.0  # lambda + optional clumping
    barcode_pool_size: int | None = None
    recovery_rate: float = 0.5; molecules_per_chain_mean: float = 20.0
    release_rate: float = 0.02; molecule_survival_rate: float = 0.8; reads_per_molecule_mean: float = 5.0
    rt_sub_rate: float = 0.0; rt_indel_rate: float = 0.0
    sequencing_sub_rate: float = 0.001; sequencing_indel_rate: float = 0.0
    index_hop_rate: float = 0.001
    barcode_length: int = BARCODE_LEN; umi_length: int = UMI_LEN; tso: str = "TTTCTTATATGGG"; chemistry: str = "v2"
    output_mode: str = "merged"; rc_fraction: float = 0.0
    variable_length: bool = True; write_read_truth: bool = False; seed: int = 0

    _RATES=("recovery_rate","release_rate","molecule_survival_rate","rt_sub_rate","rt_indel_rate",
            "sequencing_sub_rate","sequencing_indel_rate","index_hop_rate","rc_fraction")
    _POS=("wells","cells_per_droplet_mean","molecules_per_chain_mean","reads_per_molecule_mean")

    def to_dict(self):
        """Return the config as a plain dict (via `dataclasses.asdict`)."""
        return asdict(self)
    def to_json(self,p):
        """Write the config as indented JSON to path `p`."""
        Path(p).write_text(json.dumps(self.to_dict(),indent=2))
    def estimated_reads(self,n):
        """Rough upper-bound read-count estimate for `n` cells (2 chains x recovery x
        molecules/chain x survival x depth). Deliberately omits `release_rate`, since
        release redistributes existing molecules rather than adding new ones.
        """
        return int(n*2*self.recovery_rate*self.molecules_per_chain_mean*self.molecule_survival_rate*self.reads_per_molecule_mean)
    def validate(self, actual_n_cells=None, max_reads=3_000_000_000):
        """Enforce config invariants; raises `ValueError` on the first violation, else returns `self`.

        Checks: all rate-like fields in [0,1]; all positive-only fields > 0;
        `cells_per_droplet_overdispersion` >= 0; `barcode_pool_size`/`n_cells` positive or `None`;
        `output_mode` is `"merged"` (Phase 1-2 only supports merged output);
        `barcode_length`/`umi_length`/`tso` match the fixed values `pairplex.parse_barcodes`
        assumes; `index_hop_rate` must be 0 when `wells==1` (a hop can't change well);
        and the estimated read count (using `actual_n_cells` if given, else `n_cells`)
        must not exceed `max_reads` (an OOM/runaway-run guard).
        """
        for r in self._RATES:
            v=getattr(self,r)
            if not (0.0<=v<=1.0): raise ValueError(f"{r}={v} not in [0,1]")
        for r in self._POS:
            if getattr(self,r)<=0: raise ValueError(f"{r} must be > 0")
        if self.cells_per_droplet_overdispersion<0: raise ValueError("cells_per_droplet_overdispersion must be >= 0")
        if self.barcode_pool_size is not None and self.barcode_pool_size<=0: raise ValueError("barcode_pool_size must be > 0 or None")
        if self.n_cells is not None and self.n_cells<=0: raise ValueError("n_cells must be > 0 or None")
        if self.output_mode!="merged": raise ValueError("Phase 1-2: output_mode='merged' only")
        if self.barcode_length!=BARCODE_LEN or self.umi_length!=UMI_LEN:
            raise ValueError(f"Phase 1-2 fixes barcode_length={BARCODE_LEN}, umi_length={UMI_LEN}")
        if self.tso!=_TSO: raise ValueError(f"Phase 1-2 fixes tso={_TSO!r} (parser assumes it)")
        if self.wells==1 and self.index_hop_rate!=0: raise ValueError("index_hop_rate must be 0 when wells==1")
        n=actual_n_cells if actual_n_cells is not None else self.n_cells
        if n and self.estimated_reads(n)>max_reads: raise ValueError(f"est reads {self.estimated_reads(n)}>budget {max_reads}")
        return self
