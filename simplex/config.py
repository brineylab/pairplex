import json
from dataclasses import asdict, dataclass
from pathlib import Path
from ._contract import BARCODE_LEN, UMI_LEN, TSO as _TSO

@dataclass
class SimplexConfig:
    input_data: str; output_directory: str
    n_cells: int | None = None; wells: int = 96
    cells_per_droplet_mean: float = 5.0; cells_per_droplet_sd: float = 2.0
    barcode_pool_size: int | None = None
    recovery_rate: float = 0.5; molecules_per_chain_mean: float = 10.0
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

    def to_dict(self): return asdict(self)
    def to_json(self,p): Path(p).write_text(json.dumps(self.to_dict(),indent=2))
    def estimated_reads(self,n):
        return int(n*2*self.recovery_rate*self.molecules_per_chain_mean*self.molecule_survival_rate*self.reads_per_molecule_mean)
    def validate(self, actual_n_cells=None, max_reads=3_000_000_000):
        for r in self._RATES:
            v=getattr(self,r)
            if not (0.0<=v<=1.0): raise ValueError(f"{r}={v} not in [0,1]")
        for r in self._POS:
            if getattr(self,r)<=0: raise ValueError(f"{r} must be > 0")
        if self.cells_per_droplet_sd<0: raise ValueError("cells_per_droplet_sd must be >= 0")
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
