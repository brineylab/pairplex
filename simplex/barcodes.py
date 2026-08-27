"""10X barcode whitelist loading, shared by droplet-barcode assignment in `cells.py`.

Reuses PairPlex's own chemistry whitelists so simulated barcodes are drawn from the same
universe a real PairPlex run would see.
"""
import gzip
from pathlib import Path
from pairplex.utils import get_whitelist_path
def load_barcodes(chemistry,n,rng):
    """Sample `n` distinct barcodes without replacement from the 10X `chemistry` whitelist.

    Raises `ValueError` if the whitelist has fewer than `n` entries. Returns a plain list
    of barcode strings (order determined by `rng`).
    """
    p=Path(get_whitelist_path(chemistry.lower()))
    op=gzip.open if str(p).endswith(".gz") else open
    with op(p,"rt") as f: wl=[l.strip() for l in f if l.strip()]
    if n>len(wl): raise ValueError(f"need {n}, whitelist has {len(wl)}")
    return [wl[i] for i in rng.choice(len(wl),size=n,replace=False)]
