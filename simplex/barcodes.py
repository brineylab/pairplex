import gzip
from pathlib import Path
from pairplex.utils import get_whitelist_path
def load_barcodes(chemistry,n,rng):
    p=Path(get_whitelist_path(chemistry.lower()))
    op=gzip.open if str(p).endswith(".gz") else open
    with op(p,"rt") as f: wl=[l.strip() for l in f if l.strip()]
    if n>len(wl): raise ValueError(f"need {n}, whitelist has {len(wl)}")
    return [wl[i] for i in rng.choice(len(wl),size=n,replace=False)]
