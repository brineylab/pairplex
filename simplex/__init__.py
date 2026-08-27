"""SimPlex: synthetic raw-sequencing data generator + scorer for PairPlex.

Simulates the wet-lab mechanism (cells -> overloaded droplets/barcodes -> molecules
-> resident/free routing -> reads -> merged FASTQ) alongside mechanism-faithful ground
truth, then scores a PairPlex run against that truth. Public entry points are
``simplex.run(...)`` (generate a synthetic dataset + truth) and ``simplex.score(...)``
(compare a PairPlex output against the truth produced by ``run``). Imports are done
lazily inside the wrapper functions to keep ``import simplex`` cheap.
"""
from .config import SimplexConfig
__all__=["SimplexConfig","run","score"]
def run(*a,**k):
    """Lazy wrapper around :func:`simplex.run.run` — generates a synthetic dataset + truth."""
    from .run import run as r; return r(*a,**k)
def score(*a,**k):
    """Lazy wrapper around :func:`simplex.scoring.score` — scores a PairPlex output against truth."""
    from .scoring import score as s; return s(*a,**k)
