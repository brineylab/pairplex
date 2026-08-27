"""SimPlex: synthetic raw-sequencing data generator + scorer for PairPlex.

Simulates the wet-lab mechanism (cells -> overloaded droplets/barcodes -> molecules
-> resident/free routing -> reads -> merged FASTQ) alongside mechanism-faithful ground
truth, then scores a PairPlex run against that truth. Public entry points are
``simplex.run(...)`` (generate a synthetic dataset + truth) and ``simplex.score(...)``
(compare a PairPlex output against the truth produced by ``run``). See ``SimplexConfig``
for the full set of generator knobs.
"""
from .config import SimplexConfig
from .run import run
from .scoring import score

__all__ = ["SimplexConfig", "run", "score"]
