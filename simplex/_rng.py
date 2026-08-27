"""Keyed RNG stream construction, used by every stochastic stage in the pipeline.

Each pipeline stage (droplets, molecules, routing, sequencing errors, ...) draws from
its own independent RNG stream rather than sharing one global generator or offsetting a
single seed. This makes stages independent of each other and of call order, giving the
v1 reproducibility guarantee: same seed + same input + same execution layout -> identical
output (see design doc S10). It does not (yet) guarantee invariance to row reordering or
chunk boundaries, since a stage still consumes one sequential RNG over the whole table.
"""
import hashlib
import numpy as np
def rng_for(seed, stage):
    """Return a fresh `numpy.random.Generator` seeded deterministically from `(seed, stage)`.

    The pair is hashed with blake2b (never `seed+offset` or Python `hash()`, both of which
    are unstable/predictable) into a `SeedSequence`. Same `(seed, stage)` always yields
    the same stream; different `stage` strings yield independent streams.
    """
    ent = int.from_bytes(hashlib.blake2b(f"{seed}|{stage}".encode(), digest_size=16).digest(), "big")
    return np.random.default_rng(np.random.SeedSequence(ent))
