import hashlib
import numpy as np
def rng_for(seed, stage):
    ent = int.from_bytes(hashlib.blake2b(f"{seed}|{stage}".encode(), digest_size=16).digest(), "big")
    return np.random.default_rng(np.random.SeedSequence(ent))
