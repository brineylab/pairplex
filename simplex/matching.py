"""Scorer core: sequence-to-source matching and pairing-status resolution.

Used by `scoring.score` to decide, for one observed (heavy, light) pair, which truth
`source_pair_id`(s) each chain could have come from, and whether that constitutes a
correct, mispaired, ambiguous, or unmatchable pairing. Matching is locus-restricted and
edit-distance-bounded (never trusts PairPlex's own chain/source annotation, to avoid
circularity), per the frozen scoring contract (design doc S6).
"""
import edlib
from typing import NamedTuple

class OrientationResult(NamedTuple):
    """Result of resolving one (heavy-candidates, light-candidates) orientation.

    `pairing_status`: correct | mispaired | unmatchable (see `resolve`).
    `source_resolution`: unique | ambiguous | none.
    `resolved_source`: the single agreed `source_pair_id` if `source_resolution=="unique"`, else None.
    `valid_assignments`: `set[(heavy_source, light_source)]` consistent with the observation
    (used downstream to classify `origin_status`).
    """
    pairing_status: str
    source_resolution: str
    resolved_source: object
    valid_assignments: set   # set[(heavy_source, light_source)] consistent with the observation

def seq_match(a,b,max_frac=0.06,min_len=50):
    """True if strings `a`/`b` align (edlib infix/`HW` mode) within `max_frac` edit
    distance relative to the shorter string's length, and the shorter string is at
    least `min_len` (guards against short sequences matching many unrelated sources).
    False if either is falsy.
    """
    if not a or not b: return False
    short,long=(a,b) if len(a)<=len(b) else (b,a)
    if len(short)<min_len: return False
    r=edlib.align(short,long,mode="HW",task="distance")
    return 0<=r["editDistance"]<=max_frac*len(short)
def candidates(seq,locus,key_entry,max_frac=0.06,min_len=50):
    """Return the set of truth `source_pair_id`s at this key whose `locus` sequence
    matches `seq` (exact match or `seq_match` within tolerance).

    `key_entry` is one `(final_well, barcode)` entry of the truth index: `{locus: {full_seq:
    {source_pair_id, ...}}}`. Matching is restricted to `locus` and to sources actually
    present at this key (never a global search). Returns an empty set if `seq` or
    `key_entry` is falsy/`None`.
    """
    if not seq or key_entry is None: return set()
    hits=set()
    for full,sources in key_entry.get(locus,{}).items():
        if seq==full or seq_match(seq,full,max_frac,min_len): hits|=sources
    return hits
def resolve(h,l):
    """Resolve pairing status from a heavy-chain candidate source set `h` and light-chain
    candidate source set `l`.

    Rules (frozen scoring contract): empty `h` or `l` -> `unmatchable`. Non-empty but
    disjoint sets -> `mispaired` (a cross-source pair is impossible), even if one side
    is individually ambiguous — `valid_assignments` is then every cross combination
    `(hh, ll)`. A non-empty intersection is always `correct` pairing; if the
    intersection has exactly one source, `source_resolution="unique"` with that
    resolved source; if it has more than one, `source_resolution="ambiguous"` and
    `resolved_source` is `None`. `valid_assignments` for a correct pairing is restricted
    to same-source pairs (the intersection), since only same-source explains "correct".
    """
    if not h or not l:
        return OrientationResult("unmatchable","none",None,set())
    inter=h&l
    if not inter:                                   # non-empty sets, empty intersection ⇒ mispaired
        return OrientationResult("mispaired","none",None,{(hh,ll) for hh in h for ll in l})
    va={(s,s) for s in inter}                        # valid CORRECT explanations are same-source only
    if len(inter)==1:
        return OrientationResult("correct","unique",next(iter(inter)),va)
    return OrientationResult("correct","ambiguous",None,va)
