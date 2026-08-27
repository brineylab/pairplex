"""Low-level DNA string helpers: random sequence generation, reverse-complement, and
per-base mutation. Used by `barcodes`/`molecules` (UMI generation), `reads` (read-orientation
reverse-complementing), and both `molecules` (inherited RT error) and `reads` (independent
sequencing error) for the substitution/indel error model.
"""
import numpy as np, polars as pl
_A=np.array([65,67,71,84],np.uint8); _C=bytes.maketrans(b"ACGTN",b"TGCAN"); _B=np.array(list("ACGT"))
def random_dna(rng,k,length):
    """Generate `k` random DNA strings of fixed `length` (uppercase ACGT) using `rng`.

    Returns an empty object array when `k==0` (empty-safe).
    """
    if k==0: return np.array([],object)
    return _A[rng.integers(0,4,size=(k,length),dtype=np.uint8)].view(f"S{length}").reshape(k).astype(str)
def revcomp_str(s): return s.translate(_C)[::-1]
def revcomp_expr(col): return pl.col(col).str.reverse().str.replace_many(["A","C","G","T"],["T","G","C","A"])
def mutate_strings(seqs, sub_rate, indel_rate, rng):
    """Apply independent per-base substitutions and/or indels to each string in `seqs`.

    `sub_rate`/`indel_rate` are per-base Bernoulli probabilities. Substitutions always
    change the base to a different one; indels are modeled as a per-base coin flip split
    into a deletion half and an insertion half (`indel_rate/2` each). Returns
    `(mutated_seqs, per_seq_error_count)` where the count array is aligned with `seqs`.
    """
    out,cnt=[],np.zeros(len(seqs),np.int64)
    for i,s in enumerate(seqs):
        ch,n=list(s),0
        if sub_rate>0:
            for p in np.nonzero(rng.random(len(ch))<sub_rate)[0]:
                a=rng.choice(_B)
                while a==ch[p]: a=rng.choice(_B)
                ch[p]=str(a); n+=1
        if indel_rate>0:
            r=[]
            for c in ch:
                u=rng.random()
                if u<indel_rate/2: n+=1; continue
                r.append(c)
                if u>1-indel_rate/2: r.append(str(rng.choice(_B))); n+=1
            ch=r
        out.append("".join(ch)); cnt[i]=n
    return out,cnt
