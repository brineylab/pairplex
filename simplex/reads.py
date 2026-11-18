"""Stage 4 of the pipeline: independent per-read sequencing error, then assembly into
merged FASTQ read sequences.

Consumes the `reads` frame from `routing.py`, mutates each read's cDNA independently
(sequencing error, as opposed to the molecule-level inherited RT error applied earlier in
`molecules.py`), then builds the final `barcode+umi+TSO+cDNA` merged read layout consumed
by `io.write_merged_fastq`. Sits between routing and truth/output in the cells -> molecules
-> routing -> reads -> truth -> scoring pipeline.
"""
import numpy as np, polars as pl
from ._dna import mutate_strings, revcomp_expr
from ._rng import rng_for
def apply_sequencing_errors(reads,sub_rate,indel_rate,seed):
    """Apply independent per-read substitution/indel error to each read's cDNA.

    Unlike RT error (stamped once per molecule and inherited by the whole family),
    sequencing error is drawn fresh per read via the `"seqerr"` RNG stream. No-op
    (returns `reads` unchanged) if `reads` is empty or both rates are 0. Accumulates
    into the existing `n_seq_errors` column rather than overwriting it.
    """
    if reads.height==0 or (sub_rate==0 and indel_rate==0): return reads
    cdna,ne=mutate_strings(list(reads["cdna"]),sub_rate,indel_rate,rng_for(seed,"seqerr"))
    return reads.with_columns([pl.Series("cdna",cdna),(pl.col("n_seq_errors")+pl.Series(ne)).alias("n_seq_errors")])
def build_merged(reads,tso,rc_fraction,variable_length,seed,truncation="fragment_5p",fragment_rate=0.1,fragment_rate_heavy=None,fragment_keep=(0.3,0.9),
                 barcode_sub_rate=0.0,tso_sub_rate=0.0,tso_g_run_probs=None,tso_indel_rate=0.0):
    """Assemble each read into the merged layout `barcode(16)+umi(10)+TSO+cDNA` (Phase 1-2
    `output_mode="merged"`), matching what `pairplex.parse_barcodes` expects.

    If `variable_length` is set, truncates the cDNA first. Models (all per molecule = barcode+UMI,
    shared by every read of the molecule, except the legacy one):

    - `"fragment_5p"` (default): intact amplicons run TSO -> gene-specific primer, so they all have
      full length; a fraction `fragment_rate` of molecules (`fragment_rate_heavy` for IGH, if given)
      come from fragmented RNA: template switching happened at the break, so the read keeps a random
      3' suffix (a fraction drawn uniformly in `fragment_keep`, default 30-90% of the length) and
      loses the 5' part (as real truncated heavy chains do). Input chains are V(D)J only (no
      constant region), so a long kept fraction still carries CDR3 + part of V and is annotated as
      an extra contig; real fragments are mostly C-region +/- J and fail annotation (PRINCEPS_2025
      uses 0.05-0.35).
    - `"molecule_3p"`: one 3' cut (0-30%) per molecule, 5' kept (other chemistries / ablations).
    - `"per_read_both"`: legacy v1 (independent 0-10% cuts at both ends of every read); splinters
      single molecules under PairPlex's global-alignment clustering. Reproduction only. A `rc_fraction` fraction of reads
    are then emitted reverse-complemented (`"rc"` RNG stream) to simulate the opposite
    sequencing orientation. `qual` is a placeholder string of the same length as
    `read_seq` (all `"I"`), not a real quality model.

    v2d read-structure knobs (all default off = v2c layout): `barcode_sub_rate` per-base substitution in each read's
    16-nt barcode (the parser corrects Hamming-1 against the whitelist and drops the rest); `tso_sub_rate`
    per-base substitution in the 10-nt TSO body (real reads: only ~18% carry an exact TTTCTTATAT); `tso_g_run_probs`
    probabilities of 0,1,2,... G after the TSO body instead of the fixed GGG; `tso_indel_rate` share of reads with a
    1-3 nt insertion or deletion in the TSO body (shifts the cDNA start; real merged reads: ~20% of TSO bodies are
    >=4 mismatches away from TTTCTTATAT, 60% exact, G-run after an exact TSO mostly 4-5).

    Empty-safe: if `reads` is empty, returns a typed empty frame with the
    `read_id, final_well, read_seq, qual` schema.
    """
    if reads.height==0:
        return pl.DataFrame(schema={"read_id":pl.Utf8,"final_well":pl.Int64,"read_seq":pl.Utf8,"qual":pl.Utf8})
    r=reads
    if variable_length and truncation=="per_read_both":
        rng=rng_for(seed,"trunc"); lens=r["cdna"].str.len_chars().to_numpy()
        t5=rng.integers(0,np.maximum(1,lens//10)).astype(np.int64)
        nl=np.maximum(1,lens-t5-rng.integers(0,np.maximum(1,lens//10))).astype(np.int64)
        r=r.with_columns(pl.col("cdna").str.slice(pl.Series(t5),pl.Series(nl)).alias("cdna"))
    elif variable_length and truncation=="fragment_5p":
        rng=rng_for(seed,"trunc")
        mk=r.select("barcode","umi","locus" if "locus" in r.columns else pl.lit("NA").alias("locus")).unique(subset=["barcode","umi"],maintain_order=True)
        rate=np.where(mk["locus"].to_numpy()=="IGH", fragment_rate if fragment_rate_heavy is None else fragment_rate_heavy, fragment_rate)
        frag=rng.random(mk.height)<rate
        lo,hi=fragment_keep
        keep=np.where(frag,lo+(hi-lo)*rng.random(mk.height),1.0)
        mk=mk.select("barcode","umi").with_columns(pl.Series("_keep",keep))
        r=(r.join(mk,on=["barcode","umi"],how="left")
             .with_columns(pl.max_horizontal(pl.lit(1),(pl.col("cdna").str.len_chars()*pl.col("_keep")).floor().cast(pl.Int64)).alias("_nl"))
             .with_columns(pl.col("cdna").str.slice(pl.col("cdna").str.len_chars()-pl.col("_nl"),pl.col("_nl")).alias("cdna")).drop("_keep","_nl"))
    elif variable_length and truncation=="molecule_3p":
        rng=rng_for(seed,"trunc")
        mk=r.select("barcode","umi").unique(maintain_order=True)
        mk=mk.with_columns(pl.Series("_keep",1.0-0.3*rng.random(mk.height)))
        r=(r.join(mk,on=["barcode","umi"],how="left")
             .with_columns(pl.max_horizontal(pl.lit(1),(pl.col("cdna").str.len_chars()*pl.col("_keep")).floor().cast(pl.Int64)).alias("_nl"))
             .with_columns(pl.col("cdna").str.slice(0,pl.col("_nl")).alias("cdna")).drop("_keep","_nl"))
    elif variable_length:
        raise ValueError(f"unknown truncation mode {truncation!r}")
    if barcode_sub_rate>0 or tso_sub_rate>0 or tso_g_run_probs or tso_indel_rate>0:
        srng=rng_for(seed,"structure"); k=r.height; alt=np.array(list("ACGT"))
        def _mut(strings, rate):
            if rate<=0: return strings
            A=np.array([list(x) for x in strings]); hit=srng.random(A.shape)<rate
            if hit.any():
                s2=alt[srng.integers(0,4,size=int(hit.sum()))]; same=s2==A[hit]
                s2[same]=np.array([{"A":"C","C":"G","G":"T","T":"A"}.get(c,"A") for c in s2[same]]); A[hit]=s2
            return ["".join(x) for x in A]
        bcs=_mut(r["barcode"].to_list(),barcode_sub_rate)
        body=tso.rstrip("G")
        bodies=_mut([body]*k,tso_sub_rate)
        if tso_indel_rate>0:
            sel=np.flatnonzero(srng.random(k)<tso_indel_rate); bl=list(bodies)
            for i in sel:
                b=bl[i]; n=int(srng.integers(1,4)); p=int(srng.integers(0,len(b)))
                bl[i]=(b[:p]+"".join(alt[srng.integers(0,4,size=n)])+b[p:]) if srng.random()<0.5 else (b[:p]+b[p+n:])
            bodies=bl
        if tso_g_run_probs:
            pg=np.array(tso_g_run_probs,float); pg=pg/pg.sum(); g=srng.choice(len(pg),size=k,p=pg)
        else:
            g=np.full(k,len(tso)-len(body))
        tsos=[b+"G"*int(n) for b,n in zip(bodies,g)]
        r=r.with_columns(pl.Series("_bc",bcs),pl.Series("_tso",tsos)).with_columns(pl.concat_str([pl.col("_bc"),pl.col("umi"),pl.col("_tso"),pl.col("cdna")]).alias("_frag")).drop("_bc","_tso")
    else:
        r=r.with_columns(pl.concat_str([pl.col("barcode"),pl.col("umi"),pl.lit(tso),pl.col("cdna")]).alias("_frag"))
    rc=pl.Series(rng_for(seed,"rc").random(r.height)<rc_fraction)
    r=r.with_columns(rc.alias("_rc")).with_columns(
        pl.when(pl.col("_rc")).then(revcomp_expr("_frag")).otherwise(pl.col("_frag")).alias("read_seq"))
    r=r.with_columns(pl.col("read_seq").str.replace_all(".","I").alias("qual"))
    return r.select(["read_id","final_well","read_seq","qual"])
