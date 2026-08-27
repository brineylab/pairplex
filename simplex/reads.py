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
def build_merged(reads,tso,rc_fraction,variable_length,seed):
    """Assemble each read into the merged layout `barcode(16)+umi(10)+TSO+cDNA` (Phase 1-2
    `output_mode="merged"`), matching what `pairplex.parse_barcodes` expects.

    If `variable_length` is set, randomly truncates a small fraction off each end of the
    cDNA first (mimics variable read/insert length). A `rc_fraction` fraction of reads
    are then emitted reverse-complemented (`"rc"` RNG stream) to simulate the opposite
    sequencing orientation. `qual` is a placeholder string of the same length as
    `read_seq` (all `"I"`), not a real quality model.

    Empty-safe: if `reads` is empty, returns a typed empty frame with the
    `read_id, final_well, read_seq, qual` schema.
    """
    if reads.height==0:
        return pl.DataFrame(schema={"read_id":pl.Utf8,"final_well":pl.Int64,"read_seq":pl.Utf8,"qual":pl.Utf8})
    r=reads
    if variable_length:
        rng=rng_for(seed,"trunc"); lens=r["cdna"].str.len_chars().to_numpy()
        t5=rng.integers(0,np.maximum(1,lens//10)).astype(np.int64)
        nl=np.maximum(1,lens-t5-rng.integers(0,np.maximum(1,lens//10))).astype(np.int64)
        r=r.with_columns(pl.col("cdna").str.slice(pl.Series(t5),pl.Series(nl)).alias("cdna"))
    r=r.with_columns(pl.concat_str([pl.col("barcode"),pl.col("umi"),pl.lit(tso),pl.col("cdna")]).alias("_frag"))
    rc=pl.Series(rng_for(seed,"rc").random(r.height)<rc_fraction)
    r=r.with_columns(rc.alias("_rc")).with_columns(
        pl.when(pl.col("_rc")).then(revcomp_expr("_frag")).otherwise(pl.col("_frag")).alias("read_seq"))
    r=r.with_columns(pl.col("read_seq").str.replace_all(".","I").alias("qual"))
    return r.select(["read_id","final_well","read_seq","qual"])
