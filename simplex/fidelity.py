"""Sequence fidelity of PairPlex contigs against SimPlex truth (v2d).

Pairing correctness (`simplex.score`) accepts a chain within 6% edit distance of a source, so it cannot see two failure
modes that matter for repertoire analysis:

1. **Invented or erased mutations.** A contig that differs from every true sequence present at its (well, barcode)
   carries errors the consensus did not remove (RT/PCR errors are inherited by the whole read family) or mixes
   molecules of different cells.
2. **Over-collapsing clonal siblings.** Members of one expanded lineage share V, J and CDR3 and differ only by
   somatic hypermutation. If two siblings share a (well, barcode), a pipeline that clusters or merges too
   aggressively returns one contig for both, erasing real diversity. Sequencing/PCR errors must be collapsed;
   siblings must not.

`fidelity(output_dir, truth_dir)` returns a dict with:

- `contigs`: productive IGH/IGK/IGL contigs scored; `exact_share`: share whose sequence occurs exactly (as an infix)
  in a true sequence of the same locus class at its key; `le1_share` (<= 1 edit); `edits_per_100nt_mean`;
  the same restricted to keys with a single true source of that locus class (`single_source_*`: pure error rate).
- `sibling_sequences`: true sequences at keys holding >= 2 members of one clone (same `clone_id`, distinct
  sequences, each >= `min_reads` reads there); `sibling_recovered_share`: share of those that are the nearest
  true sequence of at least one contig within `max_frac` (separately resolved);
  `sibling_keys_collapsed_share`: keys where fewer siblings were recovered than present.

Needs `clone_id` in `truth_cells` (clonal sampling from an input with a `clone_id` column) for the sibling part;
otherwise only contig fidelity is reported.
"""
import re
from pathlib import Path

import edlib
import polars as pl

_LIGHT = ("IGK", "IGL")


def _cls(locus):
    return "H" if locus == "IGH" else ("L" if locus in _LIGHT else None)


def _dist(a, b):
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return edlib.align(short, long, mode="HW", task="distance")["editDistance"], len(short)


def fidelity(output_dir, truth_dir, min_reads=3, max_frac=0.02):
    out, truth = Path(output_dir), Path(truth_dir)
    comp = pl.read_parquet(truth / "truth_components.parquet").filter(pl.col("chain") >= 0)
    tc = pl.read_parquet(truth / "truth_cells.parquet")
    has_clone = "clone_id" in tc.columns
    src_clone = dict(zip(tc["source_pair_id"].to_list(), tc["clone_id"].to_list())) if has_clone else {}
    agg = (comp.with_columns(pl.col("locus").map_elements(_cls, return_dtype=pl.Utf8).alias("lc"))
               .group_by("final_well", "barcode", "lc", "sequence", "source_pair_id").agg(pl.col("n_reads").sum()))
    idx = {}
    for w, b, lc, seq, src, nr in agg.iter_rows():
        if seq is None or lc is None:
            continue
        idx.setdefault((int(w), b, lc), []).append((seq, src, int(nr)))
    contigs = []
    for f in sorted((out / "annotated").glob("*_unpaired.parquet")):
        sch = pl.read_parquet_schema(f)
        if "sequence" not in sch or "locus" not in sch:
            continue
        m = re.search(r"well(\d+)", f.name) or re.search(r"(\d+)", f.name)
        w = int(m.group(1))
        d = pl.read_parquet(f, columns=["sequence_id", "sequence", "locus", "productive"]).filter(pl.col("productive"))
        for sid, seq, loc, _ in d.iter_rows():
            lc = _cls(loc)
            if lc is None or not seq:
                continue
            contigs.append((w, sid.split("_")[0], lc, seq))
    rows, nearest_by_key = [], {}
    for w, b, lc, seq in contigs:
        cands = idx.get((w, b, lc))
        if not cands:
            continue
        best = None
        for tseq, src, nr in cands:
            e, n = _dist(seq, tseq)
            if best is None or e < best[0]:
                best = (e, n, tseq)
        rows.append({"edits": best[0], "len": best[1], "single": len({c[0] for c in cands}) == 1})
        if best[0] <= max_frac * best[1]:
            nearest_by_key.setdefault((w, b, lc), set()).add(best[2])
    res = {}
    if rows:
        r = pl.DataFrame(rows).with_columns((100 * pl.col("edits") / pl.col("len")).alias("e100"))
        s = r.filter(pl.col("single"))
        res.update(contigs=r.height, exact_share=float((r["edits"] == 0).mean()), le1_share=float((r["edits"] <= 1).mean()),
                   edits_per_100nt_mean=float(r["e100"].mean()), single_source_contigs=s.height,
                   single_source_exact_share=float((s["edits"] == 0).mean()) if s.height else None,
                   single_source_edits_per_100nt_mean=float(s["e100"].mean()) if s.height else None)
    if has_clone:
        n_sib = n_rec = n_keys = n_coll = 0
        for key, cands in idx.items():
            by_clone = {}
            for tseq, src, nr in cands:
                cl = src_clone.get(src)
                if cl is not None and nr >= min_reads:
                    by_clone.setdefault(cl, set()).add(tseq)
            for cl, seqs in by_clone.items():
                if len(seqs) < 2:
                    continue
                rec = len(seqs & nearest_by_key.get(key, set()))
                n_sib += len(seqs); n_rec += rec; n_keys += 1; n_coll += rec < len(seqs)
        res.update(sibling_keys=n_keys, sibling_sequences=n_sib,
                   sibling_recovered_share=(n_rec / n_sib) if n_sib else None,
                   sibling_keys_collapsed_share=(n_coll / n_keys) if n_keys else None)
    return res


__all__ = ["fidelity"]
