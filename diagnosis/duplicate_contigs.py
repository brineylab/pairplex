"""Same-CDR3 duplicate contigs per (well, barcode, chain class): real PairPlex output vs SimPlex.

A realism check for SimPlex: if simulated runs show many more same-CDR3 duplicate contigs than
real runs, the simulated per-well 1H+1L rule loses cells for a reason real data does not have.

Usage: python duplicate_contigs.py TAG1 "GLOB1" [TAG2 "GLOB2" ...]   (globs of *_paired/_unpaired.parquet)
"""
import glob
import os
import sys

import polars as pl

def stats(files, tag):
    rows = []
    for f in files:
        if os.path.getsize(f) < 100: continue
        d = pl.read_parquet(f)
        if 'locus' in d.columns:
            rows.append(d.select(pl.col('sequence_id').str.split('_').list.first().alias('bc'), pl.lit(f).alias('f'), 'locus', 'v_gene', 'cdr3_aa', pl.col('sequence').str.len_chars().alias('L')))
        elif 'locus:0' in d.columns:
            for i in (0, 1):
                rows.append(d.select(pl.col(f'sequence_id:{i}').str.split('_').list.first().alias('bc'), pl.lit(f).alias('f'), pl.col(f'locus:{i}').alias('locus'), pl.col(f'v_gene:{i}').alias('v_gene'), pl.col(f'cdr3_aa:{i}').alias('cdr3_aa'), pl.col(f'sequence:{i}').str.len_chars().alias('L')))
    a = pl.concat(rows).filter(pl.col('locus').is_in(['IGH', 'IGK', 'IGL']))
    a = a.with_columns(chain=pl.when(pl.col('locus') == 'IGH').then(pl.lit('H')).otherwise(pl.lit('L')))
    # duplicate = >=2 contigs, same barcode+well+chain class, same CDR3 aa
    dup = a.filter(pl.col('cdr3_aa').is_not_null()).group_by('f', 'bc', 'chain', 'cdr3_aa').agg(pl.len().alias('n'), (pl.col('L').min() / pl.col('L').max()).alias('r'))
    bcs = a.group_by('f', 'bc').agg((pl.col('chain') == 'H').sum().alias('nH'), (pl.col('chain') == 'L').sum().alias('nL'))
    both = bcs.filter((pl.col('nH') >= 1) & (pl.col('nL') >= 1))
    dbc = dup.filter(pl.col('n') >= 2).select('f', 'bc').unique()
    both_dup = both.join(dbc, on=['f', 'bc'], how='semi')
    not11 = both.filter(~((pl.col('nH') == 1) & (pl.col('nL') == 1)))
    # how many non-1H1L would become 1H1L if same-CDR3 duplicates were merged
    a2 = a.unique(['f', 'bc', 'chain', 'cdr3_aa'], keep='first') if True else a
    a2 = pl.concat([a.filter(pl.col('cdr3_aa').is_null()), a.filter(pl.col('cdr3_aa').is_not_null()).unique(['f', 'bc', 'chain', 'cdr3_aa'])])
    b2 = a2.group_by('f', 'bc').agg((pl.col('chain') == 'H').sum().alias('nH'), (pl.col('chain') == 'L').sum().alias('nL'))
    n11_before = both.filter((pl.col('nH') == 1) & (pl.col('nL') == 1)).height
    n11_after = b2.filter((pl.col('nH') == 1) & (pl.col('nL') == 1)).height
    print(f"{tag}: barcode-wells with H&L {both.height}; with a same-CDR3 duplicate contig {both_dup.height} ({both_dup.height/both.height:.1%}); "
          f"1H+1L {n11_before} -> {n11_after} after merging same-CDR3 contigs (x{n11_after/max(n11_before,1):.2f}); "
          f"dup length ratio median {dup.filter(pl.col('n')>=2)['r'].median():.2f}")
for tag, pat in [(t, p) for t, p in zip(sys.argv[1::2], sys.argv[2::2])]:
    stats(glob.glob(pat), tag)
