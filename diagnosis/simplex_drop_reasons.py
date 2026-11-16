"""Why per-well PairPlex drops simulated cells: for every cell whose two chains survived, classify
its (resident well, barcode) as paired / collision / true extra chain (leakage or ambient) / chain
lost / spurious extra contig made by PairPlex.

Usage: python simplex_drop_reasons.py SCENARIO_DIR   (contains sim/truth and pp/annotated)
"""
import polars as pl, glob, re, sys, os
os.chdir(sys.argv[1])
cells = pl.read_parquet('sim/truth/truth_cells.parquet').filter(pl.col('survived_0') & pl.col('survived_1'))
tc = pl.read_parquet('sim/truth/truth_components.parquet')
rows = []
for f in glob.glob('pp/annotated/*_unpaired.parquet') + glob.glob('pp/annotated/*_paired.parquet'):
    w = int(re.search(r'well(\d+)', f).group(1))
    if os.path.getsize(f) < 100: continue
    d = pl.read_parquet(f)
    if 'locus' in d.columns:
        rows.append(d.select(pl.col('sequence_id').str.split('_').list.first().alias('barcode'), 'locus').with_columns(well=pl.lit(w), src=pl.lit('unpaired')))
    elif 'locus:0' in d.columns:
        for i in (0, 1):
            rows.append(d.select(pl.col(f'sequence_id:{i}').str.split('_').list.first().alias('barcode'), pl.col(f'locus:{i}').alias('locus')).with_columns(well=pl.lit(w), src=pl.lit('paired')))
a = pl.concat(rows)
comp = a.group_by('well', 'barcode').agg((pl.col('locus') == 'IGH').sum().alias('nH'), pl.col('locus').is_in(['IGK', 'IGL']).sum().alias('nL'), (pl.col('src') == 'paired').any().alias('paired'))
c = cells.select('cell_id', 'barcode', pl.col('resident_well').alias('well')).join(comp, on=['well', 'barcode'], how='left').fill_null(0)
ncell = cells.group_by('barcode', pl.col('resident_well').alias('well')).len().rename({'len': 'cells_here'})
c = c.join(ncell, on=['barcode', 'well'])
# truth: distinct true chains (components) present in that well+barcode
tcn = tc.group_by(pl.col('final_well').alias('well'), 'barcode').agg((pl.col('locus') == 'IGH').sum().alias('tH'), pl.col('locus').is_in(['IGK', 'IGL']).sum().alias('tL'))
c = c.join(tcn, on=['well', 'barcode'], how='left').fill_null(0)
# is each of the cell's own chains physically present (any read) in its resident well?
own = tc.filter(pl.col('final_well').is_not_null()).select('origin_cell_id', 'chain', pl.col('final_well').alias('well')).unique()
for ch in (0, 1):
    hit = own.filter(pl.col('chain') == ch).select(pl.col('origin_cell_id').alias('cell_id'), 'well').with_columns(pl.lit(True).alias(f'own{ch}'))
    c = c.join(hit, on=['cell_id', 'well'], how='left').with_columns(pl.col(f'own{ch}').fill_null(False))
def cat(r):
    if r['paired']: return 'paired'
    if r['cells_here'] > 1: return 'collision (>=2 cells same barcode+well)'
    if not (r['own0'] and r['own1']): return 'own chain absent from home well (leaked out)'
    if r['nH'] == 0 or r['nL'] == 0: return 'chain lost in PairPlex (annotation/consensus)'
    if r['tH'] > 1 or r['tL'] > 1: return 'true extra chain in well (leak/ambient)'
    return 'spurious extra contig (PairPlex)'
c = c.with_columns(pl.struct(['paired', 'cells_here', 'own0', 'own1', 'tH', 'tL', 'nH', 'nL']).map_elements(cat, return_dtype=pl.Utf8).alias('reason'))
print(sys.argv[1], 'cells survived both:', c.height)
print(c.group_by('reason').len().with_columns((pl.col('len') / c.height).round(3).alias('frac')).sort('len', descending=True))
print(c.filter(pl.col('reason').str.starts_with('spurious')).group_by('nH', 'nL').len().sort('len', descending=True).head(6))
