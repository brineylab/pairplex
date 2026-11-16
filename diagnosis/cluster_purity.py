"""Suspect-2 test, noise-calibrated.

Null: for UMIs with >=6 reads, split reads in two halves, POA-consensus each,
identity between the two halves = same-molecule consensus-vs-consensus identity.

Test: reproduce pairplex greedy_clustering at CTHR per barcode; inside each
cluster, build a POA consensus per UMI (UMIs with >=3 reads); two UMI consensuses
are "different molecules species" if their identity < null q0.01.
Also record whether read-majority vs UMI-majority species differ.
"""
import random
import sys
from collections import defaultdict

import edlib
import polars as pl
from spoa import poa

sys.path.insert(0, "/home/jovyan/work/Software/pairplex")
from pairplex.utils import greedy_clustering  # noqa: E402

PARSED, CTHR, N_BC = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
MIN_UMI_READS = 3
random.seed(0)


def hw(a, b):
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if not s:
        return 0.0
    return 1 - edlib.align(s, l, mode="HW", task="distance")["editDistance"] / len(s)


df = pl.read_parquet(PARSED, columns=["umi", "barcode", "seq_id", "sequence"])

# ---------- null ----------
g = (df.group_by("barcode", "umi").agg(pl.col("sequence"))
       .filter(pl.col("sequence").list.len() >= 6).sample(n=3000, seed=0))
null = []
for seqs in g["sequence"]:
    seqs = list(seqs)
    random.shuffle(seqs)
    h = len(seqs) // 2
    null.append(hw(poa(seqs[:h])[0], poa(seqs[h:])[0]))
null = pl.Series(null)
q01 = null.quantile(0.01)
print(f"NULL same-molecule consensus identity (n={null.len()}): q01={q01:.4f} q05={null.quantile(.05):.4f} median={null.median():.4f}")

# ---------- test ----------
counts = df.group_by("barcode").len().filter(pl.col("len") >= 3)
bcs = counts.sample(n=min(N_BC, counts.height), seed=0)["barcode"]
sub = df.filter(pl.col("barcode").is_in(bcs.implode()))
rows = []
for (bc,), gb in sub.group_by("barcode"):
    seqs = dict(zip(gb["seq_id"], gb["sequence"]))
    umi_of = dict(zip(gb["seq_id"], gb["umi"]))
    for cl in greedy_clustering(list(seqs.items()), identity_threshold=CTHR):
        if len(cl) < 3:
            continue
        by_umi = defaultdict(list)
        for sid in cl:
            by_umi[umi_of[sid]].append(seqs[sid])
        deep = {u: v for u, v in by_umi.items() if len(v) >= MIN_UMI_READS}
        if len(deep) < 2:
            continue
        cons = {u: poa(v)[0] for u, v in deep.items()}
        # greedy species grouping of UMI consensuses at the null q01
        species = []
        for u in sorted(cons, key=lambda u: len(deep[u]), reverse=True):
            for sp in species:
                if hw(cons[u], cons[sp[0]]) >= q01:
                    sp.append(u)
                    break
            else:
                species.append([u])
        reads_sp = [sum(len(deep[u]) for u in sp) for sp in species]
        umis_sp = [len(sp) for sp in species]
        idmin = hw(cons[species[0][0]], cons[species[1][0]]) if len(species) > 1 else None
        rows.append(dict(barcode=bc, reads=len(cl), umis=len(by_umi), deep_umis=len(deep),
                         n_species=len(species), id_top2=idmin,
                         disagree=len(species) > 1 and reads_sp.index(max(reads_sp)) != umis_sp.index(max(umis_sp)),
                         minority_read_share=1 - max(reads_sp) / sum(reads_sp)))
r = pl.DataFrame(rows)
m = r.filter(pl.col("n_species") >= 2)
print(f"well={PARSED.split('/')[-1]} cthr={CTHR} | clusters with >=2 deep UMIs: {r.height}")
print(f"  clusters merging >=2 distinct molecule species: {m.height} ({m.height / max(r.height, 1):.1%})")
if m.height:
    print(f"  identity between top-2 species: q10={m['id_top2'].quantile(.1):.3f} median={m['id_top2'].median():.3f} q90={m['id_top2'].quantile(.9):.3f}")
    print(f"  read-majority != UMI-majority species: {m['disagree'].sum()} ({m['disagree'].mean():.1%} of merged)")
    print(f"  minority species read share: median={m['minority_read_share'].median():.2f}")
r.write_parquet(f"purity_v2_{PARSED.split('/')[-1][:-8]}_cthr{CTHR}.parquet")
