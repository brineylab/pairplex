"""Two PairPlex internals on real reads: molecule splitting by clustering, and UMI errors.

(1) Splitting. PairPlex clusters reads per (barcode, well) with greedy global-alignment
    clustering (pairplex.utils.greedy_clustering, threshold 0.9). Reads of ONE molecule
    (barcode+UMI) should land in one cluster. A UMI whose reads fall in >= 2 clusters is split;
    if the clusters' representatives are the same chain (truncation-tolerant identity >= 0.94),
    the split creates a spurious extra contig of the same chain (-> "1H+2L"-type compositions).
(2) UMI errors. PairPlex counts unique UMI strings with no error correction. Within a barcode,
    UMI pairs at Hamming distance 1 are compared with the random expectation; a 1-read UMI that is
    a Hamming-1 neighbour of a UMI with many reads is the classic error signature. Such UMIs
    inflate support (min_cluster_umis) without adding molecules.

Usage: python umi_integrity.py PARSED.parquet [--n-barcodes 3000] [--cthr 0.9]
"""
import argparse
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import edlib  # noqa: E402
import polars as pl  # noqa: E402

from pairplex.utils import greedy_clustering  # noqa: E402


def hw(a, b):
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    return 1 - edlib.align(s, l, mode="HW", task="distance")["editDistance"] / len(s) if s else 0.0


def ham1(a, b):
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parsed")
    ap.add_argument("--n-barcodes", type=int, default=3000)
    ap.add_argument("--cthr", type=float, default=0.9)
    a = ap.parse_args()
    df = pl.read_parquet(a.parsed, columns=["barcode", "umi", "seq_id", "sequence"])
    bcs = df.group_by("barcode").len().filter(pl.col("len") >= 3).sample(n=a.n_barcodes, seed=0)["barcode"]
    sub = df.filter(pl.col("barcode").is_in(bcs.implode()))

    umis_multi = umis_split = umis_split_same = 0
    extra_contigs_same = barcodes_with_same_split = 0
    pairs_obs = pairs_exp = 0.0
    err_like = umis_total = 0
    for (bc,), g in sub.group_by("barcode"):
        seqs = dict(zip(g["seq_id"], g["sequence"]))
        umi_of = dict(zip(g["seq_id"], g["umi"]))
        clusters = greedy_clustering(list(seqs.items()), identity_threshold=a.cthr)
        cl_of = {sid: i for i, c in enumerate(clusters) for sid in c}
        rep = {i: max((seqs[s] for s in c), key=len) for i, c in enumerate(clusters)}
        reads_by_umi = defaultdict(list)
        for sid, u in umi_of.items():
            reads_by_umi[u].append(sid)
        same_split_here = False
        for u, sids in reads_by_umi.items():
            if len(sids) < 2:
                continue
            umis_multi += 1
            cls = {cl_of[s] for s in sids if s in cl_of}  # greedy_clustering drops empty reads
            if len(cls) > 1:  # noqa: SIM102
                umis_split += 1
                cl = sorted(cls)
                if any(hw(rep[x], rep[y]) >= 0.94 for x, y in combinations(cl, 2)):
                    umis_split_same += 1
                    extra_contigs_same += len(cl) - 1
                    same_split_here = True
        barcodes_with_same_split += same_split_here
        # UMI errors
        cnt = Counter(umi_of.values())
        us = list(cnt)
        umis_total += len(us)
        n = len(us)
        if n > 1 and n <= 400:
            L = len(us[0])
            pairs_exp += n * (n - 1) / 2 * (3 * L) / (4 ** L)
            for x, y in combinations(us, 2):
                if ham1(x, y):
                    pairs_obs += 1
                    small, big = (x, y) if cnt[x] <= cnt[y] else (y, x)
                    if cnt[small] == 1 and cnt[big] >= 5:
                        err_like += 1
    print(f"{Path(a.parsed).name}: barcodes {a.n_barcodes} | cthr {a.cthr}")
    print(f"(1) UMIs with >=2 reads: {umis_multi}; split across >=2 clusters: {umis_split} ({umis_split / max(umis_multi, 1):.1%}); "
          f"split into same-chain clusters: {umis_split_same} ({umis_split_same / max(umis_multi, 1):.1%})")
    print(f"    barcodes with a same-chain split (spurious extra contig): {barcodes_with_same_split} ({barcodes_with_same_split / a.n_barcodes:.1%}); "
          f"extra same-chain clusters: {extra_contigs_same}")
    print(f"(2) UMIs: {umis_total}; Hamming-1 UMI pairs within barcodes: observed {pairs_obs:.0f} vs random expectation {pairs_exp:.2f} "
          f"(x{pairs_obs / max(pairs_exp, 1e-9):.0f}); error-like (1-read UMI next to a >=5-read UMI): {err_like} "
          f"({err_like / max(umis_total, 1):.2%} of UMIs)")


if __name__ == "__main__":
    main()
