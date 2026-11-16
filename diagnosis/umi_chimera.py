"""UMI-level chimera test on real PairPlex parsed reads.

Question: do (barcode, UMI) read families contain reads from >1 distinct V(D)J?
A molecule is one template, so all reads sharing barcode+UMI should be one
sequence up to sequencing/PCR noise (same-molecule read identity: median ~0.977,
q05 ~0.946 on donor01 VDJ_74; see noise_control.py). A minority species far below
that (identity < --species-id, default 0.85, i.e. a different V gene) inside one
UMI family is a PCR chimera / barcode-UMI swap: the 5' end (well index + barcode +
UMI) of one molecule joined to the V(D)J body of another.

Chain class of each read comes from the reverse complement of the PCR2
gene-specific primer present at the 3' end of merged reads
(UDAseq_protocol_250909 primer table). Reads that match no probe are 'unk'.

Usage:
    python umi_chimera.py PARSED.parquet OUT.parquet [--n-umis 30000] [--min-reads 4]
"""
import argparse
from collections import Counter

import edlib
import polars as pl

# RC of the gene-specific part of the PCR2 reverse primers (protocol 250909)
PROBES = {
    "IGK": ["GAACTGTGGCTGCACCATCT"],
    "IGL": ["GTCACTCTGTTCCCGCCCTC", "GTCACTCTGTTCCCACCCTC"],
    "IGHM": ["GAGTGCATCCGCCCCAAC"],
    "IGHG": ["CCACCAAGGGCCCATC"],
    "IGHA": ["CGACCAGCCCCAAGGTCTTC"],
    "IGHD": ["ATCCGGAGCCTTGGTGGG"],
}
LIGHT = {"IGK", "IGL"}


def chain_class(seq: str) -> str:
    for k, ps in PROBES.items():
        if any(p in seq for p in ps):
            return k
    return "unk"


def hw_identity(a: str, b: str) -> float:
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if not s:
        return 0.0
    return 1 - edlib.align(s, l, mode="HW", task="distance")["editDistance"] / len(s)


def species(reads: list[str], thr: float) -> list[list[str]]:
    """Greedy grouping seeded by the most frequent read sequences."""
    order = [s for s, _ in Counter(reads).most_common()]
    groups: list[list[str]] = []
    reps: list[str] = []
    counts = Counter(reads)
    for s in order:
        for i, r in enumerate(reps):
            if hw_identity(s, r) >= thr:
                groups[i].extend([s] * counts[s])
                break
        else:
            reps.append(s)
            groups.append([s] * counts[s])
    return sorted(groups, key=len, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parsed")
    ap.add_argument("out")
    ap.add_argument("--n-umis", type=int, default=30000)
    ap.add_argument("--min-reads", type=int, default=4)
    ap.add_argument("--min-len", type=int, default=150)
    ap.add_argument("--species-id", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pl.read_parquet(a.parsed, columns=["barcode", "umi", "sequence"])
    df = df.filter(pl.col("sequence").str.len_chars() >= a.min_len)
    fam = (df.group_by("barcode", "umi").agg(pl.col("sequence"))
             .filter(pl.col("sequence").list.len() >= a.min_reads))
    fam = fam.sample(n=min(a.n_umis, fam.height), seed=a.seed)

    rows = []
    for bc, umi, reads in fam.iter_rows():
        g = species(list(reads), a.species_id)
        maj = g[0]
        maj_cls = Counter(chain_class(s) for s in maj).most_common(1)[0][0]
        row = dict(barcode=bc, umi=umi, n_reads=len(reads), n_species=len(g),
                   major_class=maj_cls, major_reads=len(maj),
                   minor_class=None, minor_reads=0, id_major_minor=None)
        if len(g) > 1:
            mi = g[1]
            row.update(minor_class=Counter(chain_class(s) for s in mi).most_common(1)[0][0],
                       minor_reads=len(mi), id_major_minor=hw_identity(maj[0], mi[0]))
        rows.append(row)
    r = pl.DataFrame(rows)
    r.write_parquet(a.out)

    chim = r.filter(pl.col("minor_reads") >= 2)
    print(f"{a.parsed.split('/')[-1]}: UMI families (>= {a.min_reads} reads, len >= {a.min_len}): {r.height}")
    print(f"  families with a distinct minority V(D)J (id < {a.species_id}, >= 2 reads): "
          f"{chim.height} ({chim.height / r.height:.2%})")
    print(f"  ... any minority (>= 1 read):  {(r['n_species'] > 1).sum()} ({(r['n_species'] > 1).mean():.2%})")
    if chim.height:
        print("  minority read share: median", round(float((chim["minor_reads"] / chim["n_reads"]).median()), 3))
        print("  major -> minor chain class (known classes only):")
        known = chim.filter((pl.col("major_class") != "unk") & (pl.col("minor_class") != "unk"))
        print(known.group_by("major_class", "minor_class").len().sort("len", descending=True).head(12))
        same = known.filter(pl.col("major_class") == pl.col("minor_class")).height
        print(f"  same constant region (major==minor): {same}/{known.height} ({same / max(known.height, 1):.1%})")
        ml = known.filter(pl.col("minor_class").is_in(list(LIGHT))).height
        print(f"  minority is a light chain: {ml}/{known.height} ({ml / max(known.height, 1):.1%})")
    base = r.filter(pl.col("major_class") != "unk")["major_class"].value_counts().sort("count", descending=True)
    print("  background class mix of UMI families (major class):", dict(base.iter_rows()))


if __name__ == "__main__":
    main()
