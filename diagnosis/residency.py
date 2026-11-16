"""Do barcoded cDNA molecules stay inside their fixed cell? (cross-well residency)

UDA-seq assumes cells stay intact after the emulsion is broken, so all molecules of
one cell's chain (same droplet barcode, same V(D)J) should be found in ONE well:
the well that cell was dispensed to. Molecules of that chain found in other wells
of the same reaction left the cell (leakage of barcoded cDNA into the aqueous
phase, which UDA-seq dispenses into every well without a wash), or were moved by
index hopping / cross-well PCR events.

Method: PairPlex per-well metadata (all contigs, pass or not) for one reaction; a
hash-sampled subset of droplet barcodes; within each barcode, contig consensus
sequences from all wells are grouped into "chain species" (truncation-tolerant
identity >= --species-id). For each species: total UMIs, number of wells, UMIs in
the modal well. Leakage estimate = fraction of UMIs outside the modal well, by
species size. Under "each molecule independently leaves its cell with prob f and
lands in a uniformly random well of W", the expected outside-modal fraction for
large species is f * (1 - 1/W).

Usage:
    python residency.py METADATA_DIR OUT.parquet --wells 1-96 [--sample-mod 200]
"""
import argparse
import glob
import os
import re

import edlib
import polars as pl


def hw_identity(a: str, b: str) -> float:
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if not s:
        return 0.0
    return 1 - edlib.align(s, l, mode="HW", task="distance")["editDistance"] / len(s)


def well_files(meta_dir: str, lo: int, hi: int) -> list[tuple[int, str]]:
    out = []
    for f in glob.glob(os.path.join(meta_dir, "VDJ_*.csv")):
        m = re.search(r"VDJ_(\d+)\.csv$", f)
        if m and lo <= int(m.group(1)) <= hi and os.path.getsize(f) > 100:
            out.append((int(m.group(1)), f))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meta_dir")
    ap.add_argument("out")
    ap.add_argument("--wells", default="1-96")
    ap.add_argument("--sample-mod", type=int, default=200, help="keep barcodes with hash %% mod == 0")
    ap.add_argument("--species-id", type=float, default=0.95)
    ap.add_argument("--min-len", type=int, default=200)
    a = ap.parse_args()
    lo, hi = map(int, a.wells.split("-"))
    files = well_files(a.meta_dir, lo, hi)
    W = len(files)

    frames = []
    for w, f in files:
        frames.append(
            pl.scan_csv(f, schema_overrides={"consensus": pl.Utf8})
            .select("name", "umis", "reads", "consensus")
            .with_columns(pl.col("name").str.split("_").list.first().alias("barcode"), pl.lit(w).alias("well"))
            .filter(pl.col("barcode").hash(seed=0) % a.sample_mod == 0)
            .filter(pl.col("consensus").str.len_chars() >= a.min_len)
            .collect()
        )
    df = pl.concat(frames)
    print(f"{W} wells | sampled barcodes: {df['barcode'].n_unique()} | contigs: {df.height}")

    rows = []
    for (bc,), g in df.sort("umis", descending=True).group_by("barcode", maintain_order=True):
        reps, members = [], []
        for well, umis, cons in g.select("well", "umis", "consensus").iter_rows():
            for i, r in enumerate(reps):
                if hw_identity(cons, r) >= a.species_id:
                    members[i].append((well, umis))
                    break
            else:
                reps.append(cons)
                members.append([(well, umis)])
        for m in members:
            per_well = {}
            for w, u in m:
                per_well[w] = per_well.get(w, 0) + u
            tot = sum(per_well.values())
            rows.append(dict(barcode=bc, umis=tot, n_wells=len(per_well), modal_umis=max(per_well.values())))
    r = pl.DataFrame(rows).with_columns((1 - pl.col("modal_umis") / pl.col("umis")).alias("outside_frac"))
    r.write_parquet(a.out)

    print(f"chain species: {r.height} | in 1 well: {(r['n_wells'] == 1).mean():.1%}")
    bins = r.with_columns(pl.col("umis").cut([1, 2, 3, 5, 10, 20], labels=["1", "2", "3", "4-5", "6-10", "11-20", ">20"]).alias("umi_bin"))
    summ = (bins.group_by("umi_bin").agg(
        pl.len().alias("species"),
        pl.col("n_wells").median().alias("median_wells"),
        (pl.col("n_wells") == 1).mean().round(3).alias("frac_single_well"),
        (pl.col("umis").sum() - pl.col("modal_umis").sum()).alias("_out"), pl.col("umis").sum().alias("_tot"))
        .with_columns((pl.col("_out") / pl.col("_tot")).round(3).alias("umis_outside_modal_well"))
        .drop("_out", "_tot").sort("umi_bin"))
    pl.Config.set_tbl_rows(20)
    print(summ)
    big = r.filter(pl.col("umis") >= 6)
    if big.height:
        out = 1 - big["modal_umis"].sum() / big["umis"].sum()
        print(f"species with >=6 UMIs: outside-modal fraction {out:.3f} -> implied leakage f ~= {out / (1 - 1 / W):.3f} (W={W})")


if __name__ == "__main__":
    main()
