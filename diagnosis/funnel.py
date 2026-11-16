"""Where does PairPlex lose signal? Read / molecule funnel for one well.

Stages (pairplex.pairplex.run + utils, v0.3/0.4 per-well path):
  0 input merged reads
  1 valid barcode (whitelist exact or unique Hamming-1)            parse_barcodes
  2 barcode has >= min_cluster_reads reads in this well               partition prefilter
  3 contig passes filters (reads, UMIs, fraction)                     process_droplet
  4 contig annotated by abstar (present in *_unpaired)                abstar.run
  5 contig productive heavy/light                                     (not filtered by PairPlex; reported)
  6 contig ends in an emitted 1H+1L pair                              assign_pairs + exact-1H1L filter

Also tabulates the (barcode) composition after annotation (nH, nL) with read/UMI
support, i.e. what the exact-1H+1L rule discards and how strong those chains are.

Usage: python funnel.py PAIRPLEX_DIR WELL [--merged-dir DIR] [--min-cluster-reads 3]
"""
import argparse
import os

import polars as pl


def count_fastq_reads(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 24), b""):
            n += buf.count(b"\n")
    return n // 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pp_dir")
    ap.add_argument("well")
    ap.add_argument("--merged-dir", default=None)
    ap.add_argument("--min-cluster-reads", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d, w = a.pp_dir, a.well
    merged = os.path.join(a.merged_dir or os.path.join(d, "merged"), f"{w}.fastq")

    n_in = count_fastq_reads(merged) if os.path.exists(merged) else None
    parsed = pl.read_parquet(os.path.join(d, "parsed", f"{w}.parquet"), columns=["barcode", "umi"])
    bc = parsed.group_by("barcode").agg(pl.len().alias("reads"), pl.col("umi").n_unique().alias("umis"))
    kept_bc = bc.filter(pl.col("reads") >= a.min_cluster_reads)
    # molecules = distinct (barcode, umi)
    mol_all = parsed.unique(["barcode", "umi"]).height
    mol_kept = parsed.join(kept_bc.select("barcode"), on="barcode").unique(["barcode", "umi"]).height

    meta = pl.read_csv(os.path.join(d, "metadata", f"{w}.csv"), columns=["name", "reads", "umis", "pass_filters"])
    meta = meta.with_columns(pl.col("name").str.split("_").list.first().alias("barcode"))
    passed = meta.filter("pass_filters")

    ann = pl.read_parquet(os.path.join(d, "annotated", f"{w}_unpaired.parquet"),
                          columns=["sequence_id", "locus", "productive"])
    ann = ann.join(meta.select("name", "reads", "umis", "barcode"), left_on="sequence_id", right_on="name", how="left")
    prod = ann.filter(pl.col("productive") & pl.col("locus").is_in(["IGH", "IGK", "IGL"]))
    pairs = pl.read_parquet(os.path.join(d, "annotated", f"{w}_paired.parquet"),
                            columns=["sequence_id:0", "sequence_id:1"])
    paired_ids = pl.concat([pairs["sequence_id:0"], pairs["sequence_id:1"]]).unique()
    in_pairs = ann.filter(pl.col("sequence_id").is_in(paired_ids.implode()))

    def row(stage, reads, umis):
        return dict(stage=stage, reads=reads, umis=umis)

    rows = [
        row("0 input merged reads", n_in, None),
        row("1 valid barcode", parsed.height, mol_all),
        row(f"2 barcode >= {a.min_cluster_reads} reads in well", int(kept_bc["reads"].sum()), mol_kept),
        row("3 all contigs (clustered)", int(meta["reads"].sum()), int(meta["umis"].sum())),
        row("3b contigs passing filters", int(passed["reads"].sum()), int(passed["umis"].sum())),
        row("4 annotated by abstar", int(ann["reads"].sum()), int(ann["umis"].sum())),
        row("5 productive H/K/L", int(prod["reads"].sum()), int(prod["umis"].sum())),
        row("6 in emitted 1H+1L pairs", int(in_pairs["reads"].sum()), int(in_pairs["umis"].sum())),
    ]
    f = pl.DataFrame(rows, schema={"stage": pl.Utf8, "reads": pl.Int64, "umis": pl.Int64})
    base = f["reads"][1]
    f = f.with_columns((pl.col("reads") / base).round(3).alias("reads_vs_valid_bc"),
                       (pl.col("umis") / f["umis"][1]).round(3).alias("umis_vs_valid_bc"))
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(200)
    print(f"== {w}  (barcodes: valid {bc.height}, >= {a.min_cluster_reads} reads {kept_bc.height})")
    print(f)

    # composition of barcodes after annotation (productive chains only)
    comp = (prod.with_columns(pl.when(pl.col("locus") == "IGH").then(pl.lit("H")).otherwise(pl.lit("L")).alias("c"))
                .group_by("barcode").agg((pl.col("c") == "H").sum().alias("nH"), (pl.col("c") == "L").sum().alias("nL"),
                                         pl.col("umis").sum().alias("umis"), pl.col("reads").sum().alias("reads")))
    comp = comp.with_columns(pl.when((pl.col("nH") >= 3) | (pl.col("nL") >= 3)).then(pl.lit(">=3 of a chain"))
                               .otherwise(pl.format("{}H+{}L", "nH", "nL")).alias("composition"))
    tab = (comp.group_by("composition").agg(pl.len().alias("barcodes"), pl.col("umis").sum().alias("umis"))
               .with_columns((pl.col("barcodes") / pl.col("barcodes").sum()).round(3).alias("frac_barcodes"),
                             (pl.col("umis") / pl.col("umis").sum()).round(3).alias("frac_umis"))
               .sort("barcodes", descending=True))
    print("\nbarcode composition after annotation (productive chains):")
    print(tab)

    # strength of the chains discarded in 1H+2L / 2H+1L: is there a dominant chain?
    multi = prod.join(comp.filter(pl.col("composition").is_in(["1H+2L", "2H+1L"])).select("barcode", "composition"), on="barcode")
    if multi.height:
        dom = (multi.with_columns(pl.when(pl.col("locus") == "IGH").then(pl.lit("H")).otherwise(pl.lit("L")).alias("c"))
                    .group_by("barcode", "composition", "c").agg(pl.col("umis").sort(descending=True).alias("u"))
                    .filter(pl.col("u").list.len() == 2)
                    .with_columns((pl.col("u").list.get(0) / pl.col("u").list.sum()).alias("top_share"),
                                  (pl.col("u").list.get(0) >= 2 * pl.col("u").list.get(1)).alias("dominant_2x")))
        print(f"\nduplicated chain in 1H+2L / 2H+1L barcodes: n={dom.height}; "
              f"top chain >= 2x the other (UMIs): {dom['dominant_2x'].mean():.1%}; median top share {dom['top_share'].median():.2f}")
    if a.out:
        f.write_csv(a.out)


if __name__ == "__main__":
    main()
