"""Information ceiling for pairing on SimPlex ground truth: how many cells could ANY algorithm pair
correctly from what was sequenced, and how precise is a pooled (all-wells) barcode-level 1H+1L rule.

Tiers per cell with both chains sequenced under its own barcode:
  T1 barcode-unique: under the cell's barcode (all wells, ambient copies included) the only heavy
     source and the only light source are the cell's own chains -> unambiguous by barcode.
  T2 clean co-occurrence: the cell's heavy and light are both present in some well under its
     barcode and that (well, barcode) holds no other heavy or light source -> pairable by an ideal
     algorithm that knew which per-well 1H+1L co-occurrences are genuine (upper bound; the current
     per-well rule emits these but cannot tell them from coincidental 1H+1L).
  T3 home-well oracle (stricter): modal wells of both chains = resident well, unclaimed by others.
Sequence-level losses (annotation of fragments, clustering) are NOT modelled here: compare with the
PairPlex-observed numbers of the same run.

Usage: python simplex_ceiling.py SCENARIO_DIR [SCENARIO_DIR ...] [--out ceiling.csv]   (each holds sim/truth)
"""
import json
import sys
from pathlib import Path

import polars as pl


def ceiling(d):
    d = Path(d)
    tc = pl.read_parquet(d / "sim/truth/truth_components.parquet").drop("sequence")
    cells = pl.read_parquet(d / "sim/truth/truth_cells.parquet").select("cell_id", "barcode", "resident_well")
    tc = tc.with_columns(cls=pl.when(pl.col("locus") == "IGH").then(pl.lit("H")).otherwise(pl.lit("L")))
    # sources under each barcode, all wells: (origin_cell, class) with molecules per well
    src = tc.group_by("barcode", "origin_cell_id", "cls", "final_well").agg(pl.col("n_source_molecules").sum().alias("m"), pl.col("n_reads").sum().alias("r"))
    tot = src.group_by("barcode", "origin_cell_id", "cls").agg(pl.col("m").sum().alias("M"))
    modal = (src.sort(["m", "r"], descending=True).group_by("barcode", "origin_cell_id", "cls", maintain_order=True)
             .agg(pl.col("final_well").first().alias("modal_well"), pl.col("m").first().alias("m_top")))
    tot = tot.join(modal, on=["barcode", "origin_cell_id", "cls"])
    per_bc = tot.group_by("barcode", "cls").agg(pl.len().alias("n_src"))
    # own chains present under own barcode
    own = tot.join(cells.rename({"cell_id": "origin_cell_id"}), on=["barcode", "origin_cell_id"], how="inner")
    both = own.group_by("origin_cell_id").agg(pl.col("cls").n_unique().alias("ncls")).filter(pl.col("ncls") == 2)["origin_cell_id"]
    own = own.filter(pl.col("origin_cell_id").is_in(both.implode()))
    own = own.join(per_bc, on=["barcode", "cls"])
    # T1
    t1 = own.group_by("origin_cell_id").agg((pl.col("n_src") == 1).all().alias("t1"))
    # T2: modal well == resident well and unique among same-class sources of the barcode
    claim = tot.group_by("barcode", "cls", "modal_well").agg(pl.len().alias("n_claim"))
    o2 = own.join(claim, on=["barcode", "cls", "modal_well"])
    t2 = o2.group_by("origin_cell_id").agg(((pl.col("modal_well") == pl.col("resident_well")) & (pl.col("n_claim") == 1)).all().alias("t2"))
    t2 = t2.rename({"t2": "t3"})
    # T2: clean co-occurrence in any well
    pw = src.group_by("barcode", "final_well", "cls").agg(pl.len().alias("k"))
    pw = pw.pivot(on="cls", index=["barcode", "final_well"], values="k").fill_null(0)
    clean_w = pw.filter((pl.col("H") == 1) & (pl.col("L") == 1)).select("barcode", "final_well")
    ownw = src.join(cells.rename({"cell_id": "origin_cell_id"}).select("origin_cell_id", "barcode"), on=["barcode", "origin_cell_id"])
    ownw = ownw.join(clean_w, on=["barcode", "final_well"]).group_by("origin_cell_id", "final_well").agg(pl.col("cls").n_unique().alias("nc"))
    co = ownw.filter(pl.col("nc") == 2).select("origin_cell_id").unique().with_columns(pl.lit(True).alias("t2"))
    t = t1.join(t2, on="origin_cell_id").join(co, on="origin_cell_id", how="left").with_columns(pl.col("t2").fill_null(False))
    n = t.height
    n1 = int(t["t1"].sum())
    n12 = int((t["t1"] | t["t2"]).sum())
    n13 = int((t["t1"] | t["t3"]).sum())
    # pooled barcode-level 1H+1L rule: barcodes with exactly one H and one L source (all wells)
    w = per_bc.pivot(on="cls", index="barcode", values="n_src").fill_null(0)
    b11 = w.filter((pl.col("H") == 1) & (pl.col("L") == 1))["barcode"]
    pairs = tot.filter(pl.col("barcode").is_in(b11.implode())).group_by("barcode").agg(pl.col("origin_cell_id").n_unique().alias("nc"))
    prec11 = float((pairs["nc"] == 1).mean()) if pairs.height else None
    m = json.load(open(d / "metrics.json")) if (d / "metrics.json").exists() else {}
    return dict(scenario=d.name, cells_both_sequenced=n, T1_barcode_unique=n1, T1_or_T2_cooccur=n12, T1_or_T3_homewell=n13,
                frac_T1=round(n1 / n, 3), frac_T1T2=round(n12 / n, 3),
                barcode_1H1L_rule_pairs=pairs.height, barcode_1H1L_rule_precision=round(prec11, 3) if prec11 is not None else None)


if __name__ == "__main__":
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        i = args.index("--out")
        out = args[i + 1]
        args = args[:i] + args[i + 2:]
    rows = [ceiling(d) for d in args]
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_cols(20)
    df = pl.DataFrame(rows)
    print(df)
    if out:
        df.write_csv(out)
