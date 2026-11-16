"""Static figures for the PairPlex diagnosis deck (PNG, light surface).

Reads result files under RESULTS (default ~/work/PairPlex/diagnosis/results) and writes PNGs to
FIGURES. Each figure is skipped (with a message) if its inputs are missing, so this can be re-run
as results land. Style: thin marks, recessive axes, one y-axis, fixed categorical order
(blue, orange, aqua, yellow), visible value labels (aqua/yellow are below 3:1 contrast).

Usage: python diagnosis/figures.py [--results DIR] [--figures DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 (validated)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def save(fig, out, name):
    import textwrap
    if len(fig.axes) == 1:  # move the axis title to a wrapped, figure-level, left-aligned title
        ax = fig.axes[0]
        t = ax.get_title(loc="left")
        if t:
            ax.set_title("", loc="left")
            fig.suptitle("\n".join(textwrap.wrap(t.replace("\n", " "), 70)), x=0.01, ha="left",
                         fontweight="bold", fontsize=13)
    fig.tight_layout()
    p = out / f"{name}.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("wrote", p)


def hbar(ax, labels, values, color=C[0], fmt="{:.0%}"):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=color, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, labels)
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt.format(v), va="center", color=INK, fontsize=10)
    ax.grid(axis="y", visible=False)


def fig_funnel(R, out):
    f = R / "funnel_donor01_VDJ_74.csv"
    if not f.exists():
        return print("skip funnel")
    d = pl.read_csv(f).filter(pl.col("umis").is_not_null())
    labels = ["valid barcode", "≥3 reads in well", "contigs (UMI-sum)", "pass filters", "annotated (abstar)",
              "productive H/K/L", "in emitted 1H+1L pairs"]
    fig, ax = plt.subplots(figsize=(8, 4))
    hbar(ax, labels, d["umis_vs_valid_bc"].to_list())
    ax.set_xlabel("molecules, relative to valid-barcode molecules")
    ax.set_title("Only 6.8% of molecules reach an emitted pair (donor01, one well)")
    ax.set_xlim(0, 1.25)
    save(fig, out, "01_funnel")


def fig_leakage_curve(R, out):
    f = R / "residency_dedup_donor01_naive_id0.94.parquet"
    if not f.exists():
        return print("skip leakage curve")
    r = pl.read_parquet(f)
    b = (r.with_columns(pl.col("umis").clip(1, 25).alias("n")).group_by("n")
           .agg((1 - pl.col("modal").sum() / pl.col("umis").sum()).alias("out"), pl.len().alias("species")).sort("n")
           .filter(pl.col("species") >= 20))
    W, f_ = 76, 0.78
    n = np.arange(1, 26)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(b["n"], b["out"], color=C[0], lw=2, marker="o", ms=5, label="observed")
    ax.axhline(f_ * (1 - 1 / W), color=INK2, lw=1, ls="--")
    ax.text(25, f_ * (1 - 1 / W) + 0.02, "f = 0.78 asymptote", ha="right", color=INK2, fontsize=10)
    ax.axhline(0, color=C[1], lw=2)
    ax.text(25, 0.03, "expected if cDNA stayed in cells", ha="right", color=INK2, fontsize=10)
    ax.set_xlabel("distinct molecules (UMIs) of one chain, pooled over wells")
    ax.set_ylabel("share of molecules outside the main well")
    ax.set_ylim(-0.05, 1)
    ax.set_title("A chain's molecules are spread across wells: barcoded cDNA leaks out of fixed cells")
    save(fig, out, "02_leakage_curve")


def fig_leakage_by_chain(R, out):
    f = R / "leakage" / "leakage_by_chain.jsonl"
    if not f.exists():
        return print("skip leakage by chain")
    d = pl.DataFrame([json.loads(l) for l in open(f)]).with_columns(
        pl.col("label").str.split("_").list.get(0).alias("donor"), pl.col("label").str.split("_").list.get(1).alias("reaction"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, rx in zip(axes, ("naive", "memory")):
        s = d.filter(pl.col("reaction") == rx).sort("donor")
        x = np.arange(s.height)
        for i, (c, lab) in enumerate((("H", "heavy"), ("K", "kappa"), ("L", "lambda"))):
            v = [val if n >= 50 else np.nan for val, n in zip(s[f"f6_{c}"], s[f"n6_{c}"])]
            ax.bar(x + (i - 1) * 0.26, v, width=0.24, color=C[i], edgecolor=SURFACE, linewidth=1.5, label=lab)
        ax.set_xticks(x, [dn.replace("donor", "") for dn in s["donor"]])
        ax.set_title(f"{rx} reaction")
        ax.set_xlabel("donor")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("estimated leakage f")
    axes[0].set_ylim(0, 1)
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle("Leakage is high in every reaction, and higher for light than heavy chains", x=0.01, ha="left",
                 fontweight="bold")
    fig.legend(h, l, loc="upper right", ncols=3, bbox_to_anchor=(0.99, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = out / "03_leakage_by_chain.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("wrote", p)


def fig_satellites(out):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels = ["observed", "expected: ambient before RT (a)", "expected: re-barcoding in place (b)"]
    vals = [0.0182, 0.0175, 0.230]
    hbar(ax, labels, vals, color=C[0], fmt="{:.1%}")
    ax.set_xlabel("share of a chain's copies (other barcodes) found in the parent cell's well")
    ax.set_xlim(0, 0.3)
    ax.set_title("Cross-barcode copies land independently of the parent cell → ambient mRNA")
    save(fig, out, "05_satellites")


def fig_across_barcode(R, out):
    f = R / "across_barcode_donor01_naive.parquet"
    if not f.exists():
        return print("skip across-barcode")
    k = pl.read_parquet(f)
    bins = [(1, 1, "1"), (2, 4, "2–4"), (5, 9, "5–9"), (10, 49, "10–49"), (50, 10**9, "≥50")]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(bins))
    for i, loc in enumerate(("IGH", "IGK", "IGL")):
        s = k.filter(pl.col("locus") == loc)
        tot = s["obs"].sum()
        v = [s.filter((pl.col("n_barcodes") >= lo) & (pl.col("n_barcodes") <= hi))["obs"].sum() / tot for lo, hi, _ in bins]
        ax.bar(x + (i - 1) * 0.26, v, width=0.24, color=C[i], edgecolor=SURFACE, linewidth=1.5, label=loc)
    ax.set_xticks(x, [b[2] for b in bins])
    ax.set_xlabel("number of droplet barcodes carrying the same V(D)J")
    ax.set_ylabel("share of occurrences")
    ax.legend(ncols=3)
    ax.grid(axis="x", visible=False)
    ax.set_title("Naive heavy chains under many barcodes: not clonality (donor01 naive)")
    save(fig, out, "04_across_barcode")


def fig_presplit(out):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    groups = ["same molecule seen in\nanother well (post-pool)", "distinct molecules of a chain\nin another well (leakage)"]
    obs, exp = [0.597, 0.242], [0.307, 0.240]
    y = np.arange(2)[::-1]
    ax.barh(y + 0.17, obs, height=0.32, color=C[0], edgecolor=SURFACE, linewidth=2, label="observed")
    ax.barh(y - 0.17, exp, height=0.32, color=C[1], edgecolor=SURFACE, linewidth=2, label="expected if uniform over wells")
    for yi, o, e in zip(y, obs, exp):
        ax.text(o, yi + 0.17, f" {o:.0%}", va="center", fontsize=10, color=INK)
        ax.text(e, yi - 0.17, f" {e:.0%}", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, groups)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("share landing in the main well's i7 pool of 24")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_title("Leaked molecules ignore pool boundaries: displacement happens before the split")
    save(fig, out, "11_presplit")


def fig_identity(out):
    # distributions measured in-session (cluster_purity / noise_control): summary quantiles
    fig, ax = plt.subplots(figsize=(8, 3.8))
    rows = [("reads of the same molecule", 0.946, 0.977, C[0]),
            ("distinct light chains, same V gene", 0.90, 0.952, C[1]),
            ("distinct heavy chains, same V gene", 0.83, 0.889, C[2])]
    for i, (lab, lo, med, col) in enumerate(rows[::-1]):
        ax.plot([lo, med], [i, i], color=col, lw=6, solid_capstyle="round")
        ax.plot(med, i, "o", color=col, ms=9, mec=SURFACE, mew=2)
        ax.text(med + 0.004, i, f" median {med:.3f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(3), [r[0] for r in rows[::-1]])
    ax.axvline(0.90, color=INK2, ls="--", lw=1)
    ax.text(0.901, 2.45, "PairPlex default threshold 0.90", color=INK2, fontsize=10)
    ax.set_xlim(0.8, 1.02)
    ax.set_xlabel("identity (PairPlex metric: 1 − edit distance / shorter length)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Clustering cannot separate two light chains that share a V gene")
    save(fig, out, "06_identity")


def fig_homewell_real(R, out):
    f = R / "homewell" / "donor07_memory_purity.csv"
    if not f.exists():
        return print("skip homewell real")
    d = pl.read_csv(f)
    sub = d.filter(pl.col("set").str.starts_with("home-well conf"))
    labels = [s.replace("home-well conf ", "") for s in sub["set"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    ax.bar(x, sub["purity_within"], color=C[0], width=0.6, edgecolor=SURFACE, linewidth=2)
    for xi, v, n in zip(x, sub["purity_within"], sub["pairs"]):
        ax.text(xi, v + 0.02, f"{v:.0%}\n(n={n:,})", ha="center", fontsize=9, color=INK)
    cur = d.filter(pl.col("set") == "current PairPlex per-well")["purity_within"][0]
    ax.axhline(cur, color=C[1], lw=2)
    ax.set_xlim(-0.5, len(labels) + 0.4)
    ax.text(len(labels) - 0.45, cur, f"current\nPairPlex {cur:.0%}", ha="left", va="center", color=INK2, fontsize=10)
    ax.axhline(0.998, color=INK2, lw=1, ls="--")
    ax.text(-0.4, 1.02, "reference datasets 99.8%", color=INK2, fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("home-well pairing confidence")
    ax.set_ylabel("clonal light-chain purity")
    ax.grid(axis="x", visible=False)
    ax.set_title("Real data (donor07 memory): purity rises with confidence;\nfew pairs are confident")
    save(fig, out, "07_homewell_real")


def fig_simplex_v1(R, out):
    f = R / "simplex_v1" / "phase2_regimes.csv"
    if not f.exists():
        return print("skip simplex v1")
    d = pl.read_csv(f).group_by("release_rate", "recovery_rate").agg(pl.col("precision").mean()).sort("release_rate")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, rec in enumerate((0.9, 0.5, 0.15)):
        s = d.filter(pl.col("recovery_rate") == rec)
        ax.plot(s["release_rate"], s["precision"], color=C[i], lw=2, marker="o", ms=6, label=f"recovery {rec}")
    ax.axvspan(0.7, 0.88, color=GRID, alpha=0.6)
    ax.text(0.79, 0.62, "measured\nleakage", ha="center", color=INK2, fontsize=10)
    ax.set_xlim(-0.02, 0.9)
    ax.set_ylim(0.6, 1.02)
    ax.set_xlabel("release rate (leakage)")
    ax.set_ylabel("precision (correct / emitted)")
    ax.legend(loc="lower left")
    ax.set_title("SimPlex v1 swept leakage 0–0.2; the real value is ~0.8")
    save(fig, out, "08_simplex_v1_regimes")


def fig_simplex_v2(R, out):
    # prefer the corrected-fragment profile (v2c, split in a/ and b/), fall back to v2a
    files = sorted((R / "simplex_v2c").glob("*/simplex_validation.jsonl")) or [R / "simplex_v2" / "simplex_validation.jsonl"]
    files = [f for f in files if f.exists()]
    if not files:
        return print("skip simplex v2")
    tag = "v2c" if "simplex_v2c" in str(files[0]) else "v2a"
    rows = [json.loads(l) for f in files for l in open(f) if l.strip()]
    order = ["clean", "improved_protocol", "no_leakage", "no_ambient", "leak_0.5", "calibrated"]
    rows = sorted([r for r in rows if r["scenario"] in order], key=lambda r: order.index(r["scenario"]))
    if not rows:
        return print("skip simplex v2 (empty)")
    labels = [r["scenario"].replace("_", " ").replace("improved protocol", "improved\nprotocol") for r in rows]
    meth = [("per-well (default)", "per_well"), ("per-well, min UMIs 2", "per_well_mcu2"), ("pooled home-well", "pooled")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    x = np.arange(len(rows))
    for ax, metric, title in ((axes[0], "precision", "precision (correct / emitted pairs)"),
                              (axes[1], "cell_recall", "cell recall (correct pairs / cells with both chains)")):
        for i, (lab, k) in enumerate(meth):
            v = [r[k][metric] or 0 for r in rows]
            n = [r[k]["emitted"] for r in rows]
            ax.bar(x + (i - 1) * 0.26, v, width=0.24, color=C[i], edgecolor=SURFACE, linewidth=1.5, label=lab)
            for xi, vi, ni in zip(x, v, n):
                txt = f"{vi:.0%}" + (f"\nn={ni}" if metric == "precision" and ni < 50 else "")
                ax.text(xi + (i - 1) * 0.26, vi + 0.015, txt, ha="center", fontsize=7, color=INK)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(0, 1.22)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("vs ground truth")
    axes[0].legend(ncols=3, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=8, frameon=False)
    fig.suptitle(f"SimPlex {tag} ground truth (4,000 cells, 48 wells): leakage drives the failure; pooled pairing recovers when leakage is low")
    save(fig, out, "09_simplex_v2_scenarios")


def fig_burden(R, out):
    f = R / "burden" / "burden_vs_viability.csv"
    if not f.exists():
        return print("skip burden")
    d = pl.read_csv(f).sort("donor_n")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(d.height)
    ax.bar(x, d["occ_frac_ge10"], color=C[0], width=0.6, edgecolor=SURFACE, linewidth=2)
    for xi, v, vb in zip(x, d["occ_frac_ge10"], d["pbmc_viability"]):
        ax.text(xi, v + 0.004, f"{v:.1%}", ha="center", fontsize=8, color=INK)
        ax.text(xi, -0.022, f"{vb:.2f}", ha="center", fontsize=8, color=INK2)
    ax.text(-0.9, -0.022, "viab.", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(x, [str(n) for n in d["donor_n"]])
    ax.set_ylim(-0.03, max(d["occ_frac_ge10"]) * 1.2)
    ax.set_xlabel("donor (PBMC viability shown below)")
    ax.set_ylabel("share under ≥10 barcodes")
    ax.grid(axis="x", visible=False)
    ax.set_title("Ambient burden varies 60× between donors and does not track viability")
    save(fig, out, "10_burden_by_donor")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path.home() / "work/PairPlex/diagnosis/results"))
    ap.add_argument("--figures", default=str(Path.home() / "work/PairPlex/diagnosis/figures"))
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.figures)
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_funnel, fig_leakage_curve, fig_leakage_by_chain, fig_across_barcode, fig_homewell_real,
               fig_simplex_v1, fig_simplex_v2, fig_burden):
        try:
            fn(R, out)
        except Exception as e:  # keep going
            print(f"FAILED {fn.__name__}: {e}")
    fig_satellites(out)
    fig_identity(out)
    fig_presplit(out)


if __name__ == "__main__":
    main()
