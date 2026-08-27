import re, glob as _glob
from pathlib import Path
import polars as pl

_HEAVY = "IGH"
_LIGHT = ("IGK", "IGL")
_REQUIRED = ["well", "barcode", "locus", "reads", "umis", "cluster_fraction", "pass_filters"]
_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
_QLABELS = ["p05", "p25", "p50", "p75", "p95"]

NO_LABELED_TRUTH_CAVEAT = (
    "LIMITATION (no labeled truth): this report summarizes marginal, per-contig "
    "distributions only. These metadata files carry no labeled ground truth for "
    "which heavy/light pairs are correct or incorrect, so this report cannot confirm "
    "or refute any specific mispairing hypothesis -- use these marginals only to "
    "bracket plausible knob ranges for the synthetic generator (Phase 0A, dataset-agnostic)."
)

def _bc_from_name(name):
    return re.split(r"_contig", str(name))[0] if name is not None else None

def _well_from_text(text):
    if text is None: return None
    m = re.search(r"well0*(\d+)", str(text), re.IGNORECASE)
    return int(m.group(1)) if m else None

def _locus_from_name(name):
    if name is None: return None
    s = str(name).upper()
    for loc in (_HEAVY,) + _LIGHT:
        if loc in s: return loc
    return None

def _pick(df, *names):
    for n in names:
        if n in df.columns: return n
    return None

def _to_int_or_none(v):
    if v is None: return None
    try:
        s = str(v).strip()
        return int(float(s)) if s and s.lower() != "null" else None
    except (ValueError, TypeError):
        return None

def normalize_metadata(raw):
    """Normalize a raw PairPlex `metadata/*.csv`-like frame (or an already-normalized
    frame -- this is idempotent) into the audit's required, dataset-agnostic schema:
    well, barcode, locus, reads, umis, cluster_fraction, pass_filters.

    Real PairPlex metadata rows carry `name` (== `{barcode}_contig-{i}`) but no
    `well`/`barcode`/`locus` columns directly; those are parsed, best-effort, from
    `name` and a filename-like column (e.g. "well007_metadata.csv"). Anything that
    can't be parsed is left null rather than raising -- this is a reporting utility,
    not a gate.
    """
    df = raw if isinstance(raw, pl.DataFrame) else pl.DataFrame(raw)
    if df.height == 0:
        return pl.DataFrame(schema={"well": pl.Int64, "barcode": pl.Utf8, "locus": pl.Utf8,
            "reads": pl.Int64, "umis": pl.Int64, "cluster_fraction": pl.Float64, "pass_filters": pl.Boolean})

    name_col = _pick(df, "name", "sequence_id", "contig_id")
    file_col = _pick(df, "source_file", "filename", "file")
    well_col = _pick(df, "well")
    bc_col = _pick(df, "barcode")
    locus_col = _pick(df, "locus")
    reads_col = _pick(df, "reads", "n_reads")
    umis_col = _pick(df, "umis", "n_umis", "umi_count")
    cf_col = _pick(df, "cluster_fraction", "cluster_frac")
    pf_col = _pick(df, "pass_filters", "pass_filter")

    names = df[name_col].to_list() if name_col else [None] * df.height
    files = df[file_col].to_list() if file_col else [None] * df.height
    wells_raw = df[well_col].to_list() if well_col else [None] * df.height
    bcs_raw = df[bc_col].to_list() if bc_col else [None] * df.height
    loci_raw = df[locus_col].to_list() if locus_col else [None] * df.height

    wells, barcodes, loci = [], [], []
    for i in range(df.height):
        w = _to_int_or_none(wells_raw[i])
        if w is None: w = _well_from_text(files[i])
        wells.append(w)
        barcodes.append(bcs_raw[i] if bcs_raw[i] is not None else _bc_from_name(names[i]))
        loci.append(loci_raw[i] if loci_raw[i] is not None else _locus_from_name(names[i]))

    return pl.DataFrame({
        "well": wells, "barcode": barcodes, "locus": loci,
        "reads": df[reads_col].to_list() if reads_col else [None] * df.height,
        "umis": df[umis_col].to_list() if umis_col else [None] * df.height,
        "cluster_fraction": df[cf_col].to_list() if cf_col else [None] * df.height,
        "pass_filters": df[pf_col].to_list() if pf_col else [None] * df.height,
    }).select(_REQUIRED)

def _load(normalized_glob_or_df):
    if isinstance(normalized_glob_or_df, pl.DataFrame):
        return normalized_glob_or_df
    pattern = str(normalized_glob_or_df)
    if any(c in pattern for c in "*?["):
        paths = sorted(_glob.glob(pattern))
    elif Path(pattern).is_dir():
        paths = sorted(str(p) for p in Path(pattern).glob("**/*.parquet"))
    else:
        paths = [pattern]
    if not paths:
        return pl.DataFrame(schema={c: pl.Float64 for c in _REQUIRED})
    frames = [pl.read_parquet(p) if str(p).endswith(".parquet") else pl.read_csv(p) for p in paths]
    return pl.concat(frames, how="diagonal_relaxed")

def _quantile_rows(df, col):
    s = df[col].drop_nulls().cast(pl.Float64, strict=False)
    n = s.len()
    rows = [{"section": col, "stat": "n", "value": None, "n": n}]
    if n == 0:
        rows += [{"section": col, "stat": lbl, "value": None, "n": 0} for lbl in ["mean"] + _QLABELS]
        return rows
    rows.append({"section": col, "stat": "mean", "value": float(s.mean()), "n": n})
    rows += [{"section": col, "stat": lbl, "value": float(s.quantile(q)), "n": n}
             for q, lbl in zip(_QUANTILES, _QLABELS)]
    return rows

def _profile_category(n_h, n_l):
    if n_h == 1 and n_l == 1: return "1H+1L"
    if n_h == 1 and n_l == 2: return "1H+2L"
    if n_h == 2 and n_l == 1: return "2H+1L"
    return "other"

def _contig_profile(df):
    cats = {"1H+1L": 0, "1H+2L": 0, "2H+1L": 0, "other": 0}
    if df.height == 0 or "pass_filters" not in df.columns or "locus" not in df.columns:
        return cats, 0
    passing = df.filter(pl.col("pass_filters") == True)  # noqa: E712 (polars boolean filter)
    if passing.height == 0:
        return cats, 0
    grp = passing.group_by(["well", "barcode"]).agg([
        (pl.col("locus") == _HEAVY).sum().alias("n_h"),
        pl.col("locus").is_in(list(_LIGHT)).sum().alias("n_l"),
    ])
    for r in grp.iter_rows(named=True):
        cats[_profile_category(int(r["n_h"]), int(r["n_l"]))] += 1
    return cats, grp.height

def audit_metadata(normalized_glob_or_df, report_path):
    """Marginal-only audit of normalized PairPlex metadata: quantiles of reads/UMIs/
    cluster_fraction, and a per-(well,barcode) contig-count profile among passing
    contigs (1H+1L / 1H+2L / 2H+1L / other frequencies). No calibration gate -- this
    never compares against or fits a threshold; it only reports. See
    NO_LABELED_TRUTH_CAVEAT: these marginals cannot confirm which pairs are wrong.
    """
    df = _load(normalized_glob_or_df)
    rows = []
    for col in ("reads", "umis", "cluster_fraction"):
        if col in df.columns:
            rows += _quantile_rows(df, col)
    cats, n_keys = _contig_profile(df)
    for k, v in cats.items():
        rows.append({"section": "contig_profile", "stat": k, "value": (v / n_keys) if n_keys else None, "n": v})
    summary = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"section": pl.Utf8, "stat": pl.Utf8, "value": pl.Float64, "n": pl.Int64})

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# SimPlex Phase 0A -- real-data marginal audit (no calibration gate)", "",
        NO_LABELED_TRUTH_CAVEAT, "",
        f"n_contigs={df.height}  n_(well,barcode)_keys_with_passing_contigs={n_keys}", "",
        "## Marginal quantiles"]
    for col in ("reads", "umis", "cluster_fraction"):
        sub = [r for r in rows if r["section"] == col]
        if sub:
            lines.append("- " + col + ": " + ", ".join(
                f"{r['stat']}={r['value']}" for r in sub if r["stat"] != "n") +
                f", n={sub[0]['n']}")
    lines += ["", "## Contig-count profile per (well, barcode) [passing contigs only]"]
    for k in ("1H+1L", "1H+2L", "2H+1L", "other"):
        v = cats[k]
        frac = (v / n_keys) if n_keys else 0.0
        lines.append(f"- {k}: count={v} fraction={frac:.4f}")
    report_path.write_text("\n".join(lines) + "\n")
    return summary
