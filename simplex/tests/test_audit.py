import polars as pl
from simplex.audit import normalize_metadata, audit_metadata

_REQUIRED = ["well", "barcode", "locus", "reads", "umis", "cluster_fraction", "pass_filters"]

def _synthetic_normalized():
    # four (well,barcode) keys, one of each contig-count profile among passing contigs:
    #  A: 1H+1L, B: 1H+2L, C: 2H+1L, D: "other" (1H only)
    rows = []
    def add(well, bc, locus, reads, umis, cf, passed=True):
        rows.append({"well": well, "barcode": bc, "locus": locus, "reads": reads, "umis": umis,
                      "cluster_fraction": cf, "pass_filters": passed})
    add(0, "A", "IGH", 20, 4, 0.6)
    add(0, "A", "IGK", 15, 3, 0.4)
    add(0, "B", "IGH", 30, 5, 0.5)
    add(0, "B", "IGK", 10, 2, 0.2)
    add(0, "B", "IGL", 8, 2, 0.15)
    add(0, "C", "IGH", 25, 4, 0.45)
    add(0, "C", "IGH", 22, 4, 0.4)
    add(0, "C", "IGK", 12, 2, 0.2)
    add(0, "D", "IGH", 18, 3, 0.5)
    # a non-passing contig that must not pollute the profile
    add(0, "D", "IGK", 2, 1, 0.02, passed=False)
    return pl.DataFrame(rows).select(_REQUIRED)

def test_normalize_metadata_passthrough_already_normalized():
    df = _synthetic_normalized()
    out = normalize_metadata(df)
    assert out.columns == _REQUIRED
    assert out.height == df.height
    assert out["barcode"].to_list() == df["barcode"].to_list()

def test_normalize_metadata_parses_barcode_and_well_from_name_and_filename():
    # mimic a real PairPlex metadata/*.csv row: only "name" (barcode_contig-N), no well/barcode columns
    raw = pl.DataFrame({
        "name": ["AAACCTGAGCTAACTC-1_contig-0", "AAACCTGAGCTAACTC-1_contig-1"],
        "locus": ["IGH", "IGK"],
        "reads": [12, 9],
        "umis": [3, 2],
        "cluster_fraction": [0.5, 0.3],
        "pass_filters": [True, True],
        "source_file": ["well007_metadata.csv", "well007_metadata.csv"],
    })
    out = normalize_metadata(raw)
    assert out.columns == _REQUIRED
    assert out["well"].to_list() == [7, 7]
    assert out["barcode"].to_list() == ["AAACCTGAGCTAACTC-1", "AAACCTGAGCTAACTC-1"]
    assert out["locus"].to_list() == ["IGH", "IGK"]

def test_audit_metadata_marginal_quantiles_and_profile(tmp_path):
    df = _synthetic_normalized()
    report_path = tmp_path / "audit_report.txt"
    summary = audit_metadata(df, report_path)

    # marginal quantile rows present for the three raw/algorithm-independent observables
    for col in ("reads", "umis", "cluster_fraction"):
        sub = summary.filter(pl.col("section") == col)
        assert set(sub["stat"].to_list()) >= {"n", "mean", "p05", "p25", "p50", "p75", "p95"}

    # contig-count profile rows: exactly one key in each of 1H+1L / 1H+2L / 2H+1L, one "other"
    prof = summary.filter(pl.col("section") == "contig_profile").sort("stat")
    counts = dict(zip(prof["stat"].to_list(), prof["n"].to_list()))
    assert counts == {"1H+1L": 1, "1H+2L": 1, "2H+1L": 1, "other": 1}
    fracs = dict(zip(prof["stat"].to_list(), prof["value"].to_list()))
    assert fracs["1H+1L"] == 0.25

    # report file exists and explicitly states the no-labeled-truth limitation
    assert report_path.exists()
    text = report_path.read_text().lower()
    assert "no labeled" in text or "no-labeled" in text
    assert "cannot confirm" in text or "cannot confirm or refute" in text

def test_audit_metadata_accepts_glob_of_files(tmp_path):
    df = _synthetic_normalized()
    half = df.height // 2
    df[:half].write_parquet(tmp_path / "well000.parquet")
    df[half:].write_parquet(tmp_path / "well001.parquet")
    summary = audit_metadata(str(tmp_path / "*.parquet"), tmp_path / "report.txt")
    total_n = summary.filter((pl.col("section") == "reads") & (pl.col("stat") == "n"))["n"][0]
    assert total_n == df.height

def test_audit_metadata_empty_input_is_safe(tmp_path):
    empty = pl.DataFrame(schema={c: pl.Float64 for c in _REQUIRED})
    summary = audit_metadata(empty, tmp_path / "report_empty.txt")
    assert summary.height >= 0
    assert (tmp_path / "report_empty.txt").exists()
