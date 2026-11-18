"""Only the TSO's rGrGrG run is stripped after the TSO prefix; genuine 5' Gs are kept."""

from unittest import mock

import polars as pl
import pytest

from pairplex import pairplex as pp
from pairplex.utils import DEFAULT_WHITELIST, parse_barcodes, strip_tso_g

TSO = "TTTCTTATATGGG"
UMI = "ACGTACGTAC"


def _barcodes(n):
    with open(DEFAULT_WHITELIST) as f:
        return [next(f).strip() for _ in range(n)]


@pytest.mark.parametrize(
    "insert, k, expected",
    [
        ("GGGACT", 3, "ACT"),  # TSO run only
        ("GGGGACT", 3, "GACT"),  # one genuine 5' G kept
        ("GGGGGACT", 3, "GGACT"),
        ("GGACT", 3, "ACT"),  # shorter run (e.g. a miscalled G): strip what is there
        ("ACT", 3, "ACT"),
        ("GGGGACT", 4, "ACT"),
        ("GGGGACT", 0, "GGGGACT"),
        ("GGGGACT", None, "ACT"),  # legacy lstrip("G")
        ("GGGGG", 3, "GG"),
        ("GG", 3, ""),
        ("", 3, ""),
    ],
)
def test_strip_tso_g(insert, k, expected):
    assert strip_tso_g(insert, k) == expected


def _fastq(path, records):
    with open(path, "w") as f:
        for i, seq in enumerate(records):
            f.write(f"@r{i}\n{seq}\n+\n{'I' * len(seq)}\n")


@pytest.mark.parametrize("k, expected", [(3, "GGACAGTC"), (None, "ACAGTC"), (4, "GACAGTC")])
def test_parse_barcodes_keeps_genuine_g(tmp_path, k, expected):
    bc = _barcodes(1)[0]
    _fastq(tmp_path / "chunk.fastq", [bc + UMI + TSO + "GGACAGTC"])
    out = parse_barcodes(tmp_path / "chunk.fastq", tmp_path / "out", tso_g_max=k)
    df = pl.read_parquet(out)
    assert df["sequence"].to_list() == [expected]
    assert df["barcode"].to_list() == [bc]


def test_parse_barcodes_default_is_three():
    import inspect

    assert inspect.signature(parse_barcodes).parameters["tso_g_max"].default == 3
    assert inspect.signature(pp.run).parameters["tso_g_max"].default == 3
    assert inspect.signature(pp.run).parameters["legacy"].default is False


def _run_parsed(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv("PAIRPLEX_MAX_WORKERS", "2")
    reads = tmp_path / "well001.fastq"
    _fastq(reads, [bc + UMI + TSO + "GACAGTCAGT" * 10 for bc in _barcodes(2) for _ in range(3)])
    out = tmp_path / "out"
    with mock.patch.object(pp.abstar, "run", return_value=[]):
        pp.run(str(reads), out, quiet=True, debug=True, **kwargs)
    return pl.read_parquet(out / "parsed" / "well001.parquet")["sequence"].to_list()


def test_run_default_keeps_genuine_g(tmp_path, monkeypatch):
    seqs = _run_parsed(tmp_path, monkeypatch)
    assert seqs and all(s.startswith("GACAGT") for s in seqs)


def test_run_legacy_restores_lstrip(tmp_path, monkeypatch):
    seqs = _run_parsed(tmp_path, monkeypatch, legacy=True, tso_g_max=5)
    assert seqs and all(s.startswith("ACAGT") for s in seqs)


@pytest.mark.parametrize("bad", [-1, None, 2.5])
def test_run_rejects_bad_tso_g_max(tmp_path, bad):
    with pytest.raises(ValueError):
        pp.run(str(tmp_path), tmp_path / "out", quiet=True, tso_g_max=bad)
