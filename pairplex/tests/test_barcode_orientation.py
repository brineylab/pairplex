"""Orientation choice in parse_barcodes: an exact barcode hit beats a Hamming-1 correction."""

from unittest import mock

import abutils
import polars as pl
import pytest

from pairplex import pairplex as pp
from pairplex.utils import parse_barcodes

A = "AAAACCCCGGGGTTTT"
B = "ACGTTGCAACGTTGCA"
C = "CATGCATGGTACGTAC"
TSO = "TTTCTTATATGGG"
UMI_F = "AAAAAAAAAA"
UMI_R = "CCCCCCCCCC"
INSERT = "ACTGACTGACTGATTACA"


def mut(bc, i=5):
    """Hamming-1 variant of ``bc`` at position ``i``."""
    return bc[:i] + ("T" if bc[i] != "T" else "A") + bc[i + 1 :]


def rc(s):
    return abutils.tl.reverse_complement(s)


def read(fwd_bc, rc_bc):
    """A read whose forward start is ``fwd_bc`` and whose reverse-complement start is ``rc_bc``."""
    return fwd_bc + UMI_F + TSO + INSERT + rc(rc_bc + UMI_R + TSO)


@pytest.fixture
def whitelist(tmp_path):
    path = tmp_path / "wl.txt"
    path.write_text("\n".join([A, B, C]) + "\n")
    return str(path)


def _parse(tmp_path, whitelist, reads, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fq = tmp_path / "chunk.fastq"
    fq.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    res = parse_barcodes(fq, tmp_path / "out", whitelist_path=whitelist, return_counts=True, **kwargs)
    path, counts = res
    df = pl.read_parquet(path) if path else pl.DataFrame({"seq_id": [], "barcode": [], "umi": []})
    return df, counts


def test_exact_rc_beats_corrected_fwd(tmp_path, whitelist):
    df, counts = _parse(tmp_path, whitelist, [read(mut(A), B)])
    assert df["barcode"].to_list() == [B]
    assert df["umi"].to_list() == [UMI_R]
    assert counts["exact_rc"] == 1 and counts["switched"] == 1


def test_legacy_takes_corrected_fwd(tmp_path, whitelist):
    df, counts = _parse(tmp_path, whitelist, [read(mut(A), B)], prefer_exact=False)
    assert df["barcode"].to_list() == [A]
    assert df["umi"].to_list() == [UMI_F]
    assert all(v == 0 for v in counts.values())


def test_exact_fwd_beats_corrected_rc(tmp_path, whitelist):
    df, counts = _parse(tmp_path, whitelist, [read(A, mut(B))])
    assert df["barcode"].to_list() == [A]
    assert counts["exact_fwd"] == 1 and counts["both_exact"] == 0


def test_both_exact_keeps_forward_and_counts(tmp_path, whitelist):
    df, counts = _parse(tmp_path, whitelist, [read(A, B)])
    assert df["barcode"].to_list() == [A]
    assert df["umi"].to_list() == [UMI_F]
    assert counts["both_exact"] == 1


def test_both_corrected_keeps_forward_and_counts(tmp_path, whitelist):
    df, counts = _parse(tmp_path, whitelist, [read(mut(A), mut(B))])
    assert df["barcode"].to_list() == [A]
    assert counts["corrected_fwd"] == 1 and counts["both_corrected"] == 1


def test_only_rc_correctable(tmp_path, whitelist):
    df, counts = _parse(tmp_path, whitelist, [read("GGGGGGGGGGGGGGGG", mut(C))])
    assert df["barcode"].to_list() == [C]
    assert df["umi"].to_list() == [UMI_R]
    assert counts["corrected_rc"] == 1


def test_no_hit_dropped(tmp_path, whitelist):
    df, counts = _parse(tmp_path, whitelist, [read("GGGGGGGGGGGGGGGG", "GGGGGGGGGGGGGGGG")])
    assert df.height == 0
    assert sum(counts.values()) == 0


def test_check_rc_false_ignores_rc(tmp_path, whitelist):
    df, counts = _parse(tmp_path, whitelist, [read(mut(A), B)], check_rc=False)
    assert df["barcode"].to_list() == [A]
    assert counts["corrected_fwd"] == 1 and counts["switched"] == 0


@pytest.mark.parametrize("prefer_exact", [True, False])
def test_same_records_when_no_conflict(tmp_path, whitelist, prefer_exact):
    """Reads without an orientation conflict parse identically in both modes (incl. sequence)."""
    reads = [read(A, "G" * 16), read(mut(B), "G" * 16), read("G" * 16, C), read("G" * 16, mut(C))]
    df, _ = _parse(tmp_path, whitelist, reads, prefer_exact=prefer_exact)
    legacy, _ = _parse(tmp_path / "legacy", whitelist, reads, prefer_exact=False)
    assert df.sort("seq_id").equals(legacy.sort("seq_id"))


def test_default_return_is_path_only(tmp_path, whitelist):
    fq = tmp_path / "c.fastq"
    s = read(A, B)
    fq.write_text(f"@r\n{s}\n+\n{'I' * len(s)}\n")
    out = parse_barcodes(fq, tmp_path / "o", whitelist_path=whitelist)
    assert isinstance(out, str)


def _run(tmp_path, monkeypatch, whitelist, **kwargs):
    monkeypatch.setenv("PAIRPLEX_MAX_WORKERS", "2")
    tmp_path.mkdir(parents=True, exist_ok=True)
    fq = tmp_path / "well001.fastq"
    reads = [read(mut(A), B)] * 3 + [read(A, B)] * 3
    fq.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    out = tmp_path / "out"
    with mock.patch.object(pp.abstar, "run", return_value=[]):
        pp.run(str(fq), out, whitelist_path=whitelist, quiet=True, debug=True, **kwargs)
    return pl.read_parquet(out / "parsed" / "well001.parquet")


def test_run_default_and_legacy(tmp_path, monkeypatch, whitelist):
    new = _run(tmp_path / "new", monkeypatch, whitelist)
    assert sorted(new["barcode"].to_list()) == sorted([B] * 3 + [A] * 3)
    old = _run(tmp_path / "old", monkeypatch, whitelist, legacy=True)
    assert old["barcode"].to_list() == [A] * 6
    old2 = _run(tmp_path / "old2", monkeypatch, whitelist, barcode_prefer_exact=False)
    assert old2["barcode"].to_list() == [A] * 6


# --- interaction with the TSO G-run fix (#15): every orientation path must use strip_tso_g ---


@pytest.mark.parametrize("prefer_exact", [True, False])
def test_tso_g_strip_applies_in_every_orientation_path(tmp_path, whitelist, prefer_exact):
    """The read carries the TSO's GGG plus a genuine 5' G ("G" + INSERT). With tso_g_max=3 that G must
    survive whichever orientation code path parses the read (exact-hit path or legacy first-hit path)."""
    insert = "G" + INSERT
    r = A + UMI_F + TSO + insert + rc(mut(B) + UMI_R + TSO)
    df, _ = _parse(tmp_path / str(prefer_exact), whitelist, [r], prefer_exact=prefer_exact, tso_g_max=3)
    assert df["barcode"].to_list() == [A]
    assert df["sequence"][0].startswith(insert)


def test_tso_g_legacy_lstrip_in_exact_path(tmp_path, whitelist):
    r = A + UMI_F + TSO + "G" + INSERT + rc(mut(B) + UMI_R + TSO)
    df, _ = _parse(tmp_path, whitelist, [r], prefer_exact=True, tso_g_max=None)
    assert df["sequence"][0].startswith(INSERT)
