"""run() must always hand abstar.run an explicit MMseqs2 thread count.

With ``mmseqs_threads=None`` MMseqs2 uses every online CPU of the node and crashes when that
exceeds the size of an abstar sub-query batch (seen: 256 threads vs 67 sequences).
"""

import random
from unittest import mock

import pytest

from pairplex import pairplex as pp
from pairplex.utils import DEFAULT_WHITELIST

TSO = "TTTCTTATATGGG"


def _write_reads(path, n_barcodes=3, reads_per_chain=4):
    rng = random.Random(0)
    with open(DEFAULT_WHITELIST) as f:
        barcodes = [next(f).strip() for _ in range(n_barcodes)]
    chains = ["".join(rng.choice("ACT") + rng.choice("ACGT") for _ in range(150)) for _ in range(2)]
    with open(path, "w") as f:
        i = 0
        for bc in barcodes:
            for chain in chains:
                for _ in range(reads_per_chain):
                    umi = "".join(rng.choice("ACGT") for _ in range(10))
                    seq = bc + umi + TSO + chain
                    f.write(f"@r{i}\n{seq}\n+\n{'I' * len(seq)}\n")
                    i += 1


@pytest.mark.parametrize("threads", [None, 4])
def test_abstar_receives_explicit_mmseqs_threads(tmp_path, monkeypatch, threads):
    monkeypatch.setenv("PAIRPLEX_MAX_WORKERS", "2")
    reads = tmp_path / "well001.fastq"
    _write_reads(reads)
    kwargs = {} if threads is None else {"mmseqs_threads": threads}
    with mock.patch.object(pp.abstar, "run", return_value=[]) as abstar_run:
        pp.run(str(reads), tmp_path / "out", quiet=True, **kwargs)
    assert abstar_run.call_count == 1
    got = abstar_run.call_args.kwargs["mmseqs_threads"]
    assert got is not None
    assert got == (1 if threads is None else threads)


def test_large_well_still_gets_explicit_threads(tmp_path, monkeypatch):
    """v2-base passed None once a well had >= 1000 consensus sequences: the crashing case."""
    monkeypatch.setenv("PAIRPLEX_MAX_WORKERS", "2")
    reads = tmp_path / "well001.fastq"
    _write_reads(reads, n_barcodes=510, reads_per_chain=3)
    with mock.patch.object(pp.abstar, "run", return_value=[]) as abstar_run:
        pp.run(str(reads), tmp_path / "out", quiet=True)
    consensus = (tmp_path / "out" / "consensus" / "well001.fasta").read_text().count(">")
    assert consensus >= 1000
    assert abstar_run.call_args.kwargs["mmseqs_threads"] == 1


@pytest.mark.parametrize("bad", [0, -2, None, 1.5])
def test_invalid_mmseqs_threads_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        pp.run(str(tmp_path), tmp_path / "out", quiet=True, mmseqs_threads=bad)


def test_correct_barcodes_large_input_gets_explicit_threads(tmp_path, monkeypatch):
    """correct-barcodes annotates the raw input; v2-base passed None once it had >= 1000 reads."""
    from pairplex import correct_barcodes as cb

    monkeypatch.setenv("PAIRPLEX_MAX_WORKERS", "2")
    reads = tmp_path / "well001.fastq"
    _write_reads(reads, n_barcodes=101, reads_per_chain=6)  # 1212 reads

    class _Stop(Exception):
        pass

    with mock.patch.object(cb.abstar, "run", side_effect=_Stop) as abstar_run:
        with pytest.raises(_Stop):
            cb.run(str(reads), tmp_path / "out", temp_directory=tmp_path / "tmp", quiet=True)
    assert abstar_run.call_args.kwargs["mmseqs_threads"] == 1
