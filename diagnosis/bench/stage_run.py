"""Benchmark stage 1: run the PairPlex code of a given checkout on one dataset.

Executed with the branch checkout first on sys.path, so `import pairplex` is the code under test.
Contract with branches (see diagnosis/bench/README.md):
  * entry point: pairplex.pairplex.run(reads, out, quiet=True, **kwargs)
  * per-well outputs (optional for cross-well modes): out/annotated/<well>_{paired,unpaired}.parquet
  * final pairs (optional): out/pairs/pairs.parquet with the pooled-pairs schema
    (barcode, well, name, locus/v_gene/cdr3_aa/sequence_id/sequence :0 and :1, confidence)
After run(), if the checkout has pairplex.pooled_pairing.run_pooled_pairing and per-well outputs
exist, pooled pairing is also run (out/pooled/pooled_pairs.parquet).

Usage: python stage_run.py CHECKOUT READS OUT [--kwargs JSON]
"""
import argparse
import json
import sys
import time
from pathlib import Path


def cap_mmseqs_threads(n: int = 1) -> None:
    """mmseqs crashes when its thread count exceeds an abstar batch size; PairPlex <= v2-base passes
    None for wells with >=1000 consensus sequences (mmseqs then uses every node CPU). Replace None by
    1; an explicit count chosen by the branch is kept."""
    import abstar
    orig = abstar.run

    def run(*args, mmseqs_threads=None, **kw):
        return orig(*args, mmseqs_threads=mmseqs_threads if mmseqs_threads is not None else n, **kw)

    abstar.run = run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkout")
    ap.add_argument("reads")
    ap.add_argument("out")
    ap.add_argument("--kwargs", default="{}")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.checkout).resolve()))
    import pairplex
    import pairplex.pairplex as ppx

    assert Path(pairplex.__file__).resolve().is_relative_to(Path(a.checkout).resolve()), pairplex.__file__
    cap_mmseqs_threads()
    kwargs = json.loads(a.kwargs)
    out = Path(a.out)
    t0 = time.time()
    ppx.run(a.reads, str(out), quiet=True, **kwargs)
    info = {"kwargs": kwargs, "sec_pairplex": round(time.time() - t0)}
    ann = list((out / "annotated").glob("*_paired.parquet")) if (out / "annotated").exists() else []
    try:
        from pairplex.pooled_pairing import run_pooled_pairing
    except ImportError:
        run_pooled_pairing = None
    if run_pooled_pairing and ann:
        t1 = time.time()
        _, pinfo = run_pooled_pairing(out, out / "pooled" / "pooled_pairs.parquet")
        info["pooled_info"] = pinfo
        info["sec_pooled"] = round(time.time() - t1)
    for sub in ("temp", "parsed"):
        p = out / sub
        if p.exists():
            import shutil
            shutil.rmtree(p, ignore_errors=True)
    (out / "stage_run.json").write_text(json.dumps(info, indent=1, default=str))


if __name__ == "__main__":
    main()
