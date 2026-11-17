"""Fixed real-data benchmark input: memory wells of one reaction, 1 in N droplet barcodes.

Keeps a read when its whitelist-corrected 10x barcode (same orientation logic as
pairplex.utils.parse_barcodes) satisfies crc32(barcode) % N == 0. All wells of the reaction are
kept, so cross-well structure (leakage, home wells) is preserved for the barcodes retained.

Usage (run once per donor, repo root, simplex venv):
  python diagnosis/bench/make_real_subset.py SRC_MERGED_DIR OUT_DIR --wells 97-192 --keep-every 8 [--jobs 96]
"""
import argparse
import gzip
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import abutils  # noqa: E402

from pairplex.utils import DEFAULT_WHITELIST, correct_barcode, load_barcode_whitelist  # noqa: E402

_WL = None


def _keep(seq: str, n: int) -> bool:
    for s in (seq, abutils.tl.reverse_complement(seq)):
        bc = correct_barcode(s[:16], _WL)
        if bc is not None:  # first orientation with a (corrected) whitelist hit, as parse_barcodes
            return zlib.crc32(bc.encode()) % n == 0
    return False


def subset_well(src: str, dst: str, n: int) -> tuple[str, int, int]:
    global _WL
    if _WL is None:
        _WL = load_barcode_whitelist(DEFAULT_WHITELIST)
    tot = kept = 0
    with open(src) as fin, gzip.open(dst + ".tmp", "wt", compresslevel=3) as fout:
        while True:
            h = fin.readline()
            if not h:
                break
            s, p, q = fin.readline(), fin.readline(), fin.readline()
            tot += 1
            if _keep(s.strip(), n):
                kept += 1
                fout.write(h + s + p + q)
    Path(dst + ".tmp").rename(dst)
    return Path(src).name, tot, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--wells", default="97-192")
    ap.add_argument("--keep-every", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=96)
    a = ap.parse_args()
    lo, hi = map(int, a.wells.split("-"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for w in range(lo, hi + 1):
        src = Path(a.src) / f"VDJ_{w}.fastq"
        dst = out / f"VDJ_{w}.fastq.gz"
        if src.exists() and not dst.exists():
            jobs.append((str(src), str(dst), a.keep_every))
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, tot, kept in ex.map(subset_well, *zip(*jobs)) if jobs else []:
            rows.append({"well_file": name, "reads": tot, "kept": kept})
            print(name, tot, kept, flush=True)
    man = out / "manifest.json"
    prev = json.loads(man.read_text())["wells"] if man.exists() else []
    man.write_text(json.dumps({"src": str(Path(a.src).resolve()), "wells": prev + rows,
                               "keep_every": a.keep_every, "rule": "crc32(corrected_barcode) % N == 0"}, indent=1))


if __name__ == "__main__":
    main()
