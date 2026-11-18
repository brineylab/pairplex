"""Regenerate the golden expected tables. Run ONLY when an output change is intended and reviewed.

    python pairplex/tests/data/golden/make_golden.py      # from the repo root

Records the code commit and the abstar version used. Use legacy mode if the checkout has it, so
the golden tables always describe the reference (pre-v2) behaviour.
"""

import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import pairplex.pairplex as ppx  # noqa: E402
from pairplex.tests.golden import EXPECTED, abstar_version, extract, run_pairplex  # noqa: E402


def main():
    kwargs = {"legacy": True} if "legacy" in inspect.signature(ppx.run).parameters else {}
    with tempfile.TemporaryDirectory() as tmp:
        tables = extract(run_pairplex(Path(tmp), **kwargs))
    EXPECTED.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.write_parquet(EXPECTED / f"{name}.parquet")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (EXPECTED / "versions.json").write_text(json.dumps(
        {"code_commit": commit, "abstar": abstar_version(), "run_kwargs": kwargs,
         "rows": {k: v.height for k, v in tables.items()}}, indent=1))
    print({k: v.height for k, v in tables.items()})


if __name__ == "__main__":
    main()
