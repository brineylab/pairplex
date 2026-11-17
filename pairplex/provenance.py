"""Run provenance: record which code, dependencies, parameters and inputs produced an output.

Every ``pairplex.run`` writes ``<output>/logs/pairplex_run_<UTC timestamp>.json`` when it starts
(``status: "started"``) and rewrites it when it finishes (``status: "completed"``, with totals). A
log left at ``"started"`` therefore marks a run that crashed or was killed. ``pair-pooled`` writes
``<output stem>_run_info.json`` next to its parquet.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .version import __version__

DEPENDENCIES = ("abutils", "abstar", "polars", "numpy", "pandas", "edlib", "pyspoa", "natsort")


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(path: Path | None = None) -> dict | None:
    """Commit, branch, describe and dirty flag of the checkout containing ``path`` (default: this
    package). None when the package is not installed from a git checkout."""
    path = Path(path or Path(__file__).resolve().parent)
    commit = _run(["git", "rev-parse", "HEAD"], cwd=path)
    if commit is None:
        return None
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=path)
    return {
        "commit": commit,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=path),
        "describe": _run(["git", "describe", "--tags", "--always", "--dirty"], cwd=path),
        "dirty": bool(status),
    }


def dependency_versions() -> dict:
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    mmseqs = shutil.which("mmseqs")
    versions["mmseqs"] = _run([mmseqs, "version"]) if mmseqs else None
    return versions


def collect_provenance() -> dict:
    """Everything about the software and machine; no run-specific content."""
    return {
        "pairplex_version": __version__,
        "git": git_info(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "dependencies": dependency_versions(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
    }


def describe_inputs(files) -> list[dict]:
    """Name, size and modification time of each input (no hashing: inputs can be very large)."""
    out = []
    for f in files:
        p = Path(f)
        try:
            st = p.stat()
            out.append({"path": str(p.resolve()), "bytes": st.st_size,
                        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")})
        except OSError:
            out.append({"path": str(p), "bytes": None, "mtime": None})
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunLog:
    """A JSON run log, written at start and finalized at the end.

    >>> log = RunLog(path, tool="pairplex run", parameters={...}, inputs=[...])
    >>> ...                                # do the work
    >>> log.finish(stats={"pairs": 123})
    """

    def __init__(self, path: str | Path, tool: str, parameters: dict, inputs=None):
        self.path = Path(path)
        self.record = {
            "tool": tool,
            "status": "started",
            "started_utc": _now(),
            "finished_utc": None,
            "parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in parameters.items()},
            "inputs": describe_inputs(inputs or []),
            **collect_provenance(),
            "stats": None,
        }
        self._write()

    def finish(self, stats: dict | None = None, status: str = "completed") -> Path:
        self.record.update(status=status, finished_utc=_now(), stats=stats)
        self._write()
        return self.path

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.record, indent=2, default=str))
        tmp.replace(self.path)


def run_log_path(log_directory: str | Path, prefix: str = "pairplex_run") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(log_directory) / f"{prefix}_{stamp}.json"
    i = 1
    while path.exists():
        path = Path(log_directory) / f"{prefix}_{stamp}_{i}.json"
        i += 1
    return path
