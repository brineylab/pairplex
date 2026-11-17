"""Logging + progress-bar helpers for SimPlex.

Every long-running SimPlex entry point (`run`, `score`) uses these so a run is observable:
an INFO log line per stage (with counts) to trace progress and locate stalls, `tqdm`
progress bars on the per-item loops, and `verbose`/`quiet` switches. Import `logger` for
messages, `configure_logging()` to set the level once at an entry point, and `pbar()` for a
progress bar that respects the `quiet` switch.
"""
import logging
import sys

logger = logging.getLogger("simplex")


def configure_logging(quiet: bool = False, verbose: bool = False) -> logging.Logger:
    """Set the `simplex` logger level for one run and attach a stderr handler once.

    `quiet` -> WARNING (silence stage logs), `verbose` -> DEBUG (per-stage detail),
    otherwise INFO. Returns the package logger.
    """
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s | simplex | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
    logger.propagate = False
    return logger


def pbar(iterable=None, *, total=None, desc=None, quiet: bool = False):
    """A `tqdm.auto` progress bar that is disabled when `quiet` is True.

    Usable as `for x in pbar(items, desc=...)` or, with `iterable=None`, as a manual bar
    (`with pbar(total=n, desc=...) as bar: bar.update(1)`).
    """
    from tqdm.auto import tqdm
    return tqdm(iterable, total=total, desc=desc, disable=quiet, leave=False)
