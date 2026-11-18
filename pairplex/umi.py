# Copyright (c) 2025 Benjamin Nemoz
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

# This file is part of PairPlex.
# PairPlex is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# PairPlex is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with PairPlex. If not, see <http://www.gnu.org/licenses/>.

"""UMI error correction.

PairPlex collapses UMIs within one (well, droplet barcode) group with the *directional*
adjacency method of UMI-tools (Smith, Heger & Sudbery, Genome Res. 2017):

- nodes are the distinct raw UMIs, weighted by their read count;
- there is a directed edge ``a -> b`` when ``Hamming(a, b) == 1`` and
  ``count(a) >= 2 * count(b) - 1``;
- UMIs are visited in descending count; each UMI not yet absorbed becomes a representative and
  absorbs every UMI reachable from it along directed edges, transitively: ``a -> b -> c`` puts
  ``c`` under ``a`` although ``c`` is two substitutions away from ``a``. A UMI reached from an
  earlier representative is never re-assigned.

Ties in count are broken by the UMI string (lexicographic), so the result is deterministic.
UMI-tools leaves tie order to dict order; the groups are the same, only which tied UMI is named
representative can differ.

UMIs containing ``N``: ``N`` counts as a mismatch at its position. An N-containing UMI never
absorbs anything (it would be Hamming-1 from up to four different real UMIs, so it is no evidence
for any of them), but it can be absorbed by an N-free UMI at Hamming distance 1 that satisfies the
count rule (e.g. ``ACGTNCGTAC`` under ``ACGTACGTAC``). An N-containing UMI with no such parent
stays its own UMI. Characters other than ``ACGTN`` are compared literally and never generate
neighbours.

Neighbour generation (``4 * L`` single substitutions over ``ACGTN`` per UMI) keeps the cost
linear in the number of distinct UMIs, so barcodes with thousands of reads are cheap.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

__all__ = ["collapse_umis_directional", "collapse_umi_column"]

_ALPHABET = "ACGTN"


def _neighbours(umi: str):
    """All strings at Hamming distance 1 from ``umi`` over ``ACGTN`` (positions with other
    characters are skipped)."""
    for i, c in enumerate(umi):
        if c not in _ALPHABET:
            continue
        head, tail = umi[:i], umi[i + 1 :]
        for b in _ALPHABET:
            if b != c:
                yield head + b + tail


def collapse_umis_directional(
    umis: Iterable[str] | Mapping[str, int],
    counts: Iterable[int] | None = None,
) -> dict[str, str]:
    """
    Directional UMI collapsing (UMI-tools) within one group of reads.

    Parameters
    ----------
    umis : iterable of str, or mapping UMI -> count
        Distinct raw UMIs. If a mapping is given, its values are the counts and ``counts`` must be
        ``None``. If ``counts`` is ``None`` and ``umis`` is a plain iterable, it is treated as one
        entry per READ and counted (duplicates allowed).

    counts : iterable of int, optional
        Reads per raw UMI, aligned with ``umis`` (which must then be distinct).

    Returns
    -------
    dict[str, str]
        Mapping from every input UMI to its representative (a representative maps to itself).
    """
    if isinstance(umis, Mapping):
        if counts is not None:
            raise ValueError("pass counts either in the mapping or in `counts`, not both")
        cnt = dict(umis)
    elif counts is None:
        cnt = dict(Counter(umis))
    else:
        umis = list(umis)
        counts = list(counts)
        if len(umis) != len(counts):
            raise ValueError("umis and counts must have the same length")
        cnt = {}
        for u, c in zip(umis, counts):
            if u in cnt:
                raise ValueError(f"duplicate UMI with explicit counts: {u}")
            cnt[u] = int(c)

    order = sorted(cnt, key=lambda u: (-cnt[u], u))
    rep: dict[str, str] = {}
    for root in order:
        if root in rep:
            continue
        rep[root] = root
        stack = [root]
        while stack:
            parent = stack.pop()
            if "N" in parent:  # N-containing UMIs never absorb
                continue
            threshold = cnt[parent]
            for nb in _neighbours(parent):
                c = cnt.get(nb)
                if c is None or nb in rep:
                    continue
                if threshold >= 2 * c - 1:
                    rep[nb] = root
                    stack.append(nb)
    return rep


def collapse_umi_column(umis: list[str]) -> list[str]:
    """Per-read convenience wrapper: corrected UMI for each read UMI in ``umis`` (one group)."""
    mapping = collapse_umis_directional(umis)
    return [mapping[u] for u in umis]
