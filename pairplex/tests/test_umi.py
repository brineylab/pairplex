import itertools
import random

import polars as pl
import pytest

from pairplex.umi import collapse_umi_column, collapse_umis_directional
from pairplex.utils import process_droplet


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def _reference_directional(cnt):
    """O(n^2) reference implementation, written the UMI-tools way: build the full directed
    adjacency list, BFS components from nodes by descending count (ties by UMI), first come first
    served, N-containing UMIs never absorb."""
    umis = sorted(cnt, key=lambda u: (-cnt[u], u))
    adj = {u: [] for u in umis}
    for a, b in itertools.permutations(umis, 2):
        if "N" in a or len(a) != len(b):
            continue
        if _hamming(a, b) == 1 and cnt[a] >= 2 * cnt[b] - 1:
            adj[a].append(b)
    rep = {}
    for root in umis:
        if root in rep:
            continue
        seen, queue = {root}, [root]
        while queue:
            n = queue.pop(0)
            for m in adj[n]:
                if m not in seen:
                    seen.add(m)
                    queue.append(m)
        for u in seen:
            rep.setdefault(u, root)
    return rep


# ---------------------------------------------------------------- classic examples


def test_singleton():
    assert collapse_umis_directional({"AAAAAAAAAA": 1}) == {"AAAAAAAAAA": "AAAAAAAAAA"}


def test_empty():
    assert collapse_umis_directional({}) == {}


def test_simple_absorption():
    m = collapse_umis_directional({"AAAAAAAAAA": 10, "AAAAAAAAAC": 2})
    assert m == {"AAAAAAAAAA": "AAAAAAAAAA", "AAAAAAAAAC": "AAAAAAAAAA"}


def test_count_rule_boundary():
    # 2*count(b) - 1: a=5, b=3 -> 5 >= 5 absorbs; a=5, b=4 -> 5 >= 7 is false
    assert collapse_umis_directional({"AAAAAAAAAA": 5, "AAAAAAAAAC": 3})["AAAAAAAAAC"] == "AAAAAAAAAA"
    assert collapse_umis_directional({"AAAAAAAAAA": 5, "AAAAAAAAAC": 4})["AAAAAAAAAC"] == "AAAAAAAAAC"


def test_hamming_two_not_absorbed():
    m = collapse_umis_directional({"AAAAAAAAAA": 100, "AAAAAAAACC": 1})
    assert m["AAAAAAAACC"] == "AAAAAAAACC"


def test_chain_is_transitive():
    # a(9) -> b(4) -> c(1): c is 2 substitutions from a, still absorbed through b
    a, b, c = "AAAAAAAAAA", "AAAAAAAAAC", "AAAAAAAACC"
    m = collapse_umis_directional({a: 9, b: 4, c: 1})
    assert m == {a: a, b: a, c: a}


def test_chain_broken_by_count_rule():
    # a(9) -> b(4), but b(4) vs c(3): 4 >= 5 false, so c stays separate
    a, b, c = "AAAAAAAAAA", "AAAAAAAAAC", "AAAAAAAACC"
    m = collapse_umis_directional({a: 9, b: 4, c: 3})
    assert m == {a: a, b: a, c: c}


def test_ties_of_one_merge_deterministically():
    # 1 >= 2*1 - 1: two singleton UMIs at Hamming 1 merge; representative is the smaller string
    m = collapse_umis_directional({"AAAAAAAAAC": 1, "AAAAAAAAAA": 1})
    assert m == {"AAAAAAAAAA": "AAAAAAAAAA", "AAAAAAAAAC": "AAAAAAAAAA"}
    # order of input does not matter
    assert m == collapse_umis_directional({"AAAAAAAAAA": 1, "AAAAAAAAAC": 1})


def test_ties_above_one_do_not_merge():
    m = collapse_umis_directional({"AAAAAAAAAA": 3, "AAAAAAAAAC": 3})
    assert m == {"AAAAAAAAAA": "AAAAAAAAAA", "AAAAAAAAAC": "AAAAAAAAAC"}


def test_umi_tools_paper_example():
    # Smith et al. 2017 Fig. 1 style: one true UMI with errors, one independent UMI nearby
    cnt = {
        "ACGTACGTAC": 456,  # true
        "ACGTACGTAA": 72,  # error of the true one
        "ACGTACGTTA": 2,  # error of the error (chain)
        "ACGTACGTCC": 90,  # error of the true one
        "TTTTACGTAC": 30,  # independent
        "TTTTACGTAA": 6,  # error of the independent one
    }
    m = collapse_umis_directional(cnt)
    assert {u for u, r in m.items() if r == "ACGTACGTAC"} == {
        "ACGTACGTAC", "ACGTACGTAA", "ACGTACGTTA", "ACGTACGTCC"
    }
    assert m["TTTTACGTAA"] == "TTTTACGTAC"
    assert len(set(m.values())) == 2


def test_node_claimed_by_first_root():
    # c is reachable from both a(10) and b(6, not reachable from a); a is processed first
    a, b, c = "AAAAAAAAAA", "AAAAAAAACC", "AAAAAAAAAC"
    m = collapse_umis_directional({a: 10, b: 6, c: 1})
    assert m == {a: a, b: b, c: a}


# ---------------------------------------------------------------- N handling


def test_n_umi_absorbed_but_never_absorbs():
    real, nn = "ACGTACGTAC", "ACGTNCGTAC"
    assert collapse_umis_directional({real: 5, nn: 1})[nn] == real
    # an abundant N UMI does not absorb an N-free neighbour
    m = collapse_umis_directional({nn: 50, real: 1})
    assert m == {nn: nn, real: real}


def test_n_umi_respects_count_rule_and_double_n():
    real = "ACGTACGTAC"
    assert collapse_umis_directional({real: 2, "ACGTNCGTAC": 2})["ACGTNCGTAC"] == "ACGTNCGTAC"
    assert collapse_umis_directional({real: 50, "ACGTNNGTAC": 1})["ACGTNNGTAC"] == "ACGTNNGTAC"


def test_short_umis_only_compared_at_equal_length():
    m = collapse_umis_directional({"ACGTACGTAC": 10, "ACGTACGTA": 1})
    assert m["ACGTACGTA"] == "ACGTACGTA"


# ---------------------------------------------------------------- API forms


def test_api_forms_agree():
    reads = ["AAAAAAAAAA"] * 4 + ["AAAAAAAAAC"] + ["GGGGGGGGGG"] * 2
    by_reads = collapse_umis_directional(reads)
    by_map = collapse_umis_directional({"AAAAAAAAAA": 4, "AAAAAAAAAC": 1, "GGGGGGGGGG": 2})
    by_lists = collapse_umis_directional(["AAAAAAAAAA", "AAAAAAAAAC", "GGGGGGGGGG"], [4, 1, 2])
    assert by_reads == by_map == by_lists
    assert collapse_umi_column(reads) == ["AAAAAAAAAA"] * 5 + ["GGGGGGGGGG"] * 2
    with pytest.raises(ValueError):
        collapse_umis_directional(["A", "A"], [1, 2])
    with pytest.raises(ValueError):
        collapse_umis_directional(["A"], [1, 2])


# ---------------------------------------------------------------- randomized vs reference


@pytest.mark.parametrize("seed", range(20))
def test_matches_reference_on_random_sets(seed):
    rng = random.Random(seed)
    L = 6  # short UMIs so Hamming-1 collisions are frequent
    true = ["".join(rng.choice("ACGT") for _ in range(L)) for _ in range(rng.randint(1, 15))]
    cnt = {}
    for t in true:
        n = rng.randint(1, 60)
        for _ in range(n):
            u = list(t)
            if rng.random() < 0.25:
                u[rng.randrange(L)] = rng.choice("ACGTN" if rng.random() < 0.9 else "N")
            u = "".join(u)
            cnt[u] = cnt.get(u, 0) + 1
    assert collapse_umis_directional(cnt) == _reference_directional(cnt)


def test_large_barcode_is_fast():
    import time

    rng = random.Random(0)
    reads = []
    for _ in range(3000):
        t = "".join(rng.choice("ACGT") for _ in range(10))
        for _ in range(rng.randint(1, 20)):
            u = list(t)
            if rng.random() < 0.2:
                u[rng.randrange(10)] = rng.choice("ACGT")
            reads.append("".join(u))
    t0 = time.perf_counter()
    m = collapse_umis_directional(reads)
    assert time.perf_counter() - t0 < 5
    assert len(set(m.values())) < len(set(reads))


# ---------------------------------------------------------------- integration via process_droplet

HEAVY = "CAGGTGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCTTCTGGATACACC"
LIGHT = "GACATCCAGATGACCCAGTCTCCATCCTCCCTGTCTGCATCTGTAGGAGACAGAGTCACCATCACTTGCCGGGCAAGTCAG"


def _droplet_df():
    rows = []
    # heavy: true UMI A (6 reads) + 1-error copy (1 read) + independent UMI G (3 reads)
    heavy_umis = ["AAAAAAAAAA"] * 6 + ["AAAAAAAAAT"] + ["GGGGGGGGGG"] * 3
    # light: one true UMI (4 reads) + two 1-error copies (1 read each)
    light_umis = ["CCCCCCCCCC"] * 4 + ["CCCCCCCCCA", "CCCCCCCCCT"]
    for i, u in enumerate(heavy_umis):
        rows.append({"umi": u, "barcode": "BC", "seq_id": f"h{i}", "sequence": HEAVY})
    for i, u in enumerate(light_umis):
        rows.append({"umi": u, "barcode": "BC", "seq_id": f"l{i}", "sequence": LIGHT})
    return pl.DataFrame(rows)


def _by_reads(meta):
    return {m["reads"]: m for m in meta}


def test_process_droplet_directional(tmp_path):
    meta = process_droplet(
        "well_BC", _droplet_df(), temp_directory=tmp_path, clustering_threshold=0.9,
        min_cluster_reads=3, min_cluster_umis=2,
    )
    h, l = _by_reads(meta)[10], _by_reads(meta)[6]
    assert (h["umis"], h["umis_raw"]) == (2, 3)
    assert (l["umis"], l["umis_raw"]) == (1, 3)
    # min_cluster_umis filters on corrected counts: light had 3 raw UMIs but only 1 molecule
    assert h["pass_filters"] and not l["pass_filters"]
    assert list(h) == ["name", "reads", "umis", "umis_raw", "cluster_fraction", "consensus", "pass_filters"]


def test_process_droplet_none_is_v2_base_layout(tmp_path):
    meta = process_droplet(
        "well_BC", _droplet_df(), temp_directory=tmp_path, clustering_threshold=0.9,
        min_cluster_reads=3, min_cluster_umis=2, umi_collapse="none",
    )
    h, l = _by_reads(meta)[10], _by_reads(meta)[6]
    assert h["umis"] == 3 and l["umis"] == 3
    assert h["pass_filters"] and l["pass_filters"]
    assert list(h) == ["name", "reads", "umis", "cluster_fraction", "consensus", "pass_filters"]


def test_process_droplet_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError):
        process_droplet("well_BC", _droplet_df(), temp_directory=tmp_path, umi_collapse="cluster")


def test_run_legacy_forces_none(tmp_path, monkeypatch):
    """legacy=True must pass umi_collapse='none' to process_droplet, whatever umi_collapse says."""
    import pairplex.pairplex as pp

    seen = {}

    class _Stop(Exception):
        pass

    def fake_pool(*a, **k):
        raise _Stop

    monkeypatch.setattr(pp, "ProcessPoolExecutor", fake_pool)
    real_runlog = pp.RunLog

    def capture(path, tool, parameters, inputs):
        seen.update(parameters)
        return real_runlog(path, tool=tool, parameters=parameters, inputs=inputs)

    monkeypatch.setattr(pp, "RunLog", capture)
    fq = tmp_path / "w.fastq"
    fq.write_text("@r\nACGT\n+\nIIII\n")
    with pytest.raises(_Stop):
        pp.run(str(fq), tmp_path / "out", quiet=True, legacy=True, umi_collapse="directional")
    assert seen["umi_collapse_effective"] == "none"
    with pytest.raises(_Stop):
        pp.run(str(fq), tmp_path / "out2", quiet=True)
    assert seen["umi_collapse_effective"] == "directional"
    with pytest.raises(ValueError):
        pp.run(str(fq), tmp_path / "out3", quiet=True, umi_collapse="bogus")
