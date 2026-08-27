"""Single-factor statistical tests over REAL pairplex.run() on generated data.

These assert *regime-specific*, *mechanistic* tradeoff directions -- NOT blanket
monotonicity. Each scenario is constructed so one mechanism dominates:

  * ambient extra-contig regime -> a min_cluster_fraction filter removes low-support
    ambient contigs, so it can only *reduce* mispairs (precision up).
  * weak-real-chain regime      -> a real chain has low read support, so a high
    min_cluster_fraction deletes a *real* cluster and *costs* recall (yield down).

Every directional assertion is paired with a nontrivial-effect guard so a test that
passes only because "nothing happened" fails instead. The broad-sweep test asserts
no direction at all -- it only demonstrates the metrics compute across a grid.
"""
import pairplex, polars as pl
from simplex.run import run
from simplex.scoring import score
from simplex._testseqs import many_pairs_parquet


def metrics(ppo, truth):
    """(mispair count, resident-correct recall over singleton reference-pairable keys)."""
    ps, ks = score(ppo, truth)
    mis = int((ps["pairing_status"] == "mispaired").sum())
    # recall over SINGLETON reference-pairable keys with a UNIQUE resident-correct output
    correct_keys = {(r["well"], r["barcode"]) for r in ps.filter(
        (pl.col("pairing_status") == "correct") & (pl.col("origin_status") == "resident")
        & (pl.col("output_status") == "unique") & (pl.col("key_status") == "singleton")).to_dicts()}
    refpair = ks.filter(pl.col("reference_pairable_both") & (pl.col("key_status") == "singleton"))
    recall = sum(1 for r in refpair.to_dicts() if (r["well"], r["barcode"]) in correct_keys) / max(1, refpair.height)
    return mis, recall  # collision-key recovery is a SEPARATE metric (per-cell), not this key-level recall


def report_metrics(ppo, truth):
    """Extended, direction-free metrics used by the broad-sweep report."""
    ps, ks = score(ppo, truth)
    n_out = ps.height
    n_correct = int((ps["pairing_status"] == "correct").sum())
    n_mis = int((ps["pairing_status"] == "mispaired").sum())
    precision = n_correct / n_out if n_out else 0.0
    mispair_rate = n_mis / n_out if n_out else 0.0
    mis, recall = metrics(ppo, truth)
    refpair = ks.filter(pl.col("reference_pairable_both") & (pl.col("key_status") == "singleton"))
    yield_ = n_out / refpair.height if refpair.height else 0.0
    return {"outputs": n_out, "precision": precision, "mispair_rate": mispair_rate,
            "recall": recall, "yield": yield_}


def test_ambient_extra_contig_regime(tmp_path):
    # ambient adds a low-support extra chain: a fraction filter should reduce mispairs AND may raise recall
    inp = many_pairs_parquet(tmp_path, 60); out = tmp_path / "sim"
    rd = run(input_data=inp, output_directory=out, wells=2, cells_per_droplet_mean=2, cells_per_droplet_sd=0,
             recovery_rate=0.9, release_rate=0.2, molecule_survival_rate=1.0, index_hop_rate=0.0,
             sequencing_sub_rate=0.0, variable_length=False, seed=1)
    lo = tmp_path / "lo"; pairplex.run(sequences=str(rd), output_directory=str(lo), min_cluster_reads=3, min_cluster_umis=1, min_cluster_fraction=0.0, quiet=True)
    hi = tmp_path / "hi"; pairplex.run(sequences=str(rd), output_directory=str(hi), min_cluster_reads=3, min_cluster_umis=1, min_cluster_fraction=0.3, quiet=True)
    mis_lo, rec_lo = metrics(lo, out / "truth"); mis_hi, rec_hi = metrics(hi, out / "truth")
    assert (mis_lo, rec_lo) != (mis_hi, rec_hi)   # nontrivial effect
    assert mis_hi <= mis_lo                        # filtering ambient extra contigs reduces mispairs


def test_weak_real_chain_regime(tmp_path):
    # A real chain has low read support (low reads_per_molecule_mean, single cells, negligible
    # ambient). A high min_cluster_fraction deletes a *real* cluster -> the cell loses a chain and
    # drops out of the paired output -> recall falls. This is the precision/yield tradeoff seen from
    # the yield side: filtering that is safe against ambient becomes costly against weak real signal.
    inp = many_pairs_parquet(tmp_path, 60); out = tmp_path / "sim"
    rd = run(input_data=inp, output_directory=out, wells=2, cells_per_droplet_mean=1, cells_per_droplet_sd=0,
             recovery_rate=0.9, release_rate=0.02, molecule_survival_rate=1.0, index_hop_rate=0.0,
             reads_per_molecule_mean=3.0, molecules_per_chain_mean=3.0,
             sequencing_sub_rate=0.0, variable_length=False, seed=2)
    lo = tmp_path / "lo"; pairplex.run(sequences=str(rd), output_directory=str(lo), min_cluster_reads=1, min_cluster_umis=1, min_cluster_fraction=0.0, quiet=True)
    hi = tmp_path / "hi"; pairplex.run(sequences=str(rd), output_directory=str(hi), min_cluster_reads=1, min_cluster_umis=1, min_cluster_fraction=0.4, quiet=True)
    mis_lo, rec_lo = metrics(lo, out / "truth"); mis_hi, rec_hi = metrics(hi, out / "truth")
    assert (mis_lo, rec_lo) != (mis_hi, rec_hi)   # nontrivial effect
    assert rec_hi <= rec_lo                        # aggressive fraction filter deletes real chains -> recall cost


def test_broad_sweep_report(tmp_path):
    # Report-only harness: run a small grid and just record precision / mispair rate / recall / yield.
    # No universal-direction assertion -- only that every metric computes without error across the grid.
    inp = many_pairs_parquet(tmp_path, 60); out = tmp_path / "sim"
    rd = run(input_data=inp, output_directory=out, wells=2, cells_per_droplet_mean=2, cells_per_droplet_sd=0,
             recovery_rate=0.9, release_rate=0.1, molecule_survival_rate=1.0, index_hop_rate=0.0,
             reads_per_molecule_mean=5.0, molecules_per_chain_mean=5.0,
             sequencing_sub_rate=0.0, variable_length=False, seed=3)
    grid = [(3, 1, 0.0), (3, 1, 0.2), (3, 1, 0.4), (5, 2, 0.2)]
    rows = []
    for i, (mcr, mcu, mcf) in enumerate(grid):
        d = tmp_path / f"pp{i}"
        pairplex.run(sequences=str(rd), output_directory=str(d), min_cluster_reads=mcr,
                     min_cluster_umis=mcu, min_cluster_fraction=mcf, quiet=True)
        m = report_metrics(d, out / "truth")
        m["min_cluster_reads"] = mcr; m["min_cluster_umis"] = mcu; m["min_cluster_fraction"] = mcf
        rows.append(m)
        # every metric must be a finite number in a sane range -- the point of the harness
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["mispair_rate"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert m["yield"] >= 0.0
        assert m["outputs"] >= 0
    print("\nbroad sweep (report-only):")
    for m in rows:
        print(f"  mcr={m['min_cluster_reads']} mcu={m['min_cluster_umis']} mcf={m['min_cluster_fraction']}: "
              f"outputs={m['outputs']} precision={m['precision']:.3f} mispair_rate={m['mispair_rate']:.3f} "
              f"recall={m['recall']:.3f} yield={m['yield']:.3f}")
    assert len(rows) == len(grid)   # the whole grid computed without error
