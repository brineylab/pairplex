import json

from pairplex.provenance import RunLog, collect_provenance, describe_inputs, git_info, run_log_path
from pairplex.version import __version__


def test_runlog_started_then_completed(tmp_path):
    inp = tmp_path / "well001.fastq"
    inp.write_text("@r\nACGT\n+\nIIII\n")
    path = run_log_path(tmp_path / "logs")
    log = RunLog(path, tool="pairplex run", parameters={"output_directory": tmp_path, "min_cluster_umis": 1},
                 inputs=[inp])
    rec = json.loads(path.read_text())
    assert rec["status"] == "started" and rec["finished_utc"] is None
    assert rec["pairplex_version"] == __version__
    assert rec["parameters"] == {"output_directory": str(tmp_path), "min_cluster_umis": 1}
    assert rec["inputs"][0]["bytes"] == inp.stat().st_size
    log.finish(stats={"pairs": 7})
    rec = json.loads(path.read_text())
    assert rec["status"] == "completed" and rec["stats"] == {"pairs": 7} and rec["finished_utc"]


def test_run_log_path_never_overwrites(tmp_path):
    a = run_log_path(tmp_path)
    a.write_text("{}")
    assert run_log_path(tmp_path) != a


def test_git_info_outside_checkout_is_none(tmp_path):
    assert git_info(tmp_path) is None


def test_provenance_fields():
    p = collect_provenance()
    for k in ("pairplex_version", "git", "python", "dependencies", "argv", "hostname"):
        assert k in p
    assert "polars" in p["dependencies"]


def test_describe_missing_input(tmp_path):
    assert describe_inputs([tmp_path / "nope"])[0]["bytes"] is None
