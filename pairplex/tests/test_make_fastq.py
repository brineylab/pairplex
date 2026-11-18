"""`pairplex make-fastq` must call make_fastq.run with its real keyword arguments."""

from unittest import mock

from click.testing import CliRunner

from pairplex import make_fastq
from pairplex.cli.pairplex import cli


class _Proc:
    returncode = 0

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd

    def communicate(self):
        return b"", b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_make_fastq_cli_runs_basecalling(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "SampleSheet.csv").write_text("[Data]\n")
    out = tmp_path / "fastq"
    calls = []

    def popen(cmd, **kwargs):
        calls.append(cmd)
        return _Proc(cmd, **kwargs)

    with mock.patch.object(make_fastq.sp, "Popen", side_effect=popen):
        result = CliRunner().invoke(cli, ["make-fastq", str(run_dir), str(out)])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert len(calls) == 1
    assert calls[0].startswith("bcl2fastq ")
    assert f"--runfolder-dir {run_dir}" in calls[0]
    assert f"--output-dir {out}" in calls[0]
    assert f"--sample-sheet {run_dir / 'SampleSheet.csv'}" in calls[0]


def test_make_fastq_cli_forwards_all_options(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sheet = tmp_path / "RunManifest.csv"
    sheet.write_text("")
    with mock.patch("pairplex.cli.pairplex.run_make_fastq") as run:
        result = CliRunner().invoke(
            cli,
            ["make-fastq", str(run_dir), str(tmp_path / "o"), "--samplesheet", str(sheet),
             "--platform", "element", "--debug"],
        )
    assert result.exit_code == 0, result.output + repr(result.exception)
    run.assert_called_once_with(
        sequence_dir=str(run_dir),
        output_directory=str(tmp_path / "o"),
        samplesheet=str(sheet),
        platform="element",
        debug=True,
    )
