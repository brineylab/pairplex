import gzip, json
from pathlib import Path
import polars as pl, pytest
from simplex.run import run
def _inp(tmp,n=60):
    d={"sequence_id:0":[f"h{i}" for i in range(n)],"sequence:0":["GATTACA"*30]*n,
       "sequence_id:1":[f"l{i}" for i in range(n)],"sequence:1":["CCGGTA"*30]*n,
       "name":[f"c{i}" for i in range(n)],"locus:0":["IGH"]*n,"locus:1":["IGK"]*n}
    p=tmp/"in.parquet"; pl.DataFrame(d).write_parquet(p); return p
def test_outputs_and_manifest(tmp_path):
    out=tmp_path/"o"; run(input_data=_inp(tmp_path),output_directory=out,wells=4,cells_per_droplet_mean=1,cells_per_droplet_sd=0,variable_length=False,seed=0)
    assert list((out/"reads").glob("*.fastq.gz"))
    for f in ["truth_components","truth_cells","truth_barcodes"]: assert (out/"truth"/f"{f}.parquet").exists()
    man=json.loads((out/"run_manifest.json").read_text())
    assert "input_fingerprint" in man and man["counts"]["reads"]>0 and "polars" in man
def test_refuses_nonempty(tmp_path):
    out=tmp_path/"o"; run(input_data=_inp(tmp_path),output_directory=out,wells=4,seed=0)
    with pytest.raises(FileExistsError): run(input_data=_inp(tmp_path),output_directory=out,wells=4,seed=0)
def test_reproducible(tmp_path):
    def content(d): return sorted(gzip.open(p,"rt").read() for p in Path(d).glob("*.fastq.gz"))
    a=run(input_data=_inp(tmp_path),output_directory=tmp_path/"a",wells=4,seed=5)
    b=run(input_data=_inp(tmp_path),output_directory=tmp_path/"b",wells=4,seed=5)
    assert content(a)==content(b)
def test_zero_recovery_run(tmp_path):  # full-pipeline zero-read case must not raise
    out=tmp_path/"z"; run(input_data=_inp(tmp_path),output_directory=out,wells=4,recovery_rate=0.0,seed=0)
    assert (out/"truth"/"truth_components.parquet").exists()
    assert pl.read_parquet(out/"truth"/"truth_components.parquet").height==0
def test_zero_survival_run(tmp_path):
    out=tmp_path/"z2"; run(input_data=_inp(tmp_path),output_directory=out,wells=4,molecule_survival_rate=0.0,seed=0)
    assert (out/"truth"/"truth_barcodes.parquet").exists()
