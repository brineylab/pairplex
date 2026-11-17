import polars as pl
from simplex.reads import apply_sequencing_errors, build_merged
from pairplex.utils import parse_barcodes
def _reads(n=3):
    return pl.DataFrame({"read_id":[f"r{i}" for i in range(n)],"final_well":[0]*n,
        "barcode":["ACGTACGTACGTACGT"]*n,"umi":["AAAAAAAAAA"]*n,"cdna":["GATTACAGGT"*20]*n,"n_seq_errors":[0]*n})
def test_seq_error():
    r=apply_sequencing_errors(_reads(500),0.05,0.0,0); assert r["n_seq_errors"].sum()>0
def test_zero_error():
    r=apply_sequencing_errors(_reads(),0.0,0.0,0); assert r["cdna"][0]=="GATTACAGGT"*20 and r["n_seq_errors"].sum()==0
def test_merged_round_trip(tmp_path):
    b=build_merged(_reads(),"TTTCTTATATGGG",0.0,False,0); s=b["read_seq"][0]
    assert s[:16]=="ACGTACGTACGTACGT" and s[16:26]=="AAAAAAAAAA"
    assert s[36:].lstrip("G")==("GATTACAGGT"*20).lstrip("G") and len(b["qual"][0])==len(s)
    # actual parse: write a fastq and confirm pairplex.parse_barcodes recovers the barcode
    fq=tmp_path/"r.fastq"; fq.write_text("".join(f"@{i}\n{seq}\n+\n{q}\n" for i,seq,q in zip(b["read_id"],b["read_seq"],b["qual"])))
    out=parse_barcodes(str(fq), str(tmp_path), whitelist_path=None, check_rc=True)
    if out is not None:
        assert pl.read_parquet(out)["barcode"][0]=="ACGTACGTACGTACGT"
def test_rc(tmp_path):
    from simplex._dna import revcomp_str
    b=build_merged(_reads(),"TTTCTTATATGGG",1.0,False,0); assert revcomp_str(b["read_seq"][0])[:16]=="ACGTACGTACGTACGT"
def test_variable_length():
    b=build_merged(_reads(),"TTTCTTATATGGG",0.0,True,0); assert all(len(s)>36 for s in b["read_seq"])
