import polars as pl
from simplex.scoring import score
def _truth(tmp):
    td=tmp/"truth"; td.mkdir()
    pl.DataFrame({"final_well":[0,0],"barcode":["X","X"],"origin_cell_id":[0,0],"source_pair_id":["A","A"],
        "chain":[0,1],"locus":["IGH","IGK"],"sequence":["H_A"*20,"L_A"*20],"is_resident_source":[True,True],
        "n_source_molecules":[3,3],"n_umis":[3,3],"n_reads":[9,9],"n_reads_resident":[9,9],
        "n_reads_free":[0,0],"n_reads_index_hopped":[0,0]}).write_parquet(td/"truth_components.parquet")
    pl.DataFrame({"well":[0],"barcode":["X"],"n_resident_cells":[1],"is_collision":[False],"is_ambient_only":[False],
        "n_sequenced_both_resident_cells":[1],"n_reference_pairable_resident_cells":[1]}).write_parquet(td/"truth_barcodes.parquet")
    return td
def _pp(tmp, s0="H_A"*20, s1="L_A"*20, bc="X", loc0="IGH", loc1="IGK"):
    p=tmp/"annotated"; p.mkdir(exist_ok=True)
    pl.DataFrame({"name":[f"{bc}_d0_w0"],"well":["0"],"sequence_id:0":[f"{bc}_contig-0"],"sequence:0":[s0],
        "locus:0":[loc0],"sequence_id:1":[f"{bc}_contig-1"],"sequence:1":[s1],"locus:1":[loc1]}).write_parquet(p/"w_paired.parquet")
    return tmp
def test_correct(tmp_path):
    ps,_=score(_pp(tmp_path),_truth(tmp_path)); r=ps.to_dicts()[0]
    assert r["pairing_status"]=="correct" and r["origin_status"]=="resident"
def test_orientation_agnostic(tmp_path):
    # swap the output columns AND lie about loci; truth-based orientation must still resolve
    ps,_=score(_pp(tmp_path, s0="L_A"*20, s1="H_A"*20, loc0="IGK", loc1="IGH"), _truth(tmp_path))
    assert ps.to_dicts()[0]["pairing_status"]=="correct"
def _pp_realname(tmp, s0="H_A"*20, s1="L_A"*20, bc="X", loc0="IGH", loc1="IGK"):
    # mimic real PairPlex merged-mode output: well encoded ONLY in the filename, no "well" column
    p=tmp/"annotated"; p.mkdir(exist_ok=True)
    pl.DataFrame({"name":[f"{bc}_d0_w0"],"sequence_id:0":[f"{bc}_contig-0"],"sequence:0":[s0],
        "locus:0":[loc0],"sequence_id:1":[f"{bc}_contig-1"],"sequence:1":[s1],"locus:1":[loc1]}).write_parquet(p/"well000.fastq_paired.parquet")
    return tmp
def test_well_derived_from_filename(tmp_path):
    # no "well" column at all; scorer must parse well 0 from "well000.fastq_paired.parquet"
    ps,_=score(_pp_realname(tmp_path),_truth(tmp_path)); r=ps.to_dicts()[0]
    assert r["well"]==0
    assert r["pairing_status"]=="correct" and r["origin_status"]=="resident"
def test_missing_key(tmp_path):
    ps,ks=score(_pp(tmp_path, bc="Z"), _truth(tmp_path))
    assert ps.to_dicts()[0]["key_status"]=="unknown"
    assert ks.filter((pl.col("well")==0)&(pl.col("barcode")=="X")).to_dicts()[0]["output_status"]=="missing"
