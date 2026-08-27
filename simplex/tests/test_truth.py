import polars as pl
from simplex.truth import build_truth_components, build_truth_cells, build_truth_barcodes
def _cells():
    return pl.DataFrame({"cell_id":[0,1],"source_pair_id":["A","B"],
        "chain0_id":["hA","hB"],"chain0_seq":["HA","HB"],"chain0_locus":["IGH","IGH"],
        "chain1_id":["lA","lB"],"chain1_seq":["LA","LB"],"chain1_locus":["IGK","IGK"],
        "droplet_id":[0,0],"barcode":["X","X"],"resident_well":[0,0]})   # A,B collide on X@well0
def _status():
    return pl.DataFrame({"cell_id":[0,0,1,1],"chain":[0,1,0,1],"captured":[True,True,True,False],"n_molecules":[2,2,1,0]})
def _mols():
    return pl.DataFrame({"molecule_id":[0,1,2],"origin_cell_id":[0,0,1],"chain":[0,1,0],"survived":[True,True,True]})
def _reads():   # only cell0 produced reads; cell1 read-less but physically resident
    return pl.DataFrame({"read_id":["r0","r1"],"molecule_id":[0,1],"origin_cell_id":[0,0],"source_pair_id":["A","A"],
        "chain":[0,1],"locus":["IGH","IGK"],"barcode":["X","X"],"final_well":[0,0],
        "is_free":[False,False],"is_index_hopped":[False,False],"umi":["u0","u1"]})
def test_occupancy_from_cells():
    comp=build_truth_components(_cells(),_reads())
    tc=build_truth_cells(_cells(),_status(),_mols(),_reads())
    tb=build_truth_barcodes(_cells(),tc,comp)
    x=tb.filter((pl.col("well")==0)&(pl.col("barcode")=="X")).to_dicts()[0]
    assert x["n_resident_cells"]==2 and x["is_collision"] is True          # both counted incl read-less B
    assert x["n_sequenced_both_resident_cells"]==1                          # only A sequenced both chains
def test_cells_capture():
    tc=build_truth_cells(_cells(),_status(),_mols(),_reads())
    assert tc.filter(pl.col("cell_id")==1).to_dicts()[0]["captured_1"] is False
