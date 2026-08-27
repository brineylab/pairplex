import polars as pl
from simplex.truth import build_truth_components, build_truth_cells, build_truth_barcodes
from simplex._contract import REF_MIN_READS, REF_MIN_UMIS
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

# ---------------------------------------------------------------------------
# Resident-cell counts: n_captured_both_resident_cells, n_survived_both_resident_cells,
# n_reference_pairable_resident_cells. Two residents at the same (well,barcode) key:
#   cell 0: captured both chains, survived both chains, both chains sequenced with
#           >=REF_MIN_READS reads and >=REF_MIN_UMIS umis each -> counts toward all three.
#   cell 1: captured both chains but NOT survived on chain1 (no surviving molecule), and
#           only chain0 has reads (chain1 never sequenced) -> counts only toward captured.
# ---------------------------------------------------------------------------
def _rescount_cells():
    return pl.DataFrame({"cell_id":[0,1],"source_pair_id":["A","B"],
        "chain0_id":["h0","h1"],"chain0_seq":["HA","HB"],"chain0_locus":["IGH","IGH"],
        "chain1_id":["l0","l1"],"chain1_seq":["LA","LB"],"chain1_locus":["IGK","IGK"],
        "droplet_id":[0,1],"barcode":["Y","Y"],"resident_well":[0,0]})

def _rescount_status():
    return pl.DataFrame({"cell_id":[0,0,1,1],"chain":[0,1,0,1],
        "captured":[True,True,True,True],"n_molecules":[2,2,1,1]})

def _rescount_mols():
    # cell0: both chains have a surviving molecule. cell1: only chain0 does (chain1 absent -> not survived).
    return pl.DataFrame({"molecule_id":[0,1,2],"origin_cell_id":[0,0,1],"chain":[0,1,0],"survived":[True,True,True]})

def _rescount_reads():
    n=REF_MIN_READS
    rows=[]
    def add(cell,chain,src,locus,n_reads):
        for i in range(n_reads):
            rows.append({"read_id":f"c{cell}ch{chain}r{i}","molecule_id":cell*100+chain*10+i,
                "origin_cell_id":cell,"source_pair_id":src,"chain":chain,"locus":locus,
                "barcode":"Y","final_well":0,"is_free":False,"is_index_hopped":False,
                "umi":f"u{cell}_{chain}_{i}"})
    add(0,0,"A","IGH",n)          # cell0 chain0: n reads, n distinct umis (>=REF_MIN_READS, >=REF_MIN_UMIS)
    add(0,1,"A","IGK",n)          # cell0 chain1: same
    add(1,0,"B","IGH",n)          # cell1 chain0 only: chain1 never sequenced
    return pl.DataFrame(rows)

def test_barcodes_resident_cell_counts():
    cells,status,mols,reads=_rescount_cells(),_rescount_status(),_rescount_mols(),_rescount_reads()
    comp=build_truth_components(cells,reads)
    tc=build_truth_cells(cells,status,mols,reads)
    tb=build_truth_barcodes(cells,tc,comp)
    x=tb.filter((pl.col("well")==0)&(pl.col("barcode")=="Y")).to_dicts()[0]
    assert x["n_captured_both_resident_cells"]==2      # both cells captured on both chains
    assert x["n_survived_both_resident_cells"]==1       # only cell0 survived on both chains
    assert x["n_reference_pairable_resident_cells"]==1  # only cell0 has both chains >=REF_MIN_READS/UMIS

# ---------------------------------------------------------------------------
# Per-locus dominance + tie flags. At one key:
#   heavy (IGH): source SA and SB tie at 5 reads each -> heavy_dominance_is_tied_n_reads True.
#   light (IGK): source SA (10 reads) clearly beats SB (2 reads) -> no tie, SA is dominant.
# ---------------------------------------------------------------------------
def _dom_cells():
    return pl.DataFrame({"cell_id":[10,11],"source_pair_id":["SA","SB"],
        "chain0_id":["h10","h11"],"chain0_seq":["HSA","HSB"],"chain0_locus":["IGH","IGH"],
        "chain1_id":["l10","l11"],"chain1_seq":["LSA","LSB"],"chain1_locus":["IGK","IGK"],
        "droplet_id":[1,1],"barcode":["Z","Z"],"resident_well":[0,0]})

def _dom_status():
    return pl.DataFrame({"cell_id":[10,10,11,11],"chain":[0,1,0,1],
        "captured":[True,True,True,True],"n_molecules":[1,1,1,1]})

def _dom_mols():
    return pl.DataFrame({"molecule_id":[0,1,2,3],"origin_cell_id":[10,10,11,11],
        "chain":[0,1,0,1],"survived":[True,True,True,True]})

def _dom_reads():
    rows=[]
    def add(cell,chain,src,locus,n_reads):
        for i in range(n_reads):
            rows.append({"read_id":f"c{cell}ch{chain}r{i}","molecule_id":cell*1000+chain*100+i,
                "origin_cell_id":cell,"source_pair_id":src,"chain":chain,"locus":locus,
                "barcode":"Z","final_well":0,"is_free":False,"is_index_hopped":False,
                "umi":f"u{cell}_{chain}_{i}"})
    add(10,0,"SA","IGH",5)   # heavy SA: 5 reads
    add(10,1,"SA","IGK",10)  # light SA: 10 reads (clear winner)
    add(11,0,"SB","IGH",5)   # heavy SB: 5 reads (tie with SA)
    add(11,1,"SB","IGK",2)   # light SB: 2 reads
    return pl.DataFrame(rows)

def test_barcodes_dominance_tie_flags():
    cells,status,mols,reads=_dom_cells(),_dom_status(),_dom_mols(),_dom_reads()
    comp=build_truth_components(cells,reads)
    tc=build_truth_cells(cells,status,mols,reads)
    tb=build_truth_barcodes(cells,tc,comp)
    x=tb.filter((pl.col("well")==0)&(pl.col("barcode")=="Z")).to_dicts()[0]
    assert x["heavy_dominance_is_tied_n_reads"] is True
    assert x["dominant_light_source_by_reads"]=="SA"
    assert x["light_dominance_is_tied_n_reads"] is False

# ---------------------------------------------------------------------------
# Clonal aggregation: a single source_pair_id ("C") split across two origin_cell_ids at the
# same key must have its read support SUMMED before ranking, so it can beat a single-cell
# competitor ("D") that individually outguns either C-cell but not their combined total.
#   C: cell20 (3 reads) + cell21 (3 reads) = 6 total
#   D: cell22 (5 reads) = 5 total
# -> dominant_heavy_source_by_reads == "C", and it is NOT a tie (6 != 5).
# ---------------------------------------------------------------------------
def _clonal_cells():
    return pl.DataFrame({"cell_id":[20,21,22],"source_pair_id":["C","C","D"],
        "chain0_id":["h20","h21","h22"],"chain0_seq":["H20","H21","H22"],"chain0_locus":["IGH","IGH","IGH"],
        "chain1_id":["l20","l21","l22"],"chain1_seq":["L20","L21","L22"],"chain1_locus":["IGK","IGK","IGK"],
        "droplet_id":[2,3,4],"barcode":["W","W","W"],"resident_well":[0,0,0]})

def _clonal_status():
    return pl.DataFrame({"cell_id":[20,20,21,21,22,22],"chain":[0,1,0,1,0,1],
        "captured":[True,True,True,True,True,True],"n_molecules":[1,1,1,1,1,1]})

def _clonal_mols():
    return pl.DataFrame({"molecule_id":[0,1,2],"origin_cell_id":[20,21,22],"chain":[0,0,0],"survived":[True,True,True]})

def _clonal_reads():
    rows=[]
    def add(cell,chain,src,locus,n_reads):
        for i in range(n_reads):
            rows.append({"read_id":f"c{cell}ch{chain}r{i}","molecule_id":cell*1000+chain*100+i,
                "origin_cell_id":cell,"source_pair_id":src,"chain":chain,"locus":locus,
                "barcode":"W","final_well":0,"is_free":False,"is_index_hopped":False,
                "umi":f"u{cell}_{chain}_{i}"})
    add(20,0,"C","IGH",3)   # clone C, cell20: 3 reads
    add(21,0,"C","IGH",3)   # clone C, cell21: 3 reads (same source_pair_id, different cell)
    add(22,0,"D","IGH",5)   # single-cell competitor D: 5 reads (beats either C-cell alone, not combined)
    return pl.DataFrame(rows)

def test_barcodes_clonal_aggregation_wins_dominance():
    cells,status,mols,reads=_clonal_cells(),_clonal_status(),_clonal_mols(),_clonal_reads()
    comp=build_truth_components(cells,reads)
    tc=build_truth_cells(cells,status,mols,reads)
    tb=build_truth_barcodes(cells,tc,comp)
    x=tb.filter((pl.col("well")==0)&(pl.col("barcode")=="W")).to_dicts()[0]
    assert x["dominant_heavy_source_by_reads"]=="C"    # 6 (aggregated) beats D's 5
    assert x["heavy_dominance_is_tied_n_reads"] is False
