import pairplex, polars as pl
from simplex.barcodes import load_barcodes
from simplex._rng import rng_for
from simplex._fixtures import emit, family
from simplex._testseqs import HEAVY_A, LIGHT_A, HEAVY_B, LIGHT_B
from simplex.scoring import score

BC = load_barcodes("v2", 1, rng_for(0, "fx"))[0]


def _run(rd, ppo):
    pairplex.run(sequences=str(rd), output_directory=str(ppo),
                 min_cluster_reads=1, min_cluster_umis=1, quiet=True)


# ---------------------------------------------------------------------------
# 0. clean golden: N cells, one per barcode, all captured+survived, no free/errors
# ---------------------------------------------------------------------------
def test_clean_golden(tmp_path):
    bcs = load_barcodes("v2", 2, rng_for(1, "fx"))
    cells = pl.DataFrame({"cell_id":[0,1],"source_pair_id":["A","B"],
        "chain0_id":["hA","hB"],"chain0_seq":[HEAVY_A,HEAVY_B],"chain0_locus":["IGH","IGH"],
        "chain1_id":["lA","lB"],"chain1_seq":[LIGHT_A,LIGHT_B],"chain1_locus":["IGK","IGK"],
        "droplet_id":[0,1],"barcode":bcs,"resident_well":[0,0]})
    chain_status = pl.DataFrame({"cell_id":[0,0,1,1],"chain":[0,1,0,1],
        "captured":[True,True,True,True],"n_molecules":[1,1,1,1]})
    molecules = pl.DataFrame({"molecule_id":[0,1,2,3],"origin_cell_id":[0,0,1,1],
        "chain":[0,1,0,1],"survived":[True,True,True,True]})
    reads = pl.concat([
        family(0,0,"A",0,"IGH",HEAVY_A,0,bcs[0],"AAAAAAAAAA",False),
        family(1,0,"A",1,"IGK",LIGHT_A,0,bcs[0],"AACCAACCAA",False),
        family(2,1,"B",0,"IGH",HEAVY_B,0,bcs[1],"GGGGGGGGGG",False),
        family(3,1,"B",1,"IGK",LIGHT_B,0,bcs[1],"GGTTGGTTGG",False)])
    rd = emit(cells,chain_status,molecules,reads,tmp_path/"sim")
    ppo = tmp_path/"pp"; _run(rd,ppo)
    ps,_ = score(ppo,(tmp_path/"sim"/"truth"))
    assert ps.height >= 2
    assert (ps["pairing_status"]=="correct").all()
    assert not ((ps["pairing_status"]=="ambiguous")|(ps["pairing_status"]=="unmatchable")).any()


# ---------------------------------------------------------------------------
# 1. exact ambient mispair (heavy A resident @ well0, free light B -> well0, shared barcode)
# ---------------------------------------------------------------------------
def test_exact_ambient_mispair(tmp_path):
    cells = pl.DataFrame({"cell_id":[0,1],"source_pair_id":["A","B"],
        "chain0_id":["hA","hB"],"chain0_seq":[HEAVY_A,HEAVY_B],"chain0_locus":["IGH","IGH"],
        "chain1_id":["lA","lB"],"chain1_seq":[LIGHT_A,LIGHT_B],"chain1_locus":["IGK","IGK"],
        "droplet_id":[0,0],"barcode":[BC,BC],"resident_well":[0,1]})
    chain_status = pl.DataFrame({"cell_id":[0,0,1,1],"chain":[0,1,0,1],
        "captured":[True,False,False,True],"n_molecules":[1,0,0,1]})   # A: heavy only; B: light only
    molecules = pl.DataFrame({"molecule_id":[0,1],"origin_cell_id":[0,1],"chain":[0,1],"survived":[True,True]})
    reads = pl.concat([family(0,0,"A",0,"IGH",HEAVY_A,0,BC,"AAAAAAAAAA",False),      # A heavy resident @ well0
                       family(1,1,"B",1,"IGK",LIGHT_B,0,BC,"CCCCCCCCCC",True)])       # B light FREE -> well0
    rd = emit(cells,chain_status,molecules,reads,tmp_path/"sim")
    ppo = tmp_path/"pp"; _run(rd,ppo)
    ps,_ = score(ppo,(tmp_path/"sim"/"truth"))
    assert (ps["pairing_status"]=="mispaired").sum() >= 1


# ---------------------------------------------------------------------------
# 2. one-cell negative control: a SINGLE source cell A contributes both its heavy and
#    its light to one (well,barcode) key -- heavy A resident @ (well0,bcA) plus A's OWN
#    free light molecule routed back to the same (well0,bcA). PairPlex REALLY emits a
#    coherent same-source pair (non-vacuous), and it must NOT be classified mispaired.
# ---------------------------------------------------------------------------
def test_negative_control(tmp_path):
    bcA = load_barcodes("v2", 1, rng_for(2, "fx"))[0]
    cells = pl.DataFrame({"cell_id":[0],"source_pair_id":["A"],
        "chain0_id":["hA"],"chain0_seq":[HEAVY_A],"chain0_locus":["IGH"],
        "chain1_id":["lA"],"chain1_seq":[LIGHT_A],"chain1_locus":["IGK"],
        "droplet_id":[0],"barcode":[bcA],"resident_well":[0]})
    chain_status = pl.DataFrame({"cell_id":[0,0],"chain":[0,1],
        "captured":[True,True],"n_molecules":[1,1]})   # A: both heavy and light captured
    molecules = pl.DataFrame({"molecule_id":[0,1],"origin_cell_id":[0,0],"chain":[0,1],"survived":[True,True]})
    reads = pl.concat([family(0,0,"A",0,"IGH",HEAVY_A,0,bcA,"AAAAAAAAAA",False),   # A heavy resident @ well0 / bcA
                       family(1,0,"A",1,"IGK",LIGHT_A,0,bcA,"AACCAACCAA",True)])    # A OWN light FREE -> back to well0 / bcA
    rd = emit(cells,chain_status,molecules,reads,tmp_path/"sim")
    ppo = tmp_path/"pp"; _run(rd,ppo)
    ps,_ = score(ppo,(tmp_path/"sim"/"truth"))
    assert ps.height >= 1                                             # non-vacuous: a real pair was emitted
    assert (ps["pairing_status"]=="mispaired").sum() == 0            # ...and the same-source pair is NOT mispaired
    assert (ps["origin_status"].is_in(["resident","ambient"])).all() # coherent origin, not a cross-source mispair


# ---------------------------------------------------------------------------
# 3. same-well collision: A,B share a barcode + resident_well; A-light and B-heavy
#    absent -> heavy A + light B pair at a collision key -> mispaired.
# ---------------------------------------------------------------------------
def test_same_well_collision(tmp_path):
    cells = pl.DataFrame({"cell_id":[0,1],"source_pair_id":["A","B"],
        "chain0_id":["hA","hB"],"chain0_seq":[HEAVY_A,HEAVY_B],"chain0_locus":["IGH","IGH"],
        "chain1_id":["lA","lB"],"chain1_seq":[LIGHT_A,LIGHT_B],"chain1_locus":["IGK","IGK"],
        "droplet_id":[0,0],"barcode":[BC,BC],"resident_well":[0,0]})
    chain_status = pl.DataFrame({"cell_id":[0,0,1,1],"chain":[0,1,0,1],
        "captured":[True,False,False,True],"n_molecules":[1,0,0,1]})   # A: heavy only; B: light only
    molecules = pl.DataFrame({"molecule_id":[0,1],"origin_cell_id":[0,1],"chain":[0,1],"survived":[True,True]})
    reads = pl.concat([family(0,0,"A",0,"IGH",HEAVY_A,0,BC,"AAAAAAAAAA",False),   # A heavy resident @ well0
                       family(1,1,"B",1,"IGK",LIGHT_B,0,BC,"CCCCCCCCCC",False)])   # B light resident @ well0 (same key)
    rd = emit(cells,chain_status,molecules,reads,tmp_path/"sim")
    ppo = tmp_path/"pp"; _run(rd,ppo)
    ps,_ = score(ppo,(tmp_path/"sim"/"truth"))
    mis = ps.filter(pl.col("pairing_status")=="mispaired")
    assert mis.height >= 1
    assert (mis["key_status"]=="collision").any()


# ---------------------------------------------------------------------------
# 4. route composition: a single molecule with one index-hopped read; truth_reads
#    records final_well != amplification_well with barcode+UMI unchanged.
# ---------------------------------------------------------------------------
def test_route_composition(tmp_path):
    cells = pl.DataFrame({"cell_id":[0],"source_pair_id":["A"],
        "chain0_id":["hA"],"chain0_seq":[HEAVY_A],"chain0_locus":["IGH"],
        "chain1_id":["lA"],"chain1_seq":[LIGHT_A],"chain1_locus":["IGK"],
        "droplet_id":[0],"barcode":[BC],"resident_well":[0]})
    chain_status = pl.DataFrame({"cell_id":[0,0],"chain":[0,1],
        "captured":[True,False],"n_molecules":[1,0]})
    molecules = pl.DataFrame({"molecule_id":[0],"origin_cell_id":[0],"chain":[0],"survived":[True]})
    reads = family(0,0,"A",0,"IGH",HEAVY_A,0,BC,"AAAAAAAAAA",False,hop_one_to=1)  # one of four reads hops to well1
    emit(cells,chain_status,molecules,reads,tmp_path/"sim",write_read_truth=True)
    tr = pl.read_parquet(tmp_path/"sim"/"truth"/"truth_reads.parquet")
    hopped = tr.filter(pl.col("final_well")!=pl.col("amplification_well"))
    assert hopped.height >= 1
    row = hopped.to_dicts()[0]
    assert row["barcode"]==BC and row["umi"]=="AAAAAAAAAA"     # barcode+UMI ride along, only the well changes
    assert row["is_index_hopped"] is True
    assert (tr["barcode"]==BC).all() and (tr["umi"]=="AAAAAAAAAA").all()


# ---------------------------------------------------------------------------
# 5. joint ambiguity: two source pairs share HEAVY_A (distinct lights); the light
#    present disambiguates the jointly-ambiguous heavy -> correct.
# ---------------------------------------------------------------------------
def test_joint_ambiguity(tmp_path):
    cells = pl.DataFrame({"cell_id":[0,1],"source_pair_id":["A","B"],
        "chain0_id":["hA","hB"],"chain0_seq":[HEAVY_A,HEAVY_A],"chain0_locus":["IGH","IGH"],  # SHARED heavy
        "chain1_id":["lA","lB"],"chain1_seq":[LIGHT_A,LIGHT_B],"chain1_locus":["IGK","IGK"],
        "droplet_id":[0,0],"barcode":[BC,BC],"resident_well":[0,0]})
    chain_status = pl.DataFrame({"cell_id":[0,0,1,1],"chain":[0,1,0,1],
        "captured":[True,True,True,False],"n_molecules":[1,1,1,0]})   # A: both; B: heavy only
    molecules = pl.DataFrame({"molecule_id":[0,1,2],"origin_cell_id":[0,0,1],
        "chain":[0,1,0],"survived":[True,True,True]})
    reads = pl.concat([
        family(0,0,"A",0,"IGH",HEAVY_A,0,BC,"AAAAAAAAAA",False),   # A heavy (HEAVY_A)
        family(1,1,"B",0,"IGH",HEAVY_A,0,BC,"TTTTTTTTTT",False),   # B heavy (also HEAVY_A) -> same contig, umis={A,B}
        family(2,0,"A",1,"IGK",LIGHT_A,0,BC,"AACCAACCAA",False)])   # only LIGHT_A present -> disambiguates to A
    rd = emit(cells,chain_status,molecules,reads,tmp_path/"sim")
    ppo = tmp_path/"pp"; _run(rd,ppo)
    ps,_ = score(ppo,(tmp_path/"sim"/"truth"))
    assert ps.height >= 1
    assert (ps["pairing_status"]=="correct").all()
    assert (ps["resolved_source"]=="A").any()


# ---------------------------------------------------------------------------
# 6. missing output: resident A pair present, but a contaminant heavy contig at the
#    same key gives 2 heavies -> PairPlex rejects (needs exactly 1H+1L) -> missing.
# ---------------------------------------------------------------------------
def test_missing_output(tmp_path):
    cells = pl.DataFrame({"cell_id":[0,1],"source_pair_id":["A","B"],
        "chain0_id":["hA","hB"],"chain0_seq":[HEAVY_A,HEAVY_B],"chain0_locus":["IGH","IGH"],
        "chain1_id":["lA","lB"],"chain1_seq":[LIGHT_A,LIGHT_B],"chain1_locus":["IGK","IGK"],
        "droplet_id":[0,1],"barcode":[BC,BC],"resident_well":[0,1]})   # A resident @ well0; B home @ well1
    chain_status = pl.DataFrame({"cell_id":[0,0,1,1],"chain":[0,1,0,1],
        "captured":[True,True,True,False],"n_molecules":[1,1,1,0]})
    molecules = pl.DataFrame({"molecule_id":[0,1,2],"origin_cell_id":[0,0,1],
        "chain":[0,1,0],"survived":[True,True,True]})
    reads = pl.concat([
        family(0,0,"A",0,"IGH",HEAVY_A,0,BC,"AAAAAAAAAA",False),   # resident heavy A @ well0/BC
        family(1,0,"A",1,"IGK",LIGHT_A,0,BC,"AACCAACCAA",False),   # resident light A @ well0/BC
        family(2,1,"B",0,"IGH",HEAVY_B,0,BC,"GGGGGGGGGG",True)])    # contaminant free heavy B -> 2 heavies at key
    rd = emit(cells,chain_status,molecules,reads,tmp_path/"sim")
    ppo = tmp_path/"pp"; _run(rd,ppo)
    ps,ks = score(ppo,(tmp_path/"sim"/"truth"))
    akey = ks.filter((pl.col("well")==0)&(pl.col("barcode")==BC))
    assert akey.height == 1
    assert akey.to_dicts()[0]["output_status"]=="missing"
