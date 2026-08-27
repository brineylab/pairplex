from simplex.matching import resolve, seq_match, candidates
def test_disjoint_singletons():
    r=resolve({"A"},{"B"}); assert r[:3]==("mispaired","none",None) and r.valid_assignments=={("A","B")}
def test_disjoint_one_ambiguous():
    r=resolve({"A","B"},{"C"}); assert r[:3]==("mispaired","none",None) and r.valid_assignments=={("A","C"),("B","C")}
def test_unique():
    r=resolve({"A","B"},{"A"}); assert r[:3]==("correct","unique","A") and r.valid_assignments=={("A","A")}
def test_nonunique():
    r=resolve({"A","B"},{"A","B"}); assert r[:3]==("correct","ambiguous",None) and r.valid_assignments=={("A","A"),("B","B")}
def test_empty():
    r=resolve(set(),{"A"}); assert r[:3]==("unmatchable","none",None) and r.valid_assignments==set()
def test_seq_match():
    a="ACGT"*30; b=a[:60]+"T"+a[61:]
    assert seq_match(a,b) and not seq_match(a,"TTTT"*30) and not seq_match("ACG","ACG")  # too short

def test_candidates_locus_restriction_and_union():
    # seqA/seqC are close (2 mismatches, within the 6% tolerance @ len 80); seqB is unrelated to both.
    seqA="A"*80; seqB="C"*80; seqC="A"*78+"GG"
    entry={"IGH":{seqA:{"S1"},seqB:{"S2"},seqC:{"S3"}},"IGK":{seqA:{"S9"}}}
    # exact match with no fuzzy collision -> single source
    assert candidates(seqB,"IGH",entry)=={"S2"}
    # query fuzzy-matches two distinct stored IGH sequences -> sources are unioned
    assert candidates(seqA,"IGH",entry)=={"S1","S3"}
    # same sequence stored under IGK must not leak IGH sources (locus-restricted)
    assert candidates(seqA,"IGK",entry)=={"S9"}
    # no key_entry -> empty set
    assert candidates("x","IGH",None)==set()

def test_seq_match_infix_different_lengths():
    a="ACGT"*20  # len 80, meets min_len=50 default guard
    embedded="GGGG"+a+"TTTT"  # len 88, a is a true infix
    assert seq_match(a,embedded)
    assert not seq_match(a,"TGCA"*30)  # len 120, not contained as an infix
