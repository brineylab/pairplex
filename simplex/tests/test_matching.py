from simplex.matching import resolve, seq_match
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
