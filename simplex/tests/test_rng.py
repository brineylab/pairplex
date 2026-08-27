from simplex._rng import rng_for
def test_same_stage(): assert list(rng_for(0,"m").integers(0,10**6,50))==list(rng_for(0,"m").integers(0,10**6,50))
def test_diff_stage(): assert list(rng_for(0,"m").integers(0,10**6,50))!=list(rng_for(0,"n").integers(0,10**6,50))
