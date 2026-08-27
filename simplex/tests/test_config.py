import pytest
from simplex.config import SimplexConfig
def C(**k): return SimplexConfig(input_data="x", output_directory="o", **k)
def test_defaults(tmp_path): c=C(); assert c.output_mode=="merged"; c.to_json(tmp_path/"c.json")
def test_reject_paired():
    with pytest.raises(ValueError): C(output_mode="paired").validate()
def test_reject_bad_rate():
    with pytest.raises(ValueError): C(release_rate=1.5).validate()
def test_index_hop_one_well():
    with pytest.raises(ValueError): C(wells=1, index_hop_rate=0.01).validate()
def test_reject_fixed_structure_change():
    with pytest.raises(ValueError): C(barcode_length=12).validate()
    with pytest.raises(ValueError): C(umi_length=12).validate()
    with pytest.raises(ValueError): C(tso="GGGGGGGGGGGGG").validate()
def test_reject_nonpositive():
    for k in ["wells","cells_per_droplet_mean","molecules_per_chain_mean","reads_per_molecule_mean"]:
        with pytest.raises(ValueError): C(**{k:0}).validate()
def test_oom():
    with pytest.raises(ValueError): C(reads_per_molecule_mean=50, molecules_per_chain_mean=50).validate(actual_n_cells=10_000_000, max_reads=5_000_000_000)
