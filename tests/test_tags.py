import pytest
from vhh_library.tags import TagManager

@pytest.fixture
def tag_manager():
    return TagManager()

def test_loads_tags(tag_manager):
    tags = tag_manager.get_available_tags()
    assert isinstance(tags, dict)
    assert len(tags) > 0
    assert "6xHis" in tags

def test_build_construct_with_c_tag(tag_manager):
    aa = "QVQLVESGGGLVQ"
    dna = "CAGGTGCAGCTGGTGGAGAGCGGCGGCGGCCTGGTGCAG"
    result = tag_manager.build_construct(aa, dna, c_tag="6xHis")
    assert isinstance(result, dict)
    assert "aa_construct" in result
    assert "HHHHHH" in result["aa_construct"]

def test_build_construct_schematic(tag_manager):
    aa = "QVQLVESGGGLVQ"
    dna = "CAGGTGCAGCTGGTGGAGAGCGGCGGCGGCCTGGTGCAG"
    result = tag_manager.build_construct(aa, dna, n_tag="HA", c_tag="6xHis")
    assert "schematic" in result
    assert "VHH" in result["schematic"]
