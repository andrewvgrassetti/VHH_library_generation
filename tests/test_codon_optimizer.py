import pytest
from vhh_library.codon_optimizer import CodonOptimizer
from vhh_library.utils import translate

SAMPLE_AA = "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGK"

@pytest.fixture
def optimizer():
    return CodonOptimizer()

def test_optimize_returns_dict(optimizer):
    result = optimizer.optimize(SAMPLE_AA, "e_coli")
    assert isinstance(result, dict)
    assert "dna_sequence" in result

def test_dna_translates_correctly(optimizer):
    result = optimizer.optimize(SAMPLE_AA, "e_coli")
    dna = result["dna_sequence"]
    assert len(dna) == len(SAMPLE_AA) * 3
    translated = translate(dna)
    assert translated == SAMPLE_AA

def test_gc_content_range(optimizer):
    result = optimizer.optimize(SAMPLE_AA, "e_coli")
    assert 0.0 <= result["gc_content"] <= 1.0

def test_no_stop_codons_in_middle(optimizer):
    result = optimizer.optimize(SAMPLE_AA, "e_coli")
    dna = result["dna_sequence"]
    for i in range(0, len(dna) - 3, 3):
        codon = dna[i:i+3]
        assert codon not in ["TAA", "TAG", "TGA"], f"Stop codon found at position {i}"

def test_s_cerevisiae(optimizer):
    result = optimizer.optimize(SAMPLE_AA, "s_cerevisiae")
    assert isinstance(result["dna_sequence"], str)

def test_p_pastoris(optimizer):
    result = optimizer.optimize(SAMPLE_AA, "p_pastoris")
    assert isinstance(result["dna_sequence"], str)

def test_h_sapiens(optimizer):
    result = optimizer.optimize(SAMPLE_AA, "h_sapiens")
    assert isinstance(result["dna_sequence"], str)
    dna = result["dna_sequence"]
    assert len(dna) == len(SAMPLE_AA) * 3
    translated = translate(dna)
    assert translated == SAMPLE_AA
