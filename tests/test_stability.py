import pytest
from vhh_library.sequence import VHHSequence
from vhh_library.stability import StabilityScorer

SAMPLE_VHH = "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"

@pytest.fixture
def scorer():
    return StabilityScorer()

@pytest.fixture
def vhh():
    return VHHSequence(SAMPLE_VHH)

def test_score_returns_dict(scorer, vhh):
    result = scorer.score(vhh)
    assert isinstance(result, dict)
    assert "composite_score" in result

def test_composite_score_range(scorer, vhh):
    result = scorer.score(vhh)
    assert 0.0 <= result["composite_score"] <= 1.0

def test_pI_range(scorer, vhh):
    result = scorer.score(vhh)
    assert 3.0 <= result["pI"] <= 12.0

def test_disulfide_score(scorer, vhh):
    result = scorer.score(vhh)
    assert "disulfide_score" in result
    assert 0.0 <= result["disulfide_score"] <= 1.0

def test_predict_mutation_effect(scorer, vhh):
    delta = scorer.predict_mutation_effect(vhh, 1, "E")
    assert isinstance(delta, float)
