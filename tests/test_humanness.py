import pytest
from vhh_library.sequence import VHHSequence
from vhh_library.humanness import HumAnnotator

SAMPLE_VHH = "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"

@pytest.fixture
def annotator():
    return HumAnnotator()

@pytest.fixture
def vhh():
    return VHHSequence(SAMPLE_VHH)

def test_loads_germlines(annotator):
    assert len(annotator.germlines) > 0

def test_score_returns_dict(annotator, vhh):
    result = annotator.score(vhh)
    assert isinstance(result, dict)
    assert "composite_score" in result
    assert "germline_identity" in result

def test_composite_score_range(annotator, vhh):
    result = annotator.score(vhh)
    assert 0.0 <= result["composite_score"] <= 1.0

def test_position_scores_exist(annotator, vhh):
    result = annotator.score(vhh)
    assert "position_scores" in result
    assert len(result["position_scores"]) > 0

def test_mutation_suggestions(annotator, vhh):
    suggestions = annotator.get_mutation_suggestions(vhh, off_limits=set())
    assert isinstance(suggestions, list)
