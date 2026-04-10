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


def test_excluded_target_aas_filters_cysteine(annotator, vhh):
    """Excluding Cysteine as a target should remove all suggestions where
    suggested_aa is 'C'."""
    all_suggestions = annotator.get_mutation_suggestions(vhh, off_limits=set())
    filtered = annotator.get_mutation_suggestions(
        vhh, off_limits=set(), excluded_target_aas={"C"},
    )
    # No suggestion should have C as target
    for s in filtered:
        assert s["suggested_aa"] != "C", f"Cysteine should be excluded but found at pos {s['position']}"
    # Filtered should be a subset of all suggestions
    assert len(filtered) <= len(all_suggestions)


def test_excluded_target_aas_multiple(annotator, vhh):
    """Excluding multiple amino acids should filter all of them out."""
    excluded = {"C", "M", "W"}
    filtered = annotator.get_mutation_suggestions(
        vhh, off_limits=set(), excluded_target_aas=excluded,
    )
    for s in filtered:
        assert s["suggested_aa"] not in excluded


def test_excluded_target_aas_empty_set_no_effect(annotator, vhh):
    """An empty excluded set should have no effect."""
    all_suggestions = annotator.get_mutation_suggestions(vhh, off_limits=set())
    filtered = annotator.get_mutation_suggestions(
        vhh, off_limits=set(), excluded_target_aas=set(),
    )
    assert len(filtered) == len(all_suggestions)
