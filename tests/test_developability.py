import pytest
from vhh_library.sequence import VHHSequence
from vhh_library.developability import (
    PTMLiabilityScorer,
    ClearanceRiskScorer,
    SurfaceHydrophobicityScorer,
)

SAMPLE_VHH = "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"


@pytest.fixture
def vhh():
    return VHHSequence(SAMPLE_VHH)


# -- PTM Liability -----------------------------------------------------------

class TestPTMLiabilityScorer:
    @pytest.fixture
    def scorer(self):
        return PTMLiabilityScorer()

    def test_score_returns_dict(self, scorer, vhh):
        result = scorer.score(vhh)
        assert isinstance(result, dict)
        assert "composite_score" in result

    def test_composite_score_range(self, scorer, vhh):
        result = scorer.score(vhh)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_hits_populated(self, scorer, vhh):
        result = scorer.score(vhh)
        assert isinstance(result["hits"], list)

    def test_predict_mutation_effect(self, scorer, vhh):
        delta = scorer.predict_mutation_effect(vhh, 1, "E")
        assert isinstance(delta, float)

    def test_no_change_when_same_aa(self, scorer, vhh):
        aa = vhh.sequence[0]
        delta = scorer.predict_mutation_effect(vhh, 1, aa)
        assert delta == 0.0


# -- Clearance Risk ----------------------------------------------------------

class TestClearanceRiskScorer:
    @pytest.fixture
    def scorer(self):
        return ClearanceRiskScorer()

    def test_score_returns_dict(self, scorer, vhh):
        result = scorer.score(vhh)
        assert isinstance(result, dict)
        assert "composite_score" in result
        assert "pI" in result

    def test_composite_score_range(self, scorer, vhh):
        result = scorer.score(vhh)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_pI_deviation_non_negative(self, scorer, vhh):
        result = scorer.score(vhh)
        assert result["pI_deviation"] >= 0.0

    def test_predict_mutation_effect(self, scorer, vhh):
        delta = scorer.predict_mutation_effect(vhh, 1, "E")
        assert isinstance(delta, float)


# -- Surface Hydrophobicity --------------------------------------------------

class TestSurfaceHydrophobicityScorer:
    @pytest.fixture
    def scorer(self):
        return SurfaceHydrophobicityScorer()

    def test_score_returns_dict(self, scorer, vhh):
        result = scorer.score(vhh)
        assert isinstance(result, dict)
        assert "composite_score" in result

    def test_composite_score_range(self, scorer, vhh):
        result = scorer.score(vhh)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_n_patches_non_negative(self, scorer, vhh):
        result = scorer.score(vhh)
        assert result["n_patches"] >= 0

    def test_predict_mutation_effect(self, scorer, vhh):
        delta = scorer.predict_mutation_effect(vhh, 1, "E")
        assert isinstance(delta, float)
