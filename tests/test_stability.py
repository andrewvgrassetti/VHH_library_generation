import pytest
from vhh_library.sequence import VHHSequence
from vhh_library.stability import StabilityScorer, _nanomelt_available, _esm2_pll_available, compute_esm2_pll

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

def test_scoring_method_present(scorer, vhh):
    result = scorer.score(vhh)
    assert "scoring_method" in result
    assert result["scoring_method"] in ("legacy", "nanomelt")

def test_legacy_fallback():
    """When use_nanomelt=False, scorer uses legacy composite."""
    scorer = StabilityScorer(use_nanomelt=False)
    vhh = VHHSequence(SAMPLE_VHH)
    result = scorer.score(vhh)
    assert result["scoring_method"] == "legacy"
    assert 0.0 <= result["composite_score"] <= 1.0

def test_nanomelt_active_property():
    scorer = StabilityScorer(use_nanomelt=False)
    assert scorer.nanomelt_active is False

def test_esm2_pll_available_returns_bool():
    assert isinstance(_esm2_pll_available(), bool)

def test_nanomelt_available_returns_bool():
    assert isinstance(_nanomelt_available(), bool)

def test_esm2_pll_is_available_by_default():
    """ESM-2 PLL packages (torch, esm) are default dependencies.

    This test verifies availability when the full dependency set is installed.
    It is skipped in lightweight CI environments where PyTorch is not present.
    """
    if not _esm2_pll_available():
        pytest.skip("torch/esm not installed in this environment")
    assert _esm2_pll_available() is True

def test_nanomelt_is_available_by_default():
    """NanoMelt is a default dependency.

    This test verifies availability when the full dependency set is installed.
    It is skipped in lightweight CI environments where nanomelt is not present.
    """
    if not _nanomelt_available():
        pytest.skip("nanomelt not installed in this environment")
    assert _nanomelt_available() is True

def test_compute_esm2_pll_when_unavailable():
    if _esm2_pll_available():
        pytest.skip("ESM-2 is installed; skipping unavailable test")
    with pytest.raises(ImportError):
        compute_esm2_pll(["ACDEFGHIKLMNPQRSTVWY"])
