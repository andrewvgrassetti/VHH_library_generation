import pytest
from vhh_library.sequence import VHHSequence
from vhh_library.orthogonal_scoring import (
    HumanStringContentScorer,
    ConsensusStabilityScorer,
    NanoMeltStabilityScorer,
)

SAMPLE_VHH = "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"


@pytest.fixture
def vhh():
    return VHHSequence(SAMPLE_VHH)


# -- Human String Content Scorer --------------------------------------------

class TestHumanStringContentScorer:
    @pytest.fixture
    def scorer(self):
        return HumanStringContentScorer()

    def test_score_returns_dict(self, scorer, vhh):
        result = scorer.score(vhh)
        assert isinstance(result, dict)
        assert "composite_score" in result
        assert "total_kmers" in result
        assert "matched_kmers" in result

    def test_composite_score_range(self, scorer, vhh):
        result = scorer.score(vhh)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_total_kmers_positive(self, scorer, vhh):
        result = scorer.score(vhh)
        assert result["total_kmers"] > 0

    def test_matched_kmers_not_exceed_total(self, scorer, vhh):
        result = scorer.score(vhh)
        assert result["matched_kmers"] <= result["total_kmers"]

    def test_predict_mutation_effect(self, scorer, vhh):
        delta = scorer.predict_mutation_effect(vhh, 1, "E")
        assert isinstance(delta, float)

    def test_no_change_when_same_aa(self, scorer, vhh):
        aa = vhh.sequence[0]
        delta = scorer.predict_mutation_effect(vhh, 1, aa)
        assert delta == 0.0

    def test_custom_kmer_size(self, vhh):
        scorer_small = HumanStringContentScorer(kmer_size=5)
        result = scorer_small.score(vhh)
        assert 0.0 <= result["composite_score"] <= 1.0
        assert result["total_kmers"] > 0


# -- Consensus Stability Scorer ---------------------------------------------

class TestConsensusStabilityScorer:
    @pytest.fixture
    def scorer(self):
        return ConsensusStabilityScorer()

    def test_score_returns_dict(self, scorer, vhh):
        result = scorer.score(vhh)
        assert isinstance(result, dict)
        assert "composite_score" in result
        assert "positions_evaluated" in result
        assert "consensus_matches" in result
        assert "avg_conservation" in result

    def test_composite_score_range(self, scorer, vhh):
        result = scorer.score(vhh)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_positions_evaluated_positive(self, scorer, vhh):
        result = scorer.score(vhh)
        assert result["positions_evaluated"] > 0

    def test_consensus_matches_not_exceed_evaluated(self, scorer, vhh):
        result = scorer.score(vhh)
        assert result["consensus_matches"] <= result["positions_evaluated"]

    def test_avg_conservation_range(self, scorer, vhh):
        result = scorer.score(vhh)
        assert 0.0 <= result["avg_conservation"] <= 1.0

    def test_predict_mutation_effect(self, scorer, vhh):
        delta = scorer.predict_mutation_effect(vhh, 1, "E")
        assert isinstance(delta, float)

    def test_no_change_when_same_aa(self, scorer, vhh):
        aa = vhh.sequence[0]
        delta = scorer.predict_mutation_effect(vhh, 1, aa)
        assert delta == 0.0


# -- Integration: orthogonal scores correlate with primary scores -----------

class TestOrthogonalCorrelation:
    """Verify that orthogonal scores are computed and lie in valid ranges
    for the sample VHH and a mutated variant."""

    def test_both_scorers_run_on_same_sequence(self, vhh):
        hsc = HumanStringContentScorer()
        cons = ConsensusStabilityScorer()
        h_result = hsc.score(vhh)
        s_result = cons.score(vhh)
        assert 0.0 <= h_result["composite_score"] <= 1.0
        assert 0.0 <= s_result["composite_score"] <= 1.0

    def test_mutation_changes_orthogonal_scores(self, vhh):
        """Mutating a residue should (potentially) change orthogonal scores."""
        hsc = HumanStringContentScorer()
        cons = ConsensusStabilityScorer()

        orig_h = hsc.score(vhh)["composite_score"]
        orig_s = cons.score(vhh)["composite_score"]

        # Apply a mutation
        seq_list = list(vhh.sequence)
        seq_list[0] = "E" if seq_list[0] != "E" else "A"
        mutant = VHHSequence("".join(seq_list))

        mut_h = hsc.score(mutant)["composite_score"]
        mut_s = cons.score(mutant)["composite_score"]

        # Scores should be valid (may or may not change)
        assert 0.0 <= mut_h <= 1.0
        assert 0.0 <= mut_s <= 1.0


# -- NanoMelt Stability Scorer ----------------------------------------------

class TestNanoMeltStabilityScorer:
    """Tests for NanoMeltStabilityScorer.

    Because the nanomelt package may or may not be installed in the test
    environment, most tests use the scorer's ``is_available`` flag to decide
    what to assert.  This keeps the tests meaningful in both cases without
    requiring heavy model weights in CI.
    """

    @pytest.fixture
    def scorer(self):
        return NanoMeltStabilityScorer()

    def test_scorer_instantiation_does_not_raise(self):
        """Creating the scorer must not import nanomelt eagerly."""
        scorer = NanoMeltStabilityScorer()
        assert scorer is not None

    def test_is_available_returns_bool(self, scorer):
        """is_available must return a bool regardless of install status."""
        assert isinstance(scorer.is_available, bool)

    def test_lazy_load_sets_available_flag(self, scorer):
        """Accessing is_available should trigger lazy loading."""
        _ = scorer.is_available
        assert scorer._available is not None

    def test_predict_mutation_effect_unavailable_returns_zero(self, vhh):
        """If nanomelt is absent, predict_mutation_effect must return 0.0."""
        scorer = NanoMeltStabilityScorer()
        if scorer.is_available:
            pytest.skip("nanomelt is installed; skipping unavailable-path test")
        delta = scorer.predict_mutation_effect(vhh, 1, "E")
        assert delta == 0.0

    def test_score_raises_import_error_when_unavailable(self, vhh):
        """If nanomelt is absent, score() must raise ImportError."""
        scorer = NanoMeltStabilityScorer()
        if scorer.is_available:
            pytest.skip("nanomelt is installed; skipping unavailable-path test")
        with pytest.raises(ImportError, match="nanomelt"):
            scorer.score(vhh)

    def test_score_returns_valid_dict_when_available(self, vhh):
        """If nanomelt is installed, score() returns expected keys and ranges."""
        scorer = NanoMeltStabilityScorer()
        if not scorer.is_available:
            pytest.skip("nanomelt not installed; skipping live prediction test")
        result = scorer.score(vhh)
        assert isinstance(result, dict)
        assert "composite_score" in result
        assert "predicted_tm" in result
        assert 0.0 <= result["composite_score"] <= 1.0
        # Tm should be in a plausible range for nanobodies
        assert 20.0 <= result["predicted_tm"] <= 120.0

    def test_predict_mutation_effect_returns_float_when_available(self, vhh):
        scorer = NanoMeltStabilityScorer()
        if not scorer.is_available:
            pytest.skip("nanomelt not installed; skipping live prediction test")
        delta = scorer.predict_mutation_effect(vhh, 1, "E")
        assert isinstance(delta, float)

    def test_predict_mutation_effect_same_aa_returns_zero(self, vhh):
        scorer = NanoMeltStabilityScorer()
        if not scorer.is_available:
            pytest.skip("nanomelt not installed; skipping live prediction test")
        aa = vhh.sequence[0]
        delta = scorer.predict_mutation_effect(vhh, 1, aa)
        assert delta == 0.0

    def test_predict_mutation_effect_out_of_range_returns_zero(self, vhh):
        scorer = NanoMeltStabilityScorer()
        if not scorer.is_available:
            pytest.skip("nanomelt not installed; skipping live prediction test")
        delta = scorer.predict_mutation_effect(vhh, 9999, "A")
        assert delta == 0.0
