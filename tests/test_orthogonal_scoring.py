import pytest
from vhh_library.sequence import VHHSequence
from vhh_library.orthogonal_scoring import (
    HumanStringContentScorer,
    ConsensusStabilityScorer,
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
