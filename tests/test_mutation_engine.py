import pytest
import pandas as pd
from vhh_library.sequence import VHHSequence
from vhh_library.humanness import HumAnnotator
from vhh_library.stability import StabilityScorer
from vhh_library.mutation_engine import MutationEngine

SAMPLE_VHH = "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"

@pytest.fixture
def engine():
    h = HumAnnotator()
    s = StabilityScorer()
    return MutationEngine(h, s)

@pytest.fixture
def vhh():
    return VHHSequence(SAMPLE_VHH)

def test_rank_single_mutations(engine, vhh):
    df = engine.rank_single_mutations(vhh, off_limits=set())
    assert isinstance(df, pd.DataFrame)
    if len(df) > 0:
        assert "position" in df.columns
        assert "combined_score" in df.columns

def test_apply_mutations(engine):
    seq = "ACDEFGHIKLMNPQRSTVWY"
    result = engine.apply_mutations(seq, [(0, "M"), (1, "L")])
    assert result[0] == "M"
    assert result[1] == "L"
    assert result[2:] == seq[2:]

def test_generate_library(engine, vhh):
    ranked = engine.rank_single_mutations(vhh, off_limits=set())
    if len(ranked) >= 2:
        library = engine.generate_library(vhh, ranked.head(5), n_mutations=2, max_variants=100)
        assert isinstance(library, pd.DataFrame)
        assert len(library) > 0
        assert "n_mutations" in library.columns


def test_generate_library_min_mutations(engine, vhh):
    ranked = engine.rank_single_mutations(vhh, off_limits=set())
    if len(ranked) >= 3:
        library = engine.generate_library(
            vhh, ranked.head(5), n_mutations=3, max_variants=100, min_mutations=2,
        )
        assert isinstance(library, pd.DataFrame)
        if len(library) > 0:
            # Every variant should have at least 2 mutations
            assert library["n_mutations"].min() >= 2
            assert library["n_mutations"].max() <= 3


def test_generate_library_large_sampling(engine, vhh):
    """With high mutation counts the engine should use sampling and finish quickly."""
    ranked = engine.rank_single_mutations(vhh, off_limits=set())
    if len(ranked) >= 10:
        import time
        start = time.time()
        library = engine.generate_library(
            vhh, ranked, n_mutations=10, max_variants=200, min_mutations=8,
        )
        elapsed = time.time() - start
        assert isinstance(library, pd.DataFrame)
        assert len(library) > 0
        assert len(library) <= 200
        # Must complete in seconds, not hours
        assert elapsed < 120, f"Library generation took {elapsed:.1f}s (expected <120s)"
        assert library["n_mutations"].min() >= 8
        assert library["n_mutations"].max() <= 10


def test_generate_library_has_developability_columns(engine, vhh):
    """Library output should include the surface hydrophobicity score column."""
    ranked = engine.rank_single_mutations(vhh, off_limits=set())
    if len(ranked) >= 2:
        library = engine.generate_library(vhh, ranked.head(5), n_mutations=2, max_variants=50)
        assert isinstance(library, pd.DataFrame)
        if len(library) > 0:
            assert "surface_hydrophobicity_score" in library.columns


def test_generate_library_has_orthogonal_columns(engine, vhh):
    """Library output should include orthogonal humanness and stability score columns."""
    ranked = engine.rank_single_mutations(vhh, off_limits=set())
    if len(ranked) >= 2:
        library = engine.generate_library(vhh, ranked.head(5), n_mutations=2, max_variants=50)
        assert isinstance(library, pd.DataFrame)
        if len(library) > 0:
            for col in ("orthogonal_humanness_score", "orthogonal_stability_score"):
                assert col in library.columns, f"Missing column: {col}"
            # Orthogonal scores should be in [0, 1]
            assert library["orthogonal_humanness_score"].between(0, 1).all()
            assert library["orthogonal_stability_score"].between(0, 1).all()


def test_rank_single_mutations_excluded_target_aas(engine, vhh):
    """Excluded target AAs should be filtered from ranked mutation suggestions."""
    all_ranked = engine.rank_single_mutations(vhh, off_limits=set())
    filtered = engine.rank_single_mutations(
        vhh, off_limits=set(), excluded_target_aas={"C"},
    )
    if len(filtered) > 0:
        assert "C" not in filtered["suggested_aa"].values
    assert len(filtered) <= len(all_ranked)


def test_engine_enabled_metrics():
    """Enabled metrics should affect the combined score calculation."""
    h = HumAnnotator()
    s = StabilityScorer()
    engine_all = MutationEngine(
        h, s,
        weights={"humanness": 0.3, "stability": 0.4, "surface_hydrophobicity": 0.3},
        enabled_metrics={"humanness": True, "stability": True, "surface_hydrophobicity": True},
    )
    engine_two = MutationEngine(
        h, s,
        weights={"humanness": 0.35, "stability": 0.5},
        enabled_metrics={"humanness": True, "stability": True, "surface_hydrophobicity": False},
    )
    # Both should produce valid active weights
    aw_all = engine_all._active_weights()
    aw_two = engine_two._active_weights()
    assert len(aw_all) == 3
    assert len(aw_two) == 2
    assert abs(sum(aw_all.values()) - 1.0) < 1e-6
    assert abs(sum(aw_two.values()) - 1.0) < 1e-6


def test_generate_library_strategy_random(engine, vhh):
    """Explicitly selecting 'random' strategy should work the same as the default for large spaces."""
    ranked = engine.rank_single_mutations(vhh, off_limits=set())
    if len(ranked) >= 2:
        library = engine.generate_library(
            vhh, ranked.head(8), n_mutations=3, max_variants=50, strategy="random",
        )
        assert isinstance(library, pd.DataFrame)
        assert len(library) > 0
        assert "combined_score" in library.columns


def test_generate_library_strategy_iterative(engine, vhh):
    """Iterative refinement strategy should produce a valid library."""
    ranked = engine.rank_single_mutations(vhh, off_limits=set())
    if len(ranked) >= 4:
        library = engine.generate_library(
            vhh, ranked.head(10), n_mutations=4, max_variants=80,
            strategy="iterative", anchor_threshold=0.5, max_rounds=3,
        )
        assert isinstance(library, pd.DataFrame)
        assert len(library) > 0
        assert "combined_score" in library.columns
        assert len(library) <= 80
        # Library should be sorted descending by combined_score
        scores = list(library["combined_score"])
        assert scores == sorted(scores, reverse=True)


def test_generate_library_strategy_auto_routes(engine, vhh):
    """'auto' strategy should route to iterative for very large spaces without crashing."""
    ranked = engine.rank_single_mutations(vhh, off_limits=set())
    if len(ranked) >= 5:
        library = engine.generate_library(
            vhh, ranked.head(10), n_mutations=5, max_variants=50, strategy="auto",
        )
        assert isinstance(library, pd.DataFrame)
        assert len(library) > 0


def test_iterative_anchoring(engine, vhh):
    """Iterative strategy should identify anchors and produce diverse variants."""
    ranked = engine.rank_single_mutations(vhh, off_limits=set())
    if len(ranked) < 6:
        pytest.skip("Not enough mutation candidates for anchor test")

    import time
    start = time.time()
    library = engine.generate_library(
        vhh, ranked.head(12), n_mutations=5, max_variants=100,
        strategy="iterative", anchor_threshold=0.6, max_rounds=3,
    )
    elapsed = time.time() - start

    assert isinstance(library, pd.DataFrame)
    assert len(library) > 0
    assert elapsed < 180, f"Iterative refinement took {elapsed:.1f}s (expected <180s)"

    # Variants should be sorted by combined_score descending
    scores = list(library["combined_score"])
    assert scores == sorted(scores, reverse=True)

    # No duplicate mutation combinations
    assert library["mutations"].is_unique


def test_ptm_liability_hard_restriction(engine, vhh):
    """Mutations that would introduce isomerization/deamidation/glycosylation
    motifs should be filtered out by the hard-coded restriction."""
    from vhh_library.mutation_engine import _introduces_ptm_liability

    # DG is an isomerization motif: introducing D before G should be flagged
    parent = "QVQLVESGGGLVQ"
    mutant = "QVQLVEDGGGLVQ"  # S→D at pos 6 (0-idx), creates DG at pos 6-7
    assert _introduces_ptm_liability(parent, mutant, 6) is True

    # A benign substitution should pass
    parent2 = "ACDEFGHIKLMNPQ"
    mutant2 = "ACAEFGHIKLMNPQ"  # D→A at pos 2 (0-idx)
    assert _introduces_ptm_liability(parent2, mutant2, 2) is False


def test_stability_is_heaviest_weight():
    """Default stability weight should be the largest among all metrics."""
    h = HumAnnotator()
    s = StabilityScorer()
    engine = MutationEngine(h, s)
    max_weight_key = max(engine.weights, key=engine.weights.get)
    assert max_weight_key == "stability"
