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
    if len(ranked) >= 14:
        import time
        start = time.time()
        library = engine.generate_library(
            vhh, ranked, n_mutations=14, max_variants=200, min_mutations=12,
        )
        elapsed = time.time() - start
        assert isinstance(library, pd.DataFrame)
        assert len(library) > 0
        assert len(library) <= 200
        # Must complete in seconds, not hours
        assert elapsed < 120, f"Library generation took {elapsed:.1f}s (expected <120s)"
        assert library["n_mutations"].min() >= 12
        assert library["n_mutations"].max() <= 14


def test_generate_library_has_developability_columns(engine, vhh):
    """Library output should include the three new developability score columns."""
    ranked = engine.rank_single_mutations(vhh, off_limits=set())
    if len(ranked) >= 2:
        library = engine.generate_library(vhh, ranked.head(5), n_mutations=2, max_variants=50)
        assert isinstance(library, pd.DataFrame)
        if len(library) > 0:
            for col in ("ptm_liability_score", "clearance_risk_score", "surface_hydrophobicity_score"):
                assert col in library.columns, f"Missing column: {col}"


def test_engine_enabled_metrics():
    """Enabled metrics should affect the combined score calculation."""
    h = HumAnnotator()
    s = StabilityScorer()
    engine_all = MutationEngine(
        h, s,
        weights={"humanness": 0.2, "stability": 0.2, "ptm_liability": 0.2,
                 "clearance_risk": 0.2, "surface_hydrophobicity": 0.2},
        enabled_metrics={"humanness": True, "stability": True, "ptm_liability": True,
                         "clearance_risk": True, "surface_hydrophobicity": True},
    )
    engine_two = MutationEngine(
        h, s,
        weights={"humanness": 0.5, "stability": 0.5},
        enabled_metrics={"humanness": True, "stability": True, "ptm_liability": False,
                         "clearance_risk": False, "surface_hydrophobicity": False},
    )
    # Both should produce valid active weights
    aw_all = engine_all._active_weights()
    aw_two = engine_two._active_weights()
    assert len(aw_all) == 5
    assert len(aw_two) == 2
    assert abs(sum(aw_all.values()) - 1.0) < 1e-6
    assert abs(sum(aw_two.values()) - 1.0) < 1e-6
