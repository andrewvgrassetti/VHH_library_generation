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
