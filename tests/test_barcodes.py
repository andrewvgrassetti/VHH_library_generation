"""Tests for the BarcodeGenerator and related barcode utilities."""

import re
import pytest
import pandas as pd
from pathlib import Path

from vhh_library.barcodes import (
    BarcodeGenerator,
    _barcode_passes_rules,
    _hydrophobicity,
    _peptide_neutral_mass,
    _mz,
)
from vhh_library.utils import tryptic_digest


# ── fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVK"
    "GRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


@pytest.fixture
def generator():
    return BarcodeGenerator()


@pytest.fixture
def small_library():
    """A tiny library DataFrame with the minimum required columns."""
    return pd.DataFrame([
        {
            "variant_id": f"VAR-{i:06d}",
            "mutations": f"A{i+1}V",
            "n_mutations": 1,
            "combined_score": float(10 - i) / 10.0,
            "aa_sequence": SAMPLE_VHH[:i] + "V" + SAMPLE_VHH[i + 1:] if i < len(SAMPLE_VHH) else SAMPLE_VHH,
        }
        for i in range(10)
    ])


# ── barcode design rule tests ─────────────────────────────────────────────────

class TestBarcodeDesignRules:
    """_barcode_passes_rules must enforce all design constraints."""

    def test_valid_barcode_passes(self):
        assert _barcode_passes_rules("LVTDLTK")

    def test_must_end_in_K_or_R(self):
        assert not _barcode_passes_rules("LVTDLTA")  # ends in A

    def test_no_methionine(self):
        assert not _barcode_passes_rules("LVTDMTK")  # contains M

    def test_no_cysteine(self):
        assert not _barcode_passes_rules("LVTDCTK")  # contains C

    def test_no_NG_motif(self):
        assert not _barcode_passes_rules("LVTDNGK")  # contains NG

    def test_no_NS_motif(self):
        assert not _barcode_passes_rules("LNSVLTK")  # contains NS

    def test_no_glycosylation_sequon_NST(self):
        # N-X-T where X is not P should fail
        assert not _barcode_passes_rules("LNATLTK")  # N-A-T

    def test_glycosylation_sequon_NPS_passes(self):
        # N-P-S should NOT trigger the sequon rule (X == P is excepted)
        # but check length and basic residue presence
        # ANPSK: A-N-P-S-K → N at pos 1, P at pos 2 (X=P), S at pos 3 → OK
        assert _barcode_passes_rules("ANPSVR")

    def test_must_have_basic_residue(self):
        # No K, R, or H → should fail
        # All hydrophobic, no basic residue (except the terminal R which is both cleavage and basic)
        assert _barcode_passes_rules("LVTDLTR")  # has R (basic) → should pass
        assert not _barcode_passes_rules("LVTDLTF")  # ends in F (non-basic, non-K/R) → fail (also fails K/R rule)
        # Test a sequence that would pass length/cleavage but lacks basic residue (impossible
        # to construct, since K/R at the end always qualifies as basic).  Instead verify H counts.
        assert _barcode_passes_rules("HLTDLTR")  # has H (basic)

    def test_length_too_short(self):
        assert not _barcode_passes_rules("LVTK")  # only 4 chars

    def test_length_too_long(self):
        assert not _barcode_passes_rules("A" * 12 + "K")  # 13 chars → too long
        assert _barcode_passes_rules("A" * 11 + "K")  # 12 chars → OK

    def test_no_internal_lysine(self):
        assert not _barcode_passes_rules("LVKDLTK")  # internal K

    def test_no_internal_arginine(self):
        assert not _barcode_passes_rules("LVRDLTR")  # internal R

    def test_exactly_6_chars_passes(self):
        assert _barcode_passes_rules("AFHLD" + "K")  # 6 chars, ends K, has H (basic)

    def test_exactly_12_chars_passes(self):
        seq = "AFEQTIVHLHEK"
        assert len(seq) == 12
        # check passes (no forbidden motifs)
        assert _barcode_passes_rules(seq)


# ── tryptic digest tests ───────────────────────────────────────────────────────

class TestTrypticDigest:
    def test_simple_cleavage_after_K(self):
        peptides = tryptic_digest("ACDEFK")
        assert "ACDEFK" in peptides

    def test_simple_cleavage_after_R(self):
        peptides = tryptic_digest("ACDEFR")
        assert "ACDEFR" in peptides

    def test_no_cleavage_before_P(self):
        # KP: trypsin does NOT cleave the K-P bond
        peptides = tryptic_digest("ACDKPEF")
        # The K is before P, so no cleavage there
        assert "ACDKPEF" in peptides

    def test_multiple_sites(self):
        peptides = tryptic_digest("AAKCDE")
        # Should produce ["AAK", "CDE"]
        assert "AAK" in peptides
        assert "CDE" in peptides

    def test_empty_sequence(self):
        assert tryptic_digest("") == []

    def test_no_cleavage_sites(self):
        # No K or R in sequence
        peptides = tryptic_digest("ACDEF")
        assert peptides == ["ACDEF"]

    def test_missed_cleavages_0(self):
        peptides = tryptic_digest("AAKBCR", missed_cleavages=0)
        assert "AAK" in peptides
        assert "BCR" in peptides
        assert "AAKBCR" not in peptides

    def test_missed_cleavages_1(self):
        peptides = tryptic_digest("AAKBCR", missed_cleavages=1)
        assert "AAKBCR" in peptides  # 1 missed cleavage

    def test_sample_vhh_digest_produces_peptides(self):
        peptides = tryptic_digest(SAMPLE_VHH)
        assert len(peptides) > 0
        # All peptides except the last should end with K or R
        for p in peptides[:-1]:
            assert p[-1] in "KR", f"Peptide {p!r} does not end in K/R"


# ── barcode pool integrity ─────────────────────────────────────────────────────

class TestBarcodePool:
    def test_pool_loads(self, generator):
        assert len(generator.pool) > 0, "Barcode pool should not be empty"

    def test_pool_has_200_plus_barcodes(self, generator):
        assert len(generator.pool) >= 250

    def test_all_pool_barcodes_pass_rules(self, generator):
        failures = [bc for bc in generator.pool if not _barcode_passes_rules(bc)]
        assert failures == [], f"Barcodes failing design rules: {failures[:5]}"

    def test_pool_barcodes_are_unique(self, generator):
        assert len(generator.pool) == len(set(generator.pool))


# ── barcode uniqueness / collision check ─────────────────────────────────────

class TestBarcodeUniqueness:
    def test_no_barcode_collides_with_sample_vhh_tryptic_peptides(self, generator):
        """Pool barcodes should not match tryptic peptides of the sample VHH."""
        vhh_tryptic = set(tryptic_digest(SAMPLE_VHH, missed_cleavages=1))
        collisions = [bc for bc in generator.pool if bc in vhh_tryptic]
        assert collisions == [], f"Colliding barcodes: {collisions}"

    def test_barcodes_in_pool_are_unique(self, generator):
        assert len(generator.pool) == len(set(generator.pool))


# ── assign_barcodes tests ─────────────────────────────────────────────────────

class TestAssignBarcodes:
    def test_returns_dataframe(self, generator, small_library):
        result = generator.assign_barcodes(small_library, top_n=5)
        assert isinstance(result, pd.DataFrame)

    def test_correct_number_of_rows(self, generator, small_library):
        result = generator.assign_barcodes(small_library, top_n=5)
        assert len(result) == 5

    def test_required_columns_present(self, generator, small_library):
        result = generator.assign_barcodes(small_library, top_n=5)
        for col in ("barcode_id", "barcode_peptide", "barcoded_sequence", "barcode_tryptic_peptide"):
            assert col in result.columns, f"Missing column: {col}"

    def test_barcode_ids_are_unique(self, generator, small_library):
        result = generator.assign_barcodes(small_library, top_n=10)
        ids = list(result["barcode_id"])
        assert len(ids) == len(set(ids))

    def test_barcoded_sequence_contains_linker(self, generator, small_library):
        result = generator.assign_barcodes(small_library, top_n=5, linker="GGS")
        for _, row in result.iterrows():
            assert "GGS" in row["barcoded_sequence"]

    def test_barcode_peptides_pass_design_rules(self, generator, small_library):
        result = generator.assign_barcodes(small_library, top_n=10)
        for _, row in result.iterrows():
            bp = row["barcode_peptide"]
            assert _barcode_passes_rules(bp), f"Barcode {bp!r} fails design rules"

    def test_top_n_selects_highest_scoring_variants(self, generator, small_library):
        result = generator.assign_barcodes(small_library, top_n=5)
        # Top 5 should include the 5 highest combined_score rows
        expected_top5 = small_library.nlargest(5, "combined_score")
        assert set(result["variant_id"]) == set(expected_top5["variant_id"])

    def test_empty_library_returns_empty(self, generator):
        empty = pd.DataFrame(columns=["combined_score", "aa_sequence", "variant_id"])
        result = generator.assign_barcodes(empty)
        assert len(result) == 0

    def test_missing_required_columns_raises(self, generator):
        bad_df = pd.DataFrame([{"variant_id": "V1", "combined_score": 0.5}])
        with pytest.raises(ValueError, match="aa_sequence"):
            generator.assign_barcodes(bad_df)


# ── generate_barcoded_fasta tests ─────────────────────────────────────────────

class TestBarcodedFasta:
    def test_returns_string(self, generator, small_library):
        barcoded = generator.assign_barcodes(small_library, top_n=3)
        fasta = generator.generate_barcoded_fasta(barcoded)
        assert isinstance(fasta, str)

    def test_fasta_has_header_lines(self, generator, small_library):
        barcoded = generator.assign_barcodes(small_library, top_n=3)
        fasta = generator.generate_barcoded_fasta(barcoded)
        lines = fasta.strip().split("\n")
        header_lines = [l for l in lines if l.startswith(">")]
        assert len(header_lines) == 3

    def test_fasta_sequences_match_barcoded_sequence(self, generator, small_library):
        barcoded = generator.assign_barcodes(small_library, top_n=3)
        fasta = generator.generate_barcoded_fasta(barcoded)
        lines = fasta.strip().split("\n")
        seq_lines = [l for l in lines if not l.startswith(">")]
        expected = list(barcoded["barcoded_sequence"])
        assert seq_lines == expected

    def test_fasta_header_contains_barcode_id(self, generator, small_library):
        barcoded = generator.assign_barcodes(small_library, top_n=3)
        fasta = generator.generate_barcoded_fasta(barcoded)
        lines = fasta.strip().split("\n")
        header_lines = [l for l in lines if l.startswith(">")]
        for h in header_lines:
            assert "BC-" in h


# ── generate_barcode_reference tests ─────────────────────────────────────────

class TestBarcodeReferenceTable:
    def test_returns_dataframe(self, generator, small_library):
        barcoded = generator.assign_barcodes(small_library, top_n=5)
        ref = generator.generate_barcode_reference(barcoded)
        assert isinstance(ref, pd.DataFrame)

    def test_expected_columns_present(self, generator, small_library):
        barcoded = generator.assign_barcodes(small_library, top_n=5)
        ref = generator.generate_barcode_reference(barcoded)
        for col in ("variant_id", "barcode_id", "barcode_peptide",
                    "barcode_tryptic_peptide", "neutral_mass_da",
                    "mz_1plus", "mz_2plus", "mz_3plus"):
            assert col in ref.columns, f"Missing column: {col}"

    def test_mz_values_are_positive(self, generator, small_library):
        barcoded = generator.assign_barcodes(small_library, top_n=5)
        ref = generator.generate_barcode_reference(barcoded)
        assert (ref["mz_1plus"] > 0).all()
        assert (ref["mz_2plus"] > 0).all()
        assert (ref["mz_3plus"] > 0).all()

    def test_mz_2plus_less_than_1plus(self, generator, small_library):
        """Higher charge state should give lower m/z."""
        barcoded = generator.assign_barcodes(small_library, top_n=5)
        ref = generator.generate_barcode_reference(barcoded)
        assert (ref["mz_2plus"] < ref["mz_1plus"]).all()

    def test_mz_3plus_less_than_2plus(self, generator, small_library):
        barcoded = generator.assign_barcodes(small_library, top_n=5)
        ref = generator.generate_barcode_reference(barcoded)
        assert (ref["mz_3plus"] < ref["mz_2plus"]).all()

    def test_hydrophobicity_column_present(self, generator, small_library):
        barcoded = generator.assign_barcodes(small_library, top_n=5)
        ref = generator.generate_barcode_reference(barcoded)
        assert "hydrophobicity" in ref.columns

    def test_source_column_present(self, generator, small_library):
        barcoded = generator.assign_barcodes(small_library, top_n=5)
        ref = generator.generate_barcode_reference(barcoded)
        assert "source" in ref.columns


# ── barcode source tracking tests ────────────────────────────────────────────

class TestBarcodeSourceTracking:
    def test_pool_barcodes_have_source(self, generator, small_library):
        """Barcodes drawn from the pool should be tagged as validated_proteomics."""
        result = generator.assign_barcodes(small_library, top_n=5)
        sources = result["barcode_source"]
        assert all(s == "validated_proteomics" for s in sources), (
            f"Expected all validated_proteomics, got: {list(sources)}"
        )

    def test_pool_entries_have_source_metadata(self, generator):
        """The pool should carry source metadata for every sequence."""
        assert len(generator._pool_sources) == len(generator.pool)
        for seq in generator.pool:
            assert seq in generator._pool_sources

    def test_barcode_source_column_in_assignment(self, generator, small_library):
        result = generator.assign_barcodes(small_library, top_n=3)
        assert "barcode_source" in result.columns


# ── biophysical distribution plot tests ──────────────────────────────────────

class TestPlotBarcodeDistributions:
    def test_returns_figure(self, generator, small_library):
        import matplotlib.figure
        barcoded = generator.assign_barcodes(small_library, top_n=5)
        ref = generator.generate_barcode_reference(barcoded)
        fig = generator.plot_barcode_distributions(ref)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_has_three_axes(self, generator, small_library):
        barcoded = generator.assign_barcodes(small_library, top_n=5)
        ref = generator.generate_barcode_reference(barcoded)
        fig = generator.plot_barcode_distributions(ref)
        assert len(fig.axes) == 3

    def test_plot_with_empty_ref_table(self, generator):
        empty = pd.DataFrame()
        fig = generator.plot_barcode_distributions(empty)
        assert fig is not None

    def test_plot_without_source_column(self, generator, small_library):
        """Plot should work even if source column is absent."""
        barcoded = generator.assign_barcodes(small_library, top_n=5)
        ref = generator.generate_barcode_reference(barcoded)
        ref = ref.drop(columns=["source"])
        fig = generator.plot_barcode_distributions(ref)
        assert len(fig.axes) == 3
