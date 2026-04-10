import pytest
from vhh_library.sequence import VHHSequence

SAMPLE_VHH = "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"

def test_valid_vhh():
    seq = VHHSequence(SAMPLE_VHH)
    assert seq.validation_result["valid"] == True

def test_invalid_too_short():
    seq = VHHSequence("QVQLVES")
    assert seq.validation_result["valid"] == False
    assert len(seq.validation_result["errors"]) > 0

def test_invalid_too_long():
    seq = VHHSequence("Q" * 200)
    assert seq.validation_result["valid"] == False

def test_imgt_numbering():
    seq = VHHSequence(SAMPLE_VHH)
    assert isinstance(seq.imgt_numbered, dict)
    assert 1 in seq.imgt_numbered
    assert seq.imgt_numbered[1] == "Q"

def test_get_regions():
    seq = VHHSequence(SAMPLE_VHH)
    regions = seq.regions
    assert "FR1" in regions
    assert "CDR1" in regions
    assert "FR2" in regions
    assert "CDR2" in regions
    assert "FR3" in regions
    assert "CDR3" in regions
    assert "FR4" in regions

def test_cdr_positions():
    seq = VHHSequence(SAMPLE_VHH)
    cdrs = seq.cdr_positions
    assert isinstance(cdrs, set)
    assert len(cdrs) > 0
    assert 26 in cdrs

def test_length():
    seq = VHHSequence(SAMPLE_VHH)
    assert seq.length == len(SAMPLE_VHH)
