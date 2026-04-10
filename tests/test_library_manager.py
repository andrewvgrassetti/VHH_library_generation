import pytest
import os
import json
import pandas as pd
from vhh_library.library_manager import LibraryManager

@pytest.fixture
def manager():
    return LibraryManager()

def test_session_id_format(manager):
    sid = manager.session_id
    assert isinstance(sid, str)
    assert len(sid) == 15
    assert "_" in sid

def test_create_variant_id(manager):
    vid = manager.create_variant_id(1)
    assert vid.startswith("VHH-")
    assert vid.endswith("-000001")

def test_save_load_roundtrip(manager):
    data = {"test_key": "test_value", "number": 42}
    output_dir = "sessions_test"
    paths = manager.save_session(data, output_dir=output_dir)
    assert "json" in paths
    loaded = manager.load_session(paths["json"])
    assert loaded["test_key"] == "test_value"
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

def test_export_fasta(manager):
    df = pd.DataFrame({
        "variant_id": ["VHH-001", "VHH-002"],
        "aa_sequence": ["QVQLVES", "QVQLVQS"]
    })
    filepath = "test_export.fasta"
    manager.export_fasta(df, filepath)
    with open(filepath, "r") as f:
        content = f.read()
    assert ">VHH-001" in content
    assert "QVQLVES" in content
    os.remove(filepath)
