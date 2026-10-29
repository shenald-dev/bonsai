"""
Tests for Bonsai Universal Ingestion Engine.
"""

import json
import pytest
from pathlib import Path
from bonsai.core.ingest import ingest_source, DocumentBlock

def test_ingest_prompt_goal():
    blocks = ingest_source("Extract clean JSON from email receipts")
    assert len(blocks) == 1
    assert blocks[0].source == "user_goal"
    assert "receipts" in blocks[0].content

def test_ingest_files(tmp_path):
    # Create test text and json files
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("System event: worker pool down", encoding="utf-8")

    json_file = tmp_path / "data.json"
    json_file.write_text(json.dumps([{"id": 1, "msg": "alert1"}, {"id": 2, "msg": "alert2"}]), encoding="utf-8")

    blocks = ingest_source(str(tmp_path))
    assert len(blocks) == 3  # 1 txt + 2 json items
    sources = [b.source for b in blocks]
    assert any("sample.txt" in s for s in sources)
    assert any("data.json" in s for s in sources)

def test_ingest_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No valid data or text found"):
        ingest_source(str(empty_dir))
