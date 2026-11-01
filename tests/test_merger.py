"""
Tests for Bonsai LoRA Adapter Fusion & Merging Engine.
"""

from unittest.mock import MagicMock, patch
from pathlib import Path
from bonsai.trainers.merger import BonsaiMerger


def test_merger_initialization(tmp_path):
    merger = BonsaiMerger(
        base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        adapter_path=str(tmp_path / "adapter"),
        output_dir=str(tmp_path / "merged"),
        device="cpu"
    )
    assert merger.base_model == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert merger.device == "cpu"
    assert merger.output_dir == tmp_path / "merged"


@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("peft.PeftModel.from_pretrained")
def test_merger_execution(mock_peft_from_pretrained, mock_base_from_pretrained, mock_tok_from_pretrained, tmp_path):
    # Setup mock tokenizer
    mock_tok = MagicMock()
    mock_tok_from_pretrained.return_value = mock_tok

    # Setup mock base model
    mock_base = MagicMock()
    mock_base_from_pretrained.return_value = mock_base

    # Setup mock PEFT model and merge_and_unload
    mock_peft = MagicMock()
    mock_merged = MagicMock()
    mock_peft.merge_and_unload.return_value = mock_merged
    mock_peft_from_pretrained.return_value.to.return_value = mock_peft

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir(parents=True)
    merged_dir = tmp_path / "merged"

    merger = BonsaiMerger(
        base_model="test-base-model",
        adapter_path=str(adapter_dir),
        output_dir=str(merged_dir),
        device="cpu"
    )

    telemetry = merger.merge()

    assert telemetry["base_model"] == "test-base-model"
    assert telemetry["merged_output_dir"] == str(merged_dir)
    assert "duration_seconds" in telemetry
    assert "total_size_mb" in telemetry

    # Verify calls
    mock_peft.merge_and_unload.assert_called_once()
    mock_merged.save_pretrained.assert_called_once_with(str(merged_dir), safe_serialization=True)
    mock_tok.save_pretrained.assert_called_once_with(str(merged_dir))
