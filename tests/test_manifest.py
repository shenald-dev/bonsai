"""
Tests for Bonsai Manifest State Serialization and Loading.
"""

from pathlib import Path
from bonsai.core.manifest import BonsaiManifest, load_manifest

def test_manifest_roundtrip(tmp_path):
    manifest = BonsaiManifest(
        version="0.1.0",
        created_at="2026-10-29T12:00:00",
        output_dir=str(tmp_path),
        base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        adapter_dir=str(tmp_path / "adapter"),
        train_dataset=str(tmp_path / "cultivate.jsonl"),
        exam_dataset=str(tmp_path / "exam.jsonl"),
        modelfile=str(tmp_path / "Modelfile"),
        runner_script=str(tmp_path / "serve.py"),
        report_svg=str(tmp_path / "bonsai-report.svg"),
        report_html=str(tmp_path / "bonsai-report.html"),
        hardware={"gpu_name": "Test GPU", "vram_mb": 4096},
        training_telemetry={"initial_loss": 3.5, "final_loss": 2.1},
        exam_scorecard={"accuracy": 95.0}
    )

    manifest_file = manifest.save(tmp_path)
    assert manifest_file.exists()

    loaded = load_manifest(tmp_path)
    assert loaded.version == "0.1.0"
    assert loaded.base_model == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert loaded.training_telemetry["initial_loss"] == 3.5
    assert loaded.hardware["gpu_name"] == "Test GPU"
