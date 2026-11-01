"""
Tests for Bonsai CLI Commands & Argument Routing.
"""

from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from bonsai.cli import app

runner = CliRunner()

def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Cultivate massive foundation models" in result.output
    assert "probe" in result.output
    assert "prune" in result.output
    assert "exam" in result.output
    assert "merge" in result.output
    assert "serve" in result.output
    assert "ui" in result.output
    assert "version" in result.output

def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Bonsai" in result.output
    assert "version" in result.output

def test_cli_probe():
    result = runner.invoke(app, ["probe"])
    assert result.exit_code == 0
    assert "Machine Telemetry & Auto-Downshift Profile" in result.output
    assert "Learning Agility" in result.output

def test_cli_merge_help():
    result = runner.invoke(app, ["merge", "--help"])
    assert result.exit_code == 0
    assert "Fuses LoRA synaptic adapter weights" in result.output

def test_cli_ui_help():
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output
    assert "--host" in result.output
    assert "--no-browser" in result.output

@patch("bonsai.trainers.trainer.BonsaiTrainer.train")
@patch("bonsai.evaluators.exam.BonsaiExam.run_exam")
def test_cli_prune_mocked(mock_run_exam, mock_train, tmp_path):
    mock_train.return_value = {
        "initial_loss": 2.5,
        "final_loss": 1.1,
        "loss_reduction_pct": 56.0,
        "duration_seconds": 1.2,
        "peak_vram_mb": 1500.0
    }
    mock_scorecard = MagicMock()
    mock_scorecard.base_schema_validity_pct = 75.0
    mock_scorecard.bonsai_schema_validity_pct = 100.0
    mock_scorecard.base_task_accuracy_pct = 70.0
    mock_scorecard.bonsai_task_accuracy_pct = 95.0
    mock_scorecard.base_hallucination_pct = 25.0
    mock_scorecard.bonsai_hallucination_pct = 0.0
    mock_scorecard.base_tokens_sec = 25.0
    mock_scorecard.bonsai_tokens_sec = 30.0
    mock_scorecard.base_avg_latency_ms = 120.0
    mock_scorecard.bonsai_avg_latency_ms = 80.0
    mock_scorecard.peak_vram_mb = 1500.0
    mock_scorecard.sample_comparisons = []
    mock_run_exam.return_value = mock_scorecard

    output_dir = tmp_path / "bonsai_out"
    result = runner.invoke(app, ["prune", "Extract clean invoices from emails", "-o", str(output_dir)])

    assert result.exit_code == 0
    assert "Hardware Auto-Probe" in result.output
    assert "Universal Ingestion" in result.output
    assert "Teacher Cultivation" in result.output
    assert "Pocket Distillation" in result.output
    assert "Blind Holdout Proof Machine" in result.output
    assert (output_dir / "bonsai_manifest.json").exists()
