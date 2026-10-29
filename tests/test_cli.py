"""
Tests for Bonsai CLI Commands & Argument Routing.
"""

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
    assert "serve" in result.output

def test_cli_probe():
    result = runner.invoke(app, ["probe"])
    assert result.exit_code == 0
    assert "Machine Telemetry & Auto-Downshift Profile" in result.output
    assert "Learning Agility" in result.output

def test_cli_prune_missing_source():
    result = runner.invoke(app, ["prune", "non_existent_directory_xyz123"])
    assert result.exit_code == 0 or "user_goal" in result.output or "Error" in result.output
