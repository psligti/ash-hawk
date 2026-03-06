"""Test scenario new generator CLI command."""

from pathlib import Path
from click.testing import CliRunner

from ash_hawk.cli.main import cli


def test_scenario_new_help() -> None:
    """Test new command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["scenario", "new", "--help"])
    assert result.exit_code == 0
    assert "create" in result.output.lower() or "new" in result.output.lower()


def test_scenario_new_requires_type_and_name() -> None:
    """Test new command requires --type and --name."""
    runner = CliRunner()
    result = runner.invoke(cli, ["scenario", "new"])
    # Should fail due to missing required options
    assert result.exit_code != 0
