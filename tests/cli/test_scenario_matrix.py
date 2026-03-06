"""Test scenario matrix CLI command."""

from click.testing import CliRunner

from ash_hawk.cli.main import cli


def test_scenario_matrix_help() -> None:
    """Test matrix command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["scenario", "matrix", "--help"])
    assert result.exit_code == 0
    assert "matrix" in result.output.lower()


def test_scenario_matrix_requires_sut() -> None:
    """Test matrix command requires --sut."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["scenario", "matrix", "examples/scenarios/", "--policies", "react", "--models", "m1"],
    )
    # Should fail due to missing --sut
    assert result.exit_code != 0 or "sut" in result.output.lower()
