"""Test scenario record/replay CLI commands."""

from click.testing import CliRunner

from ash_hawk.cli.main import cli


def test_scenario_record_help() -> None:
    """Test record command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["scenario", "record", "--help"])
    assert result.exit_code == 0
    assert "record" in result.output.lower()


def test_scenario_replay_help() -> None:
    """Test replay command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["scenario", "replay", "--help"])
    assert result.exit_code == 0
    assert "replay" in result.output.lower()


def test_scenario_record_requires_sut() -> None:
    """Test record command requires --sut."""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["scenario", "record", "examples/scenarios/hello_world.scenario.yaml"]
    )
    # Should fail due to missing --sut
    assert result.exit_code != 0


def test_scenario_replay_requires_run() -> None:
    """Test replay command requires --run."""
    runner = CliRunner()
    result = runner.invoke(cli, ["scenario", "replay"])
    # Should fail due to missing --run
    assert result.exit_code != 0
