import yaml
from click.testing import CliRunner

from ash_hawk.cli.main import cli


def test_scenario_new_generator(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "scenario",
            "new",
            "--type",
            "agentic_sdk",
            "--name",
            "tmp_hello",
            "--dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0

    scenario_path = tmp_path / "tmp_hello" / "tmp_hello.scenario.yaml"
    tool_mocks_dir = tmp_path / "tmp_hello" / "tool_mocks" / "tmp_hello"

    assert scenario_path.exists()
    assert tool_mocks_dir.exists()

    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    assert scenario["id"] == "tmp_hello"
    assert scenario["sut"]["type"] == "agentic_sdk"
    assert scenario["sut"]["adapter"] == "sdk_dawn_kestrel"
