import re
import tempfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from ash_hawk.cli.main import cli
from ash_hawk.types import EvalOutcome, EvalTranscript


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def mock_dawn_runner(monkeypatch):
    class FakeDawnKestrelAgentRunner:
        def __init__(self, provider: str, model: str, **kwargs):
            self.provider = provider
            self.model = model
            self.kwargs = kwargs

        async def run(self, task, policy_enforcer, config):
            del policy_enforcer
            del config
            prompt = task.input if isinstance(task.input, str) else str(task.input)
            return (
                EvalTranscript(
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "Mock response"},
                    ],
                    agent_response="Mock response",
                ),
                EvalOutcome.success(),
            )

    monkeypatch.setattr("ash_hawk.agents.DawnKestrelAgentRunner", FakeDawnKestrelAgentRunner)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_storage(temp_dir):
    storage_path = Path(temp_dir) / "storage"
    storage_path.mkdir()
    return str(storage_path)


@pytest.fixture
def sample_suite_file(temp_dir):
    suite_path = Path(temp_dir) / "suites" / "test-suite.yaml"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_data = {
        "id": "test-suite",
        "name": "Test Suite",
        "description": "A test suite for CLI testing",
        "agent": {"name": "build"},
        "tasks": [
            {
                "id": "task-001",
                "description": "First task",
                "input": "What is 1 + 1?",
            },
            {
                "id": "task-002",
                "description": "Second task",
                "input": "What is 2 + 2?",
            },
        ],
    }
    with open(suite_path, "w") as f:
        yaml.dump(suite_data, f)
    return str(suite_path)


class TestCliMain:
    def test_cli_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "ash-hawk" in result.output

    def test_cli_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "run" in result.output
        assert "list" in result.output
        assert "report" in result.output


class TestCliInit:
    def test_init_creates_file(self, runner, temp_storage):
        output_path = Path(temp_storage) / "new-suite.yaml"
        result = runner.invoke(cli, ["init", str(output_path)])
        assert result.exit_code == 0
        assert output_path.exists()

    def test_init_file_content(self, runner, temp_storage):
        output_path = Path(temp_storage) / "new-suite.yaml"
        result = runner.invoke(cli, ["init", str(output_path), "--name", "my-custom-suite"])
        assert result.exit_code == 0

        with open(output_path) as f:
            data = yaml.safe_load(f)

        assert data["id"] == "my-custom-suite"
        assert "tasks" in data
        assert len(data["tasks"]) >= 1

    def test_init_refuses_overwrite(self, runner, temp_storage):
        output_path = Path(temp_storage) / "existing-suite.yaml"
        output_path.write_text("existing content")

        result = runner.invoke(cli, ["init", str(output_path)])
        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_init_force_overwrites(self, runner, temp_storage):
        output_path = Path(temp_storage) / "existing-suite.yaml"
        output_path.write_text("existing content")

        result = runner.invoke(cli, ["init", str(output_path), "--force"])
        assert result.exit_code == 0

        with open(output_path) as f:
            data = yaml.safe_load(f)
        assert "id" in data


class TestCliRun:
    def test_run_uses_suite_agent_default(self, runner, sample_suite_file, temp_storage):
        result = runner.invoke(
            cli,
            [
                "run",
                sample_suite_file,
                "--storage",
                temp_storage,
            ],
        )
        assert result.exit_code == 0
        assert "Suite:" in result.output

    def test_run_suite_basic(self, runner, sample_suite_file, temp_storage):
        result = runner.invoke(
            cli,
            [
                "run",
                sample_suite_file,
                "--storage",
                temp_storage,
                "--agent",
                "build",
            ],
        )
        assert result.exit_code == 0

    def test_cli_agent_overrides_suite_agent_default(self, runner, sample_suite_file, temp_storage):
        result = runner.invoke(
            cli,
            ["run", sample_suite_file, "--storage", temp_storage, "--agent", "general"],
        )
        assert result.exit_code == 0

        match = re.search(r"Run ID: (run-[a-f0-9]+)", result.output)
        assert match
        run_id = match.group(1)

        summary_path = Path(temp_storage) / "test-suite" / "runs" / run_id / "summary.json"
        assert summary_path.exists()
        summary = yaml.safe_load(summary_path.read_text())
        assert summary["envelope"]["agent_name"] == "general"

    def test_run_uses_custom_runner_from_suite(self, runner, temp_dir, temp_storage):
        suite_dir = Path(temp_dir) / "suites"
        suite_dir.mkdir(parents=True, exist_ok=True)

        runner_file = suite_dir / "custom_runner.py"
        runner_file.write_text(
            "from ash_hawk.types import EvalOutcome, EvalTranscript\n"
            "\n"
            "class CustomRunner:\n"
            "    def __init__(self, **kwargs):\n"
            "        self.kwargs = kwargs\n"
            "\n"
            "    async def run(self, task, policy_enforcer, config):\n"
            "        del task\n"
            "        del policy_enforcer\n"
            "        del config\n"
            "        return EvalTranscript(agent_response='custom'), EvalOutcome.success()\n"
        )

        suite_path = suite_dir / "suite-custom.yaml"
        suite_path.write_text(
            yaml.safe_dump(
                {
                    "id": "suite-custom",
                    "name": "Custom Runner Suite",
                    "agent": {
                        "class": "CustomRunner",
                        "location": "./custom_runner.py",
                    },
                    "tasks": [{"id": "task-1", "input": "hello"}],
                }
            )
        )

        result = runner.invoke(
            cli,
            ["run", str(suite_path), "--storage", temp_storage],
        )
        assert result.exit_code == 0

    def test_run_fails_without_cli_or_suite_agent(self, runner, temp_dir, temp_storage):
        suite_path = Path(temp_dir) / "suites" / "no-agent-suite.yaml"
        suite_path.parent.mkdir(parents=True, exist_ok=True)
        with open(suite_path, "w") as f:
            yaml.safe_dump(
                {
                    "id": "no-agent-suite",
                    "name": "No Agent Suite",
                    "tasks": [{"id": "task-1", "input": "hello"}],
                },
                f,
            )

        result = runner.invoke(
            cli,
            [
                "run",
                str(suite_path),
                "--storage",
                temp_storage,
            ],
        )
        assert result.exit_code != 0
        assert "No agent configured" in result.output
        assert "Suite:" in result.output

    def test_run_with_parallelism(self, runner, sample_suite_file, temp_storage):
        result = runner.invoke(
            cli,
            [
                "run",
                sample_suite_file,
                "--storage",
                temp_storage,
                "--parallelism",
                "2",
                "--agent",
                "build",
            ],
        )
        assert result.exit_code == 0

    def test_run_passes_mcp_servers_to_default_runner(
        self, runner, temp_dir, temp_storage, monkeypatch
    ):
        captured: dict[str, object] = {}

        class CapturingDawnKestrelAgentRunner:
            def __init__(self, provider: str, model: str, **kwargs):
                captured["provider"] = provider
                captured["model"] = model
                captured["kwargs"] = kwargs

            async def run(self, task, policy_enforcer, config):
                del task
                del policy_enforcer
                del config
                return EvalTranscript(agent_response="ok"), EvalOutcome.success()

        monkeypatch.setattr(
            "ash_hawk.agents.DawnKestrelAgentRunner", CapturingDawnKestrelAgentRunner
        )

        suite_path = Path(temp_dir) / "suites" / "mcp-suite.yaml"
        suite_path.parent.mkdir(parents=True, exist_ok=True)
        suite_path.write_text(
            yaml.safe_dump(
                {
                    "id": "mcp-suite",
                    "name": "MCP Suite",
                    "agent": {
                        "name": "build",
                        "mcp_servers": [
                            {
                                "name": "note-lark",
                                "command": "note-lark-mcp-stdio",
                            }
                        ],
                    },
                    "tasks": [{"id": "task-1", "input": "hello"}],
                }
            )
        )

        result = runner.invoke(
            cli,
            ["run", str(suite_path), "--storage", temp_storage],
        )
        assert result.exit_code == 0
        assert captured["provider"] == "zai-coding-plan"
        assert isinstance(captured["kwargs"], dict)
        assert captured["kwargs"] == {
            "mcp_servers": [
                {
                    "name": "note-lark",
                    "command": "note-lark-mcp-stdio",
                    "args": [],
                    "env": {},
                }
            ]
        }

    def test_run_nonexistent_suite(self, runner, temp_storage):
        result = runner.invoke(
            cli,
            ["run", "/nonexistent/path.yaml", "--storage", temp_storage, "--agent", "build"],
        )
        assert result.exit_code != 0

    def test_run_creates_storage(self, runner, sample_suite_file, temp_storage):
        storage_path = Path(temp_storage) / "new-storage"
        result = runner.invoke(
            cli,
            ["run", sample_suite_file, "--storage", str(storage_path), "--agent", "build"],
        )
        assert result.exit_code == 0
        assert storage_path.exists()


class TestCliList:
    def test_list_suites_empty(self, runner, temp_storage):
        result = runner.invoke(cli, ["list", "--storage", temp_storage])
        assert result.exit_code == 0
        assert "No suites found" in result.output

    def test_list_suites_with_data(self, runner, sample_suite_file, temp_storage):
        runner.invoke(
            cli, ["run", sample_suite_file, "--storage", temp_storage, "--agent", "build"]
        )
        result = runner.invoke(cli, ["list", "--storage", temp_storage])
        assert result.exit_code == 0
        assert "test-suite" in result.output

    def test_list_runs(self, runner, sample_suite_file, temp_storage):
        runner.invoke(
            cli, ["run", sample_suite_file, "--storage", temp_storage, "--agent", "build"]
        )
        result = runner.invoke(cli, ["list", "--runs", "--storage", temp_storage])
        assert result.exit_code == 0


class TestCliReport:
    def test_report_nonexistent_run(self, runner, temp_storage):
        result = runner.invoke(
            cli,
            ["report", "run-nonexistent", "--storage", temp_storage],
        )
        assert result.exit_code == 1

    def test_report_after_run(self, runner, sample_suite_file, temp_storage):
        run_result = runner.invoke(
            cli,
            ["run", sample_suite_file, "--storage", temp_storage, "--agent", "build"],
        )
        assert run_result.exit_code == 0

        import re

        match = re.search(r"Run ID: (run-[a-f0-9]+)", run_result.output)
        assert match, f"Could not find Run ID in output: {run_result.output}"
        run_id = match.group(1)

        result = runner.invoke(
            cli,
            ["report", run_id, "--storage", temp_storage],
        )
        assert result.exit_code == 0
        assert "Run Report" in result.output

    def test_report_verbose(self, runner, sample_suite_file, temp_storage):
        run_result = runner.invoke(
            cli,
            ["run", sample_suite_file, "--storage", temp_storage, "--agent", "build"],
        )
        assert run_result.exit_code == 0

        import re

        match = re.search(r"Run ID: (run-[a-f0-9]+)", run_result.output)
        assert match
        run_id = match.group(1)

        result = runner.invoke(
            cli,
            ["report", run_id, "--storage", temp_storage, "--verbose"],
        )
        assert result.exit_code == 0
        assert "Trial Details" in result.output


class TestCliReportCI:
    """Tests for confidence interval display in report command."""

    def test_report_shows_confidence_interval_for_pass_rate(
        self, runner, sample_suite_file, temp_storage
    ):
        """Report should display 95% CI with pass rate, e.g., '100% [34%-100%]'."""
        run_result = runner.invoke(
            cli,
            ["run", sample_suite_file, "--storage", temp_storage, "--agent", "build"],
        )
        assert run_result.exit_code == 0

        import re

        match = re.search(r"Run ID: (run-[a-f0-9]+)", run_result.output)
        assert match
        run_id = match.group(1)

        result = runner.invoke(
            cli,
            ["report", run_id, "--storage", temp_storage],
        )
        assert result.exit_code == 0
        # CI should be displayed as [lower%-upper%] after the pass rate
        assert re.search(r"\d+\.\d% \[\d+%-\d+%\]", result.output), (
            f"Expected CI format in output, got: {result.output}"
        )

    def test_report_ci_edge_case_zero_trials(self, runner, temp_storage, temp_dir):
        """Report with 0 completed trials should handle CI gracefully."""
        from ash_hawk.cli.report import format_pass_rate_with_ci

        # 0 trials should not crash and should show reasonable bounds
        formatted = format_pass_rate_with_ci(pass_rate=0.0, successes=0, total=0)
        # When 0 trials, we can't compute meaningful CI - should show pass rate only
        assert "0.0%" in formatted

    def test_report_ci_edge_case_all_pass(self, runner, temp_storage):
        """Report with 100% pass rate should show asymmetric CI."""
        from ash_hawk.cli.report import format_pass_rate_with_ci

        # 10/10 pass should show CI that extends below 100%
        formatted = format_pass_rate_with_ci(pass_rate=1.0, successes=10, total=10)
        assert "100%" in formatted
        assert "[" in formatted and "]" in formatted

    def test_report_ci_edge_case_all_fail(self, runner, temp_storage):
        """Report with 0% pass rate should show asymmetric CI."""
        from ash_hawk.cli.report import format_pass_rate_with_ci

        # 0/10 pass should show CI that extends above 0%
        formatted = format_pass_rate_with_ci(pass_rate=0.0, successes=0, total=10)
        assert "0.0%" in formatted
        assert "[" in formatted and "]" in formatted


class TestCliIntegration:
    def test_full_workflow(self, runner, temp_dir):
        storage_path = Path(temp_dir) / "storage"
        storage_path.mkdir()

        suite_path = Path(temp_dir) / "suites" / "workflow-suite.yaml"
        suite_path.parent.mkdir(parents=True, exist_ok=True)

        init_result = runner.invoke(
            cli,
            ["init", str(suite_path), "--name", "workflow"],
        )
        assert init_result.exit_code == 0

        run_result = runner.invoke(
            cli,
            [
                "run",
                str(suite_path),
                "--storage",
                str(storage_path),
                "--agent",
                "build",
            ],
        )
        assert run_result.exit_code == 0

        list_result = runner.invoke(cli, ["list", "--storage", str(storage_path)])
        assert list_result.exit_code == 0
        assert "workflow" in list_result.output

        list_runs_result = runner.invoke(
            cli,
            ["list", "--runs", "--storage", str(storage_path)],
        )
        assert list_runs_result.exit_code == 0
