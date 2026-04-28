"""CLI unit and integration tests.

Run fast tests: pytest -m unit
Skip integration: pytest -m "not integration"
"""

import logging
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from ash_hawk.agents.agent_resolver import AgentResolution
from ash_hawk.cli.main import cli
from ash_hawk.context import setup_eval_logging
from ash_hawk.improve.loop import ImprovementResult
from ash_hawk.thin_runtime import RuntimeGoal
from ash_hawk.types import EvalOutcome, EvalTranscript


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
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


@pytest.mark.unit
class TestCliMain:
    def test_cli_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "ash-hawk" in result.output

    def test_cli_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "improve" in result.output
        assert "thin" in result.output
        assert "thin-runtime" in result.output

    def test_thin_runtime_run_command(self, runner, monkeypatch, tmp_path):
        captured: dict[str, object] = {}

        class FakeHarness:
            def execute(self, goal, **kwargs):
                captured["goal"] = goal
                captured.update(kwargs)
                return SimpleNamespace(
                    run_id="test-run",
                    agent=SimpleNamespace(name="coordinator"),
                    success=True,
                    error=None,
                    selected_tool_names=["select_next_action"],
                    artifact_dir=str(tmp_path / "artifacts"),
                    model_dump=lambda mode="json": {
                        "run_id": "test-run",
                        "success": True,
                        "agent": {"name": "coordinator"},
                    },
                )

        monkeypatch.setattr(
            "ash_hawk.cli.thin_runtime.create_default_harness",
            lambda **_: FakeHarness(),
        )

        result = runner.invoke(
            cli,
            [
                "thin-runtime",
                "run",
                "Test runtime goal",
                "--goal-id",
                "goal-123",
                "--agent",
                "coordinator",
                "--max-iterations",
                "5",
                "--workdir",
                str(tmp_path),
                "--skill",
                "process-control",
                "--tool",
                "select_next_action",
            ],
        )

        assert result.exit_code == 0
        assert isinstance(captured["goal"], RuntimeGoal)
        assert captured["scenario_path"] is None
        assert captured["requested_skills"] == ["process-control"]
        assert captured["tool_execution_order"] == ["select_next_action"]
        assert "test-run" in result.output

    def test_thin_runtime_run_uses_agent_default_goal(self, runner, monkeypatch, tmp_path):
        captured: dict[str, object] = {}

        class FakeHarness:
            def __init__(self) -> None:
                self.catalog = SimpleNamespace(
                    agents=[
                        SimpleNamespace(
                            name="improver",
                            default_goal_description="Improve the agent against the specified evals until the measured outcome meaningfully improves.",
                        )
                    ]
                )

            def execute(self, goal, **kwargs):
                captured["goal"] = goal
                captured.update(kwargs)
                return SimpleNamespace(run_id="default-goal-run")

        monkeypatch.setattr(
            "ash_hawk.cli.thin_runtime.create_default_harness",
            lambda **_: FakeHarness(),
        )

        result = runner.invoke(
            cli,
            [
                "thin-runtime",
                "run",
                "--agent",
                "improver",
                "--workdir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        assert isinstance(captured["goal"], RuntimeGoal)
        assert captured["goal"].description == (
            "Improve the agent against the specified evals until the measured outcome meaningfully improves."
        )
        assert "default-goal-run" in result.output

    def test_thin_runtime_run_errors_without_goal_or_agent_default(
        self, runner, monkeypatch, tmp_path
    ):
        class FakeHarness:
            def __init__(self) -> None:
                self.catalog = SimpleNamespace(
                    agents=[SimpleNamespace(name="coordinator", default_goal_description="")]
                )

        monkeypatch.setattr(
            "ash_hawk.cli.thin_runtime.create_default_harness",
            lambda **_: FakeHarness(),
        )

        result = runner.invoke(
            cli,
            [
                "thin-runtime",
                "run",
                "--agent",
                "coordinator",
                "--workdir",
                str(tmp_path),
            ],
        )

        assert result.exit_code != 0
        assert "has no default_goal_description" in result.output

    def test_thin_runtime_run_explicit_goal_overrides_agent_default(
        self, runner, monkeypatch, tmp_path
    ):
        captured: dict[str, object] = {}

        class FakeHarness:
            def __init__(self) -> None:
                self.catalog = SimpleNamespace(
                    agents=[
                        SimpleNamespace(
                            name="improver",
                            default_goal_description="Improve the default eval target.",
                        )
                    ]
                )

            def execute(self, goal, **kwargs):
                captured["goal"] = goal
                captured.update(kwargs)
                return SimpleNamespace(run_id="explicit-goal-run")

        monkeypatch.setattr(
            "ash_hawk.cli.thin_runtime.create_default_harness",
            lambda **_: FakeHarness(),
        )

        result = runner.invoke(
            cli,
            [
                "thin-runtime",
                "run",
                "Use the user supplied goal",
                "--agent",
                "improver",
                "--workdir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        assert isinstance(captured["goal"], RuntimeGoal)
        assert captured["goal"].description == "Use the user supplied goal"
        assert "explicit-goal-run" in result.output

    def test_improve_command_shows_process_header_and_mutes_logs(
        self, runner, monkeypatch, tmp_path
    ):
        agent_dir = tmp_path / "bolt_merlin" / "agent"
        agent_dir.mkdir(parents=True)

        async def fake_improve(**kwargs):
            logging.getLogger("ash_hawk.improve.loop").warning("suppressed logger noise")
            kwargs["console"].print("[cyan]Simulated step:[/cyan] preparing worktree")
            return ImprovementResult(
                iterations=1,
                initial_score=0.25,
                final_score=1.0,
                initial_pass_rate=0.25,
                final_pass_rate=1.0,
                patches_proposed=[],
                patches_applied=["prompt.md"],
                mutation_history=[],
                iteration_logs=[],
                convergence_achieved=True,
                stop_reasons=[],
            )

        monkeypatch.setattr(
            "ash_hawk.cli.main.resolve_agent_path",
            lambda _agent, _workdir: AgentResolution(
                path=agent_dir,
                name="bolt_merlin",
                resolved_from="cli_path",
            ),
        )
        monkeypatch.setattr("ash_hawk.improve.loop.improve", fake_improve)

        try:
            result = runner.invoke(cli, ["improve", "suite.yaml", "--agent", "bolt_merlin"])
        finally:
            setup_eval_logging(logging.INFO)

        assert result.exit_code == 0
        assert "Improve Run" in result.output
        assert "Console shows process steps only" in result.output
        assert "Simulated step:" in result.output
        assert "suppressed logger noise" not in result.output

    def test_improve_directory_expands_to_scenarios(self, runner, monkeypatch, tmp_path):
        agent_dir = tmp_path / "bolt_merlin" / "agent"
        agent_dir.mkdir(parents=True)
        scenarios_dir = tmp_path / "evals" / "scenarios"
        scenarios_dir.mkdir(parents=True)
        a = scenarios_dir / "a.scenario.yaml"
        b = scenarios_dir / "b.scenario.yaml"
        a.write_text("id: a\n", encoding="utf-8")
        b.write_text("id: b\n", encoding="utf-8")

        captured: dict[str, object] = {}

        async def fake_improve(**kwargs):
            captured.update(kwargs)
            return ImprovementResult(
                iterations=1,
                initial_score=0.25,
                final_score=1.0,
                initial_pass_rate=0.25,
                final_pass_rate=1.0,
                patches_proposed=[],
                patches_applied=[],
                mutation_history=[],
                iteration_logs=[],
                convergence_achieved=True,
                stop_reasons=[],
            )

        monkeypatch.setattr(
            "ash_hawk.cli.main.resolve_agent_path",
            lambda _agent, _workdir: AgentResolution(
                path=agent_dir,
                name="bolt_merlin",
                resolved_from="cli_path",
            ),
        )
        monkeypatch.setattr("ash_hawk.improve.loop.improve", fake_improve)

        result = runner.invoke(cli, ["improve", str(scenarios_dir), "--agent", "bolt_merlin"])

        assert result.exit_code == 0
        assert captured["suite_path"] == [str(a), str(b)]
        assert "Resolved paths:" in result.output

    def test_improve_glob_expands_to_scenarios(self, runner, monkeypatch, tmp_path):
        agent_dir = tmp_path / "bolt_merlin" / "agent"
        agent_dir.mkdir(parents=True)
        scenarios_dir = tmp_path / "evals" / "scenarios"
        scenarios_dir.mkdir(parents=True)
        a = scenarios_dir / "a.scenario.yaml"
        b = scenarios_dir / "b.scenario.yaml"
        a.write_text("id: a\n", encoding="utf-8")
        b.write_text("id: b\n", encoding="utf-8")

        captured: dict[str, object] = {}

        async def fake_improve(**kwargs):
            captured.update(kwargs)
            return ImprovementResult(
                iterations=1,
                initial_score=0.25,
                final_score=1.0,
                initial_pass_rate=0.25,
                final_pass_rate=1.0,
                patches_proposed=[],
                patches_applied=[],
                mutation_history=[],
                iteration_logs=[],
                convergence_achieved=True,
                stop_reasons=[],
            )

        monkeypatch.setattr(
            "ash_hawk.cli.main.resolve_agent_path",
            lambda _agent, _workdir: AgentResolution(
                path=agent_dir,
                name="bolt_merlin",
                resolved_from="cli_path",
            ),
        )
        monkeypatch.setattr("ash_hawk.improve.loop.improve", fake_improve)

        result = runner.invoke(
            cli,
            ["improve", str(scenarios_dir / "*.scenario.yaml"), "--agent", "bolt_merlin"],
        )

        assert result.exit_code == 0
        assert captured["suite_path"] == [str(a), str(b)]

    def test_improve_pack_expands_to_scenarios(self, runner, monkeypatch, tmp_path):
        agent_dir = tmp_path / "bolt_merlin" / "agent"
        agent_dir.mkdir(parents=True)
        scenarios_dir = tmp_path / "evals" / "scenarios"
        foundation_dir = scenarios_dir / "curriculum" / "foundation"
        foundation_dir.mkdir(parents=True)
        a = foundation_dir / "a.scenario.yaml"
        b = foundation_dir / "b.scenario.yaml"
        a.write_text(
            "schema_version: v1\nid: a\nsut: {type: agentic_sdk, adapter: mock}\n", encoding="utf-8"
        )
        b.write_text(
            "schema_version: v1\nid: b\nsut: {type: agentic_sdk, adapter: mock}\n", encoding="utf-8"
        )
        pack = scenarios_dir / "phase1_evidence_pack.yaml"
        pack.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "v1",
                    "id": "phase1",
                    "scenarios": [
                        {"scenario": "./curriculum/foundation/a.scenario.yaml"},
                        {"scenario": "./curriculum/foundation/b.scenario.yaml"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        captured: dict[str, object] = {}

        async def fake_improve(**kwargs):
            captured.update(kwargs)
            return ImprovementResult(
                iterations=1,
                initial_score=0.25,
                final_score=1.0,
                initial_pass_rate=0.25,
                final_pass_rate=1.0,
                patches_proposed=[],
                patches_applied=[],
                mutation_history=[],
                iteration_logs=[],
                convergence_achieved=True,
                stop_reasons=[],
            )

        monkeypatch.setattr(
            "ash_hawk.cli.main.resolve_agent_path",
            lambda _agent, _workdir: AgentResolution(
                path=agent_dir,
                name="bolt_merlin",
                resolved_from="cli_path",
            ),
        )
        monkeypatch.setattr("ash_hawk.improve.loop.improve", fake_improve)

        result = runner.invoke(cli, ["improve", str(pack), "--agent", "bolt_merlin"])

        assert result.exit_code == 0
        assert captured["suite_path"] == [str(a.resolve()), str(b.resolve())]
        assert "Resolved paths:" in result.output

    def test_backfill_memory_command_prints_summary(self, runner, monkeypatch, tmp_path):
        captured: dict[str, object] = {}

        def fake_backfill(
            runs_dir: Path,
            memory_dir: Path,
            *,
            force: bool = False,
            include_improve_cycle: bool = False,
        ):
            captured["runs_dir"] = runs_dir
            captured["memory_dir"] = memory_dir
            captured["force"] = force
            captured["include_improve_cycle"] = include_improve_cycle
            return {"imported_episodes": 7, "semantic_rules": 3, "skip_precision_basis": 2500}

        monkeypatch.setattr("ash_hawk.improve.loop.backfill_memory", fake_backfill)

        result = runner.invoke(
            cli,
            [
                "backfill-memory",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--memory-dir",
                str(tmp_path / "memory"),
            ],
        )

        assert result.exit_code == 0
        assert captured["runs_dir"] == tmp_path / "runs"
        assert captured["memory_dir"] == tmp_path / "memory"
        assert captured["include_improve_cycle"] is False
        assert "Imported episodes:" in result.output


@pytest.mark.integration
class TestCliRun:
    def test_run_uses_suite_agent_default(
        self, runner, sample_suite_file, temp_storage, mock_dawn_runner
    ):
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

    def test_run_suite_basic(self, runner, sample_suite_file, temp_storage, mock_dawn_runner):
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

    def test_cli_agent_overrides_suite_agent_default(
        self, runner, sample_suite_file, temp_storage, mock_dawn_runner
    ):
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

    def test_run_with_parallelism(self, runner, sample_suite_file, temp_storage, mock_dawn_runner):
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
            ],
            "project_root": suite_path.parent.resolve(),
        }

    @pytest.mark.unit
    def test_run_nonexistent_suite(self, runner, temp_storage):
        result = runner.invoke(
            cli,
            ["run", "/nonexistent/path.yaml", "--storage", temp_storage, "--agent", "build"],
        )
        assert result.exit_code != 0

    def test_run_creates_storage(self, runner, sample_suite_file, temp_storage, mock_dawn_runner):
        storage_path = Path(temp_storage) / "new-storage"
        result = runner.invoke(
            cli,
            ["run", sample_suite_file, "--storage", str(storage_path), "--agent", "build"],
        )
        assert result.exit_code == 0
        assert storage_path.exists()
