"""Tests for the calibrate CLI command."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from ash_hawk.cli.calibrate import _compute_recommended_threshold
from ash_hawk.cli.main import cli
from ash_hawk.storage import FileStorage
from ash_hawk.types import (
    CalibrationSample,
    EvalOutcome,
    EvalRunSummary,
    EvalStatus,
    EvalSuite,
    EvalTask,
    EvalTrial,
    GraderResult,
    RunEnvelope,
    SuiteMetrics,
    TokenUsage,
    ToolSurfacePolicy,
    TrialEnvelope,
    TrialResult,
)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def ground_truth_file(temp_dir):
    """Legacy format ground truth file for backward compatibility tests."""
    gt_path = Path(temp_dir) / "ground_truth.json"
    gt_data = {
        "samples": [
            {"task_id": "task-001", "actual_passed": True},
            {"task_id": "task-002", "actual_passed": False},
            {"task_id": "task-003", "actual_passed": True},
        ]
    }
    with open(gt_path, "w") as f:
        json.dump(gt_data, f)
    return str(gt_path)


@pytest.fixture
def simple_ground_truth_file(temp_dir):
    """Simple format ground truth file: {trial_id: bool}."""
    gt_path = Path(temp_dir) / "simple_ground_truth.json"
    gt_data = {
        "trial-001": True,
        "trial-002": False,
        "trial-003": True,
    }
    with open(gt_path, "w") as f:
        json.dump(gt_data, f)
    return str(gt_path)


@pytest.fixture
def mock_storage_with_run(temp_dir):
    storage_path = Path(temp_dir) / ".ash-hawk"
    storage = FileStorage(storage_path)

    suite = EvalSuite(
        id="test-suite",
        name="Test Suite",
        tasks=[
            EvalTask(id="task-001", input="Test 1"),
            EvalTask(id="task-002", input="Test 2"),
            EvalTask(id="task-003", input="Test 3"),
        ],
    )

    envelope = RunEnvelope(
        run_id="test-run",
        suite_id="test-suite",
        suite_hash="abc123",
        harness_version="0.1.0",
        agent_name="test-agent",
        provider="test",
        model="test-model",
        tool_policy_hash="policy-hash",
        python_version="3.11.0",
        os_info="test",
        created_at="2024-01-01T00:00:00Z",
    )

    policy = ToolSurfacePolicy()
    trial_data = [
        ("trial-001", 0.9, True),
        ("trial-002", 0.3, False),
        ("trial-003", 0.7, True),
        ("trial-004", 0.8, True),
    ]
    trials = []
    for i, (trial_id, score, passed) in enumerate(trial_data):
        trial = EvalTrial(
            id=trial_id,
            task_id=f"task-00{i + 1}",
            status=EvalStatus.COMPLETED,
            result=TrialResult(
                trial_id=trial_id,
                outcome=EvalOutcome.success(),
                grader_results=[
                    GraderResult(
                        grader_type="llm_judge",
                        grader_id=f"judge-{trial_id}",
                        score=score,
                        passed=passed,
                    )
                ],
                aggregate_score=score,
                aggregate_passed=passed,
            ),
        )
        trial_envelope = TrialEnvelope(
            trial_id=trial_id,
            run_id="test-run",
            task_id=f"task-00{i + 1}",
            policy_snapshot=policy,
            created_at="2024-01-01T00:00:00Z",
        )
        trials.append(trial)

        async def save_trial_fn(t=trial, te=trial_envelope, p=policy, s=storage):
            await s.save_trial("test-suite", "test-run", t, te, p)

        asyncio.run(save_trial_fn())

    summary = EvalRunSummary(
        envelope=envelope,
        metrics=SuiteMetrics(
            suite_id="test-suite",
            run_id="test-run",
            total_tasks=4,
            completed_tasks=4,
            passed_tasks=3,
            pass_rate=0.75,
            mean_score=0.675,
            total_tokens=TokenUsage(),
            created_at="2024-01-01T00:01:00Z",
        ),
        trials=trials,
    )

    async def setup_storage():
        await storage.save_suite(suite)
        await storage.save_run_envelope("test-suite", envelope)
        await storage.save_summary("test-suite", "test-run", summary)

    asyncio.run(setup_storage())

    return str(storage_path), "test-suite", "test-run"


class TestCalibrateHelp:
    def test_calibrate_command_registered(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "calibrate" in result.output

    def test_calibrate_help_shows_options(self, runner):
        result = runner.invoke(cli, ["calibrate", "--help"])
        assert result.exit_code == 0
        assert "--ground-truth" in result.output
        assert "--run" in result.output
        assert "--output" in result.output
        assert "--grader" in result.output


class TestCalibrateRequiredArgs:
    def test_calibrate_requires_ground_truth(self, runner):
        result = runner.invoke(cli, ["calibrate", "--run", "run-123"])
        assert result.exit_code != 0
        assert "ground-truth" in result.output.lower() or "required" in result.output.lower()

    def test_calibrate_requires_run(self, runner, temp_dir):
        gt_path = Path(temp_dir) / "gt.json"
        gt_path.write_text('{"samples": []}')
        result = runner.invoke(cli, ["calibrate", "--ground-truth", str(gt_path)])
        assert result.exit_code != 0
        assert "run" in result.output.lower() or "required" in result.output.lower()

    def test_calibrate_validates_ground_truth_exists(self, runner):
        result = runner.invoke(
            cli,
            ["calibrate", "--ground-truth", "/nonexistent/gt.json", "--run", "run-123"],
        )
        assert result.exit_code != 0


class TestCalibrateOutput:
    def test_calibrate_outputs_json_with_calibration_curve(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "result.json"
        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        assert output_path.exists()
        with open(output_path) as f:
            output = json.load(f)
        assert "metrics" in output

    def test_calibrate_outputs_ece_in_metrics(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "result.json"
        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        with open(output_path) as f:
            output = json.load(f)
        assert "metrics" in output
        assert "ece" in output["metrics"]

    def test_calibrate_outputs_brier_score_in_metrics(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "result.json"
        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        with open(output_path) as f:
            output = json.load(f)
        assert "metrics" in output
        assert "brier_score" in output["metrics"]

    def test_calibrate_outputs_grader_name(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "result.json"
        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        with open(output_path) as f:
            output = json.load(f)
        assert "grader_name" in output

    def test_calibrate_default_grader_is_llm_judge(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "result.json"
        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        with open(output_path) as f:
            output = json.load(f)
        assert output.get("grader_name") == "llm_judge"

    def test_calibrate_custom_grader(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "result.json"
        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--grader",
                "custom_grader",
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        with open(output_path) as f:
            output = json.load(f)
        assert output.get("grader_name") == "custom_grader"

    def test_calibrate_writes_to_output_file(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "calibration_result.json"
        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        assert output_path.exists()
        with open(output_path) as f:
            output = json.load(f)
        assert "metrics" in output
        assert "ece" in output["metrics"]
        assert "brier_score" in output["metrics"]


class TestCalibrateCompute:
    def test_calibrate_computes_ece_from_matched_trials(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "calibration.json"

        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        with open(output_path) as f:
            output = json.load(f)

        assert "metrics" in output
        assert "ece" in output["metrics"]
        assert isinstance(output["metrics"]["ece"], (int, float))
        assert output["metrics"]["ece"] >= 0.0

    def test_calibrate_computes_brier_score_from_matched_trials(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "calibration.json"

        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        with open(output_path) as f:
            output = json.load(f)

        assert "metrics" in output
        assert "brier_score" in output["metrics"]
        assert isinstance(output["metrics"]["brier_score"], (int, float))
        assert 0.0 <= output["metrics"]["brier_score"] <= 1.0

    def test_calibrate_outputs_samples_matched_count(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "calibration.json"

        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        with open(output_path) as f:
            output = json.load(f)

        assert "samples_matched" in output
        assert output["samples_matched"] == 3

    def test_calibrate_outputs_samples_total_count(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "calibration.json"

        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        with open(output_path) as f:
            output = json.load(f)

        assert "samples_total" in output
        assert output["samples_total"] == 3

    def test_calibrate_outputs_run_id(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "calibration.json"

        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        with open(output_path) as f:
            output = json.load(f)

        assert "run_id" in output
        assert output["run_id"] == run_id

    def test_calibrate_outputs_ground_truth_file_path(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "calibration.json"

        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        with open(output_path) as f:
            output = json.load(f)

        assert "ground_truth_file" in output

    def test_calibrate_outputs_recommended_threshold_computed(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        """Test that recommended_threshold is computed from sample analysis."""
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "calibration.json"

        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        with open(output_path) as f:
            output = json.load(f)

        assert "recommended_threshold" in output
        # Threshold should be in valid range [0.1, 0.9]
        assert 0.1 <= output["recommended_threshold"] <= 0.9

    def test_calibrate_outputs_rationale_for_threshold(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        """Test that calibration output includes rationale for threshold choice."""
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "calibration.json"

        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        with open(output_path) as f:
            output = json.load(f)

        assert "rationale" in output
        assert isinstance(output["rationale"], str)
        assert len(output["rationale"]) > 0

    def test_calibrate_handles_partial_ground_truth_gracefully(
        self, runner, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run

        gt_path = Path(temp_dir) / "partial_gt.json"
        gt_data = {"trial-001": True}
        with open(gt_path, "w") as f:
            json.dump(gt_data, f)

        output_path = Path(temp_dir) / "calibration.json"

        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                str(gt_path),
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        with open(output_path) as f:
            output = json.load(f)

        assert output["samples_matched"] == 1
        assert output["samples_total"] == 1

    def test_calibrate_warns_on_unlabeled_trials(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "calibration.json"

        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        assert "trial-004" in result.output or "1 trial" in result.output.lower()

    def test_calibrate_uses_specified_grader(
        self, runner, simple_ground_truth_file, mock_storage_with_run, temp_dir
    ):
        storage_path, suite_id, run_id = mock_storage_with_run
        output_path = Path(temp_dir) / "calibration.json"

        result = runner.invoke(
            cli,
            [
                "calibrate",
                "--ground-truth",
                simple_ground_truth_file,
                "--run",
                run_id,
                "--storage",
                storage_path,
                "--suite",
                suite_id,
                "--grader",
                "llm_judge",
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        with open(output_path) as f:
            output = json.load(f)

        assert output["grader_name"] == "llm_judge"


class TestComputeRecommendedThreshold:
    def test_returns_threshold_that_maximizes_accuracy(self):
        samples = [
            CalibrationSample(predicted=0.9, actual=True),
            CalibrationSample(predicted=0.8, actual=True),
            CalibrationSample(predicted=0.2, actual=False),
            CalibrationSample(predicted=0.3, actual=False),
        ]
        threshold, rationale = _compute_recommended_threshold(samples)
        assert 0.1 <= threshold <= 0.9
        assert "accuracy" in rationale.lower()

    def test_empty_samples_returns_default(self):
        samples: list[CalibrationSample] = []
        threshold, rationale = _compute_recommended_threshold(samples)
        assert threshold == 0.5
        assert "no samples" in rationale.lower()

    def test_uniform_predictions_returns_default(self):
        samples = [
            CalibrationSample(predicted=0.5, actual=True),
            CalibrationSample(predicted=0.5, actual=False),
            CalibrationSample(predicted=0.5, actual=True),
        ]
        threshold, rationale = _compute_recommended_threshold(samples)
        assert threshold == 0.5
        assert "uniform" in rationale.lower()

    def test_threshold_in_valid_range(self):
        samples = [
            CalibrationSample(predicted=0.95, actual=True),
            CalibrationSample(predicted=0.05, actual=False),
        ]
        threshold, _ = _compute_recommended_threshold(samples)
        assert 0.1 <= threshold <= 0.9

    def test_rationale_includes_accuracy_comparison(self):
        samples = [
            CalibrationSample(predicted=0.9, actual=True),
            CalibrationSample(predicted=0.1, actual=False),
            CalibrationSample(predicted=0.85, actual=True),
        ]
        threshold, rationale = _compute_recommended_threshold(samples)
        assert "%" in rationale
        assert "0.5" in rationale
