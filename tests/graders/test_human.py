"""Tests for human grader interfaces and review workflows."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ash_hawk.graders.base import Grader
from ash_hawk.graders.human import (
    ManualReviewGrader,
    ReviewBatch,
    ReviewDecision,
    ReviewExporter,
    ReviewImporter,
    ReviewItem,
    calculate_agreement_metrics,
    calculate_cohen_kappa,
    calculate_percent_agreement,
)
from ash_hawk.review import (
    LoggingReviewHook,
    NullReviewHook,
    ReviewStatus,
    ReviewWorkflow,
    create_simple_workflow,
)
from ash_hawk.types import (
    EvalStatus,
    EvalTranscript,
    EvalTrial,
    GraderSpec,
    TokenUsage,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_transcript() -> EvalTranscript:
    return EvalTranscript(
        messages=[{"role": "user", "content": "Hello"}],
        tool_calls=[],
        token_usage=TokenUsage(input=10, output=20),
        cost_usd=0.001,
        duration_seconds=1.5,
        agent_response="Hello! How can I help?",
    )


@pytest.fixture
def sample_trial() -> EvalTrial:
    return EvalTrial(
        id="trial_001",
        task_id="task_001",
        status=EvalStatus.COMPLETED,
        input_snapshot="What is 2+2?",
    )


@pytest.fixture
def sample_grader_spec() -> GraderSpec:
    return GraderSpec(
        grader_type="manual_review",
        config={},
    )


# =============================================================================
# REVIEW ITEM TESTS
# =============================================================================


def test_review_item_creation() -> None:
    item = ReviewItem(
        trial_id="trial_001",
        task_id="task_001",
        input="Test input",
        agent_response="Test response",
    )
    assert item.trial_id == "trial_001"
    assert item.task_id == "task_001"
    assert item.input == "Test input"
    assert item.agent_response == "Test response"


def test_review_item_serialization() -> None:
    item = ReviewItem(
        trial_id="trial_001",
        task_id="task_001",
        input={"prompt": "test"},
        agent_response={"answer": "result"},
    )
    data = item.model_dump()
    assert data["trial_id"] == "trial_001"
    assert isinstance(data["input"], dict)


# =============================================================================
# REVIEW DECISION TESTS
# =============================================================================


def test_review_decision_creation() -> None:
    decision = ReviewDecision(
        trial_id="trial_001",
        passed=True,
        score=0.95,
        rationale="Excellent response",
    )
    assert decision.trial_id == "trial_001"
    assert decision.passed is True
    assert decision.score == 0.95
    assert decision.rationale == "Excellent response"
    assert decision.reviewed_at is not None


def test_review_decision_with_labels() -> None:
    decision = ReviewDecision(
        trial_id="trial_001",
        passed=False,
        score=0.3,
        labels=["incomplete", "needs_clarification"],
        issues=["Did not address the question"],
    )
    assert decision.labels == ["incomplete", "needs_clarification"]
    assert decision.issues == ["Did not address the question"]


# =============================================================================
# MANUAL REVIEW GRADER TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_manual_review_grader_name() -> None:
    grader = ManualReviewGrader()
    assert grader.name == "manual_review"


@pytest.mark.asyncio
async def test_manual_review_grader_export(
    sample_trial: EvalTrial, sample_transcript: EvalTranscript
) -> None:
    grader = ManualReviewGrader()
    item = grader.export_for_review(sample_trial, sample_transcript)

    assert item.trial_id == sample_trial.id
    assert item.task_id == sample_trial.task_id
    assert item.agent_response == sample_transcript.agent_response


@pytest.mark.asyncio
async def test_manual_review_grader_import_decision() -> None:
    grader = ManualReviewGrader()
    decision = ReviewDecision(
        trial_id="trial_001",
        passed=True,
        score=0.9,
        rationale="Good work",
    )
    result = grader.import_review(decision)

    assert result.grader_type == "manual_review"
    assert result.passed is True
    assert result.score == 0.9
    assert result.details["rationale"] == "Good work"


@pytest.mark.asyncio
async def test_manual_review_grader_returns_pending_without_review(
    sample_trial: EvalTrial,
    sample_transcript: EvalTranscript,
    sample_grader_spec: GraderSpec,
) -> None:
    grader = ManualReviewGrader()
    result = await grader.grade(sample_trial, sample_transcript, sample_grader_spec)

    assert result.passed is False
    assert result.score == 0.0
    assert result.details["status"] == "pending_review"


@pytest.mark.asyncio
async def test_manual_review_grader_returns_imported_result(
    sample_trial: EvalTrial,
    sample_transcript: EvalTranscript,
    sample_grader_spec: GraderSpec,
) -> None:
    grader = ManualReviewGrader()

    decision = ReviewDecision(
        trial_id=sample_trial.id,
        passed=True,
        score=0.85,
        reviewer_id="reviewer_001",
    )
    grader.import_review(decision)

    result = await grader.grade(sample_trial, sample_transcript, sample_grader_spec)

    assert result.passed is True
    assert result.score == 0.85
    assert result.grader_id == "reviewer_001"


# =============================================================================
# INTER-ANNOTATOR AGREEMENT TESTS
# =============================================================================


def test_calculate_percent_agreement_unanimous() -> None:
    decisions = [
        ReviewDecision(trial_id="t1", passed=True, score=1.0),
        ReviewDecision(trial_id="t1", passed=True, score=0.9),
        ReviewDecision(trial_id="t1", passed=True, score=0.95),
    ]
    agreement = calculate_percent_agreement(decisions)
    assert agreement == 1.0


def test_calculate_percent_agreement_split() -> None:
    decisions = [
        ReviewDecision(trial_id="t1", passed=True, score=1.0),
        ReviewDecision(trial_id="t1", passed=False, score=0.0),
    ]
    agreement = calculate_percent_agreement(decisions)
    assert agreement == 0.5


def test_calculate_percent_agreement_single() -> None:
    decisions = [ReviewDecision(trial_id="t1", passed=True, score=1.0)]
    agreement = calculate_percent_agreement(decisions)
    assert agreement == 1.0


def test_calculate_cohen_kappa_perfect_agreement() -> None:
    decisions = [
        ReviewDecision(trial_id="t1", passed=True, score=1.0),
        ReviewDecision(trial_id="t1", passed=True, score=0.9),
    ]
    kappa = calculate_cohen_kappa(decisions)
    assert kappa == 1.0


def test_calculate_cohen_kappa_disagreement() -> None:
    decisions = [
        ReviewDecision(trial_id="t1", passed=True, score=1.0),
        ReviewDecision(trial_id="t1", passed=False, score=0.0),
    ]
    kappa = calculate_cohen_kappa(decisions)
    assert kappa == 0.0


def test_calculate_cohen_kappa_wrong_count() -> None:
    decisions = [ReviewDecision(trial_id="t1", passed=True, score=1.0)]
    kappa = calculate_cohen_kappa(decisions)
    assert kappa is None


def test_calculate_agreement_metrics() -> None:
    decisions = [
        ReviewDecision(trial_id="t1", passed=True, score=1.0, reviewer_id="r1"),
        ReviewDecision(trial_id="t1", passed=True, score=0.9, reviewer_id="r2"),
    ]
    metrics = calculate_agreement_metrics("t1", decisions)

    assert metrics.trial_id == "t1"
    assert metrics.num_reviewers == 2
    assert metrics.agreement_score == 1.0
    assert metrics.cohen_kappa == 1.0


# =============================================================================
# REVIEW EXPORTER/IMPORTER TESTS
# =============================================================================


def test_export_batch_json() -> None:
    items = [
        ReviewItem(trial_id="t1", task_id="task1", input="q1", agent_response="a1"),
        ReviewItem(trial_id="t2", task_id="task1", input="q2", agent_response="a2"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "batch.json"
        result_path = ReviewExporter.export_batch(items, output_path, format="json")

        assert result_path.exists()
        with open(result_path) as f:
            data = json.load(f)
        assert "batch_id" in data
        assert len(data["items"]) == 2


def test_export_batch_csv() -> None:
    items = [
        ReviewItem(trial_id="t1", task_id="task1", input="q1", agent_response="a1"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "batch.csv"
        result_path = ReviewExporter.export_batch(items, output_path, format="csv")

        assert result_path.exists()
        with open(result_path) as f:
            content = f.read()
        assert "trial_id" in content
        assert "t1" in content


def test_import_from_json_single() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "decision.json"
        decision_data = {
            "trial_id": "t1",
            "passed": True,
            "score": 0.9,
            "rationale": "Good",
        }
        with open(file_path, "w") as f:
            json.dump(decision_data, f)

        decisions = ReviewImporter.import_from_json(file_path)
        assert len(decisions) == 1
        assert decisions[0].trial_id == "t1"
        assert decisions[0].passed is True


def test_import_from_json_list() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "decisions.json"
        decisions_data = [
            {"trial_id": "t1", "passed": True, "score": 0.9},
            {"trial_id": "t2", "passed": False, "score": 0.3},
        ]
        with open(file_path, "w") as f:
            json.dump(decisions_data, f)

        decisions = ReviewImporter.import_from_json(file_path)
        assert len(decisions) == 2


def test_import_from_csv() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "decisions.csv"
        csv_content = "trial_id,passed,score,rationale,reviewer_id\nt1,true,0.9,Good,r1\nt2,false,0.3,Bad,r1\n"
        with open(file_path, "w") as f:
            f.write(csv_content)

        decisions = ReviewImporter.import_from_csv(file_path)
        assert len(decisions) == 2
        assert decisions[0].passed is True
        assert decisions[1].passed is False


# =============================================================================
# REVIEW WORKFLOW TESTS
# =============================================================================


def test_review_workflow_create_batch() -> None:
    workflow = ReviewWorkflow()
    items = [
        ReviewItem(trial_id="t1", task_id="task1", input="q1", agent_response="a1"),
    ]
    batch = workflow.create_batch(items)

    assert batch.batch_id is not None
    assert len(batch.items) == 1
    assert workflow.pending_count == 1


def test_review_workflow_export_batch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        workflow = ReviewWorkflow(export_path=Path(tmpdir))
        items = [
            ReviewItem(trial_id="t1", task_id="task1", input="q1", agent_response="a1"),
        ]
        batch = workflow.create_batch(items)
        result_path = workflow.export_batch(batch)

        assert result_path.exists()


def test_review_workflow_import_decisions() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "export"
        import_path = Path(tmpdir) / "import"
        export_path.mkdir()
        import_path.mkdir()

        workflow = ReviewWorkflow(export_path=export_path, import_path=import_path)
        items = [
            ReviewItem(trial_id="t1", task_id="task1", input="q1", agent_response="a1"),
        ]
        workflow.create_batch(items)

        decision_file = import_path / "decision.json"
        with open(decision_file, "w") as f:
            json.dump({"trial_id": "t1", "passed": True, "score": 0.9}, f)

        decisions = workflow.import_decisions()

        assert len(decisions) == 1
        assert workflow.completed_count == 1
        assert workflow.pending_count == 0


def test_review_workflow_status() -> None:
    workflow = ReviewWorkflow()
    items = [
        ReviewItem(trial_id="t1", task_id="task1", input="q1", agent_response="a1"),
        ReviewItem(trial_id="t2", task_id="task1", input="q2", agent_response="a2"),
    ]
    batch = workflow.create_batch(items)

    assert workflow.status(batch.batch_id) == ReviewStatus.PENDING

    workflow._completed_decisions["t1"] = ReviewDecision(trial_id="t1", passed=True, score=0.9)
    assert workflow.status(batch.batch_id) == ReviewStatus.IN_PROGRESS

    workflow._completed_decisions["t2"] = ReviewDecision(trial_id="t2", passed=True, score=0.8)
    assert workflow.status(batch.batch_id) == ReviewStatus.COMPLETED


def test_review_workflow_get_decision() -> None:
    workflow = ReviewWorkflow()
    decision = ReviewDecision(trial_id="t1", passed=True, score=0.9)
    workflow._completed_decisions["t1"] = decision

    result = workflow.get_decision("t1")
    assert result is not None
    assert result.passed is True

    assert workflow.get_decision("nonexistent") is None


# =============================================================================
# HOOK TESTS
# =============================================================================


def test_null_review_hook() -> None:
    hook = NullReviewHook()
    batch = ReviewBatch(batch_id="batch_001", items=[])
    decisions = []

    hook.on_export(batch)
    hook.on_import(decisions)
    hook.on_complete(batch, decisions)


def test_logging_review_hook() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "log.jsonl"
        hook = LoggingReviewHook(log_path=log_path)

        batch = ReviewBatch(batch_id="batch_001", items=[])
        decisions = [ReviewDecision(trial_id="t1", passed=True, score=0.9)]

        hook.on_export(batch)
        hook.on_import(decisions)
        hook.on_complete(batch, decisions)

        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 3
        event1 = json.loads(lines[0])
        assert event1["event"] == "export"


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


def test_create_simple_workflow() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir) / "export"
        import_dir = Path(tmpdir) / "import"

        workflow = create_simple_workflow(export_dir, import_dir, enable_logging=False)

        assert workflow._export_path is not None
        assert workflow._import_path is not None
        assert isinstance(workflow._hook, NullReviewHook)


def test_create_simple_workflow_with_logging() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir) / "export"
        import_dir = Path(tmpdir) / "import"

        workflow = create_simple_workflow(export_dir, import_dir, enable_logging=True)

        assert isinstance(workflow._hook, LoggingReviewHook)


# =============================================================================
# IMPORT VERIFICATION
# =============================================================================


def test_imports_work() -> None:
    from ash_hawk.graders.human import HumanGrader, ManualReviewGrader
    from ash_hawk.review import ReviewWorkflow as RW, create_simple_workflow as csf

    assert issubclass(ManualReviewGrader, HumanGrader)
    assert issubclass(HumanGrader, Grader)
    assert RW is not None
    assert csf is not None
