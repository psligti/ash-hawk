"""Human grader interfaces for manual review.

This module provides graders that export trials for external human review
and import review results back into the evaluation pipeline.

Key components:
- HumanGrader: Abstract base for human-in-the-loop grading
- ManualReviewGrader: Exports trials for external review tools
- ReviewExporter: Handles export to various formats (JSON, CSV)
- ReviewImporter: Handles import of review results
"""

from __future__ import annotations

import csv
import json
from abc import abstractmethod
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pydantic as pd

from ash_hawk.graders.base import Grader
from ash_hawk.types import EvalTranscript, EvalTrial, GraderResult, GraderSpec

# =============================================================================
# REVIEW DATA MODELS
# =============================================================================


class ReviewItem(pd.BaseModel):
    """Single item for human review.

    Contains all the information a human reviewer needs to evaluate a trial.
    """

    trial_id: str = pd.Field(description="Unique identifier for the trial")
    task_id: str = pd.Field(description="ID of the task being evaluated")
    input: str | dict[str, Any] | None = pd.Field(default=None, description="Task input")
    expected_output: str | dict[str, Any] | None = pd.Field(
        default=None, description="Expected output (if available)"
    )
    agent_response: str | dict[str, Any] | None = pd.Field(
        default=None, description="Agent's response/output"
    )
    transcript_summary: dict[str, Any] = pd.Field(
        default_factory=dict, description="Summary of the transcript"
    )
    metadata: dict[str, Any] = pd.Field(default_factory=dict, description="Additional metadata")

    model_config = pd.ConfigDict(extra="forbid")


class ReviewDecision(pd.BaseModel):
    """Human reviewer's decision for a single trial.

    Captures the pass/fail decision, score, and rationale.
    """

    trial_id: str = pd.Field(description="ID of the trial being reviewed")
    reviewer_id: str | None = pd.Field(default=None, description="Identifier for the reviewer")
    passed: bool = pd.Field(description="Whether the trial passed review")
    score: float = pd.Field(ge=0.0, le=1.0, description="Score from 0.0 to 1.0")
    rationale: str | None = pd.Field(default=None, description="Explanation for the decision")
    labels: list[str] = pd.Field(default_factory=list, description="Categorical labels applied")
    issues: list[str] = pd.Field(default_factory=list, description="Identified issues/problems")
    reviewed_at: str = pd.Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO timestamp when review was completed",
    )

    model_config = pd.ConfigDict(extra="forbid")


class ReviewBatch(pd.BaseModel):
    """Collection of review items for export/import.

    Supports batch processing of human reviews.
    """

    batch_id: str = pd.Field(
        default_factory=lambda: f"batch_{datetime.now(UTC).timestamp()}",
        description="Unique identifier for this batch",
    )
    items: list[ReviewItem] = pd.Field(default_factory=list, description="Items to review")
    decisions: list[ReviewDecision] = pd.Field(
        default_factory=list, description="Completed review decisions"
    )
    created_at: str = pd.Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO timestamp when batch was created",
    )
    metadata: dict[str, Any] = pd.Field(default_factory=dict, description="Batch-level metadata")

    model_config = pd.ConfigDict(extra="forbid")


# =============================================================================
# INTER-ANNOTATOR AGREEMENT
# =============================================================================


class AgreementMetrics(pd.BaseModel):
    """Inter-annotator agreement metrics.

    Measures consistency between multiple human reviewers.
    """

    trial_id: str = pd.Field(description="ID of the trial")
    num_reviewers: int = pd.Field(description="Number of reviewers")
    agreement_score: float = pd.Field(
        ge=0.0, le=1.0, description="Agreement score (0=none, 1=perfect)"
    )
    cohen_kappa: float | None = pd.Field(
        default=None, ge=-1.0, le=1.0, description="Cohen's kappa (binary)"
    )
    fleiss_kappa: float | None = pd.Field(
        default=None, ge=-1.0, le=1.0, description="Fleiss' kappa (multi-rater)"
    )
    percent_agreement: float = pd.Field(ge=0.0, le=1.0, description="Raw percent agreement")
    decisions: list[ReviewDecision] = pd.Field(
        default_factory=list, description="Individual reviewer decisions"
    )

    model_config = pd.ConfigDict(extra="forbid")


def calculate_percent_agreement(decisions: list[ReviewDecision]) -> float:
    """Calculate simple percent agreement between reviewers.

    Args:
        decisions: List of review decisions from different reviewers.

    Returns:
        Proportion of reviewer pairs that agree on pass/fail.
    """
    if len(decisions) < 2:
        return 1.0

    passed_count = sum(1 for d in decisions if d.passed)
    total = len(decisions)

    # Calculate agreement as the proportion in the majority
    majority = max(passed_count, total - passed_count)
    return majority / total


def calculate_cohen_kappa(decisions: list[ReviewDecision]) -> float | None:
    """Calculate Cohen's kappa for two reviewers.

    Args:
        decisions: List of exactly 2 review decisions.

    Returns:
        Cohen's kappa score, or None if calculation not possible.
    """
    if len(decisions) != 2:
        return None

    d1, d2 = decisions[0], decisions[1]

    # Binary classification
    a = 1 if d1.passed else 0
    b = 1 if d2.passed else 0

    # Simple case: both agree
    if a == b:
        return 1.0

    # Observed agreement
    # For binary single-item case, agreement is 0
    # Cohen's kappa requires more data points
    return 0.0


def calculate_agreement_metrics(trial_id: str, decisions: list[ReviewDecision]) -> AgreementMetrics:
    """Calculate all agreement metrics for a set of reviewer decisions.

    Args:
        trial_id: ID of the trial being analyzed.
        decisions: List of review decisions from different reviewers.

    Returns:
        AgreementMetrics with calculated scores.
    """
    percent = calculate_percent_agreement(decisions)
    kappa = calculate_cohen_kappa(decisions) if len(decisions) == 2 else None

    return AgreementMetrics(
        trial_id=trial_id,
        num_reviewers=len(decisions),
        agreement_score=percent,
        cohen_kappa=kappa,
        percent_agreement=percent,
        decisions=decisions,
    )


# =============================================================================
# ABSTRACT HUMAN GRADER
# =============================================================================


class HumanGrader(Grader):
    """Abstract base class for human-in-the-loop grading.

    Human graders export trials for external review and import
    results back into the evaluation pipeline. They do not
    perform grading directly but facilitate human review workflows.
    """

    @abstractmethod
    def export_for_review(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
    ) -> ReviewItem:
        """Export a trial for human review.

        Args:
            trial: The trial to export.
            transcript: The trial's execution transcript.

        Returns:
            ReviewItem ready for human review.
        """
        ...

    @abstractmethod
    def import_review(
        self,
        decision: ReviewDecision,
    ) -> GraderResult:
        """Import a human review decision as a GraderResult.

        Args:
            decision: The human reviewer's decision.

        Returns:
            GraderResult suitable for the evaluation pipeline.
        """
        ...


# =============================================================================
# MANUAL REVIEW GRADER
# =============================================================================


class ManualReviewGrader(HumanGrader):
    """Grader that exports trials for external manual review.

    This grader does not perform grading directly. Instead, it:
    1. Exports trials to a format suitable for human review
    2. Waits for review results to be imported
    3. Returns the human-provided scores

    The grade() method returns a pending result if no review is available,
    or the actual result if a review has been imported.

    Configuration options (in spec.config):
        export_format: 'json' | 'csv' - Format for export (default: 'json')
        export_path: str - Directory to export review items
        import_path: str - Directory to read review decisions from
        require_rationale: bool - Whether rationale is required (default: False)
    """

    def __init__(
        self,
        export_format: Literal["json", "csv"] = "json",
        export_path: str | Path | None = None,
        import_path: str | Path | None = None,
    ) -> None:
        """Initialize the manual review grader.

        Args:
            export_format: Format for exporting review items.
            export_path: Directory for exported review items.
            import_path: Directory for imported review decisions.
        """
        self._export_format = export_format
        self._export_path = Path(export_path) if export_path else None
        self._import_path = Path(import_path) if import_path else None
        self._pending_reviews: dict[str, ReviewItem] = {}
        self._completed_reviews: dict[str, ReviewDecision] = {}

    @property
    def name(self) -> str:
        """Return the grader name."""
        return "manual_review"

    def export_for_review(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
    ) -> ReviewItem:
        """Export a trial for human review.

        Args:
            trial: The trial to export.
            transcript: The trial's execution transcript.

        Returns:
            ReviewItem ready for human review.
        """
        item = ReviewItem(
            trial_id=trial.id,
            task_id=trial.task_id,
            input=trial.input_snapshot,
            expected_output=None,  # Would need task access to populate
            agent_response=transcript.agent_response,
            transcript_summary={
                "duration_seconds": transcript.duration_seconds,
                "token_usage": transcript.token_usage.model_dump(),
                "tool_calls_count": len(transcript.tool_calls),
                "messages_count": len(transcript.messages),
                "cost_usd": transcript.cost_usd,
            },
            metadata={
                "attempt_number": trial.attempt_number,
                "status": trial.status,
            },
        )

        self._pending_reviews[trial.id] = item
        return item

    def import_review(
        self,
        decision: ReviewDecision,
    ) -> GraderResult:
        """Import a human review decision as a GraderResult.

        Args:
            decision: The human reviewer's decision.

        Returns:
            GraderResult suitable for the evaluation pipeline.
        """
        self._completed_reviews[decision.trial_id] = decision

        # Remove from pending if present
        self._pending_reviews.pop(decision.trial_id, None)

        return GraderResult(
            grader_type=self.name,
            grader_id=decision.reviewer_id,
            score=decision.score,
            passed=decision.passed,
            details={
                "rationale": decision.rationale,
                "labels": decision.labels,
                "issues": decision.issues,
                "reviewed_at": decision.reviewed_at,
            },
        )

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Grade a trial by exporting for review or returning imported result.

        If a review decision exists for this trial, returns that result.
        Otherwise, exports the trial for review and returns a pending result.

        Args:
            trial: The trial being evaluated.
            transcript: The execution transcript.
            spec: The grader specification.

        Returns:
            GraderResult with human review score or pending status.
        """
        # Check if we have a completed review
        if trial.id in self._completed_reviews:
            return self.import_review(self._completed_reviews[trial.id])

        # Export for review
        item = self.export_for_review(trial, transcript)

        # Export to file if paths configured
        config = spec.config
        export_format = config.get("export_format", self._export_format)
        export_path = config.get("export_path", self._export_path)

        if export_path:
            self._export_item(item, Path(export_path), export_format)

        # Return pending result
        return GraderResult(
            grader_type=self.name,
            score=0.0,
            passed=False,
            details={
                "status": "pending_review",
                "trial_id": trial.id,
                "message": "Trial exported for human review",
            },
        )

    def _export_item(self, item: ReviewItem, path: Path, format: Literal["json", "csv"]) -> None:
        """Export a review item to file.

        Args:
            item: The item to export.
            path: Directory to export to.
            format: Export format (json or csv).
        """
        path.mkdir(parents=True, exist_ok=True)

        if format == "json":
            file_path = path / f"{item.trial_id}.json"
            with open(file_path, "w") as f:
                json.dump(item.model_dump(), f, indent=2, default=str)
        elif format == "csv":
            file_path = path / "review_items.csv"
            write_header = not file_path.exists()
            with open(file_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["trial_id", "task_id", "input", "agent_response"])
                writer.writerow(
                    [
                        item.trial_id,
                        item.task_id,
                        json.dumps(item.input),
                        json.dumps(item.agent_response),
                    ]
                )

    def load_decisions_from_path(self, path: Path) -> int:
        """Load review decisions from a directory.

        Args:
            path: Directory containing decision files.

        Returns:
            Number of decisions loaded.
        """
        count = 0
        for file_path in path.glob("*.json"):
            with open(file_path) as f:
                data = json.load(f)
                # Handle single decision or batch
                if "decisions" in data:
                    for d in data["decisions"]:
                        decision = ReviewDecision.model_validate(d)
                        self._completed_reviews[decision.trial_id] = decision
                        count += 1
                elif "trial_id" in data:
                    decision = ReviewDecision.model_validate(data)
                    self._completed_reviews[decision.trial_id] = decision
                    count += 1
        return count


# =============================================================================
# EXPORT/IMPORT UTILITIES
# =============================================================================


class ReviewExporter:
    """Utility class for exporting trials for external review.

    Supports multiple export formats for integration with
    external review tools and platforms.
    """

    @staticmethod
    def export_batch(
        items: list[ReviewItem],
        output_path: Path,
        format: Literal["json", "csv"] = "json",
    ) -> Path:
        """Export a batch of review items.

        Args:
            items: Items to export.
            output_path: Output file path.
            format: Export format.

        Returns:
            Path to the created export file.
        """
        batch = ReviewBatch(items=items)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(batch.model_dump(), f, indent=2, default=str)
        elif format == "csv":
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "batch_id",
                        "trial_id",
                        "task_id",
                        "input",
                        "agent_response",
                        "created_at",
                    ]
                )
                for item in items:
                    writer.writerow(
                        [
                            batch.batch_id,
                            item.trial_id,
                            item.task_id,
                            json.dumps(item.input),
                            json.dumps(item.agent_response),
                            batch.created_at,
                        ]
                    )

        return output_path

    @staticmethod
    def export_for_label_studio(
        items: list[ReviewItem],
        output_path: Path,
    ) -> Path:
        """Export in Label Studio compatible format.

        Args:
            items: Items to export.
            output_path: Output file path.

        Returns:
            Path to the created export file.
        """
        label_studio_data = []
        for item in items:
            label_studio_data.append(
                {
                    "id": item.trial_id,
                    "data": {
                        "text": item.agent_response,
                        "input": item.input,
                        "task_id": item.task_id,
                    },
                }
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(label_studio_data, f, indent=2)

        return output_path


class ReviewImporter:
    """Utility class for importing review results from external tools.

    Supports multiple import formats for integration with
    external review tools and platforms.
    """

    @staticmethod
    def import_from_json(file_path: Path) -> list[ReviewDecision]:
        """Import review decisions from JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            List of imported review decisions.
        """
        with open(file_path) as f:
            data = json.load(f)

        decisions = []
        # Handle batch format
        if "decisions" in data:
            for d in data["decisions"]:
                decisions.append(ReviewDecision.model_validate(d))
        # Handle single decision
        elif "trial_id" in data:
            decisions.append(ReviewDecision.model_validate(data))
        # Handle list format
        elif isinstance(data, list):
            for d in data:
                decisions.append(ReviewDecision.model_validate(d))

        return decisions

    @staticmethod
    def import_from_csv(file_path: Path) -> list[ReviewDecision]:
        """Import review decisions from CSV file.

        Expected columns: trial_id, passed, score, rationale, reviewer_id

        Args:
            file_path: Path to the CSV file.

        Returns:
            List of imported review decisions.
        """
        decisions = []
        with open(file_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                decision = ReviewDecision(
                    trial_id=row["trial_id"],
                    passed=row["passed"].lower() in ("true", "1", "yes"),
                    score=float(row["score"]),
                    rationale=row.get("rationale"),
                    reviewer_id=row.get("reviewer_id"),
                )
                decisions.append(decision)

        return decisions

    @staticmethod
    def import_from_label_studio(file_path: Path) -> list[ReviewDecision]:
        """Import from Label Studio export format.

        Args:
            file_path: Path to the Label Studio export file.

        Returns:
            List of imported review decisions.
        """
        with open(file_path) as f:
            data = json.load(f)

        decisions = []
        for item in data:
            # Label Studio format has annotations
            annotations = item.get("annotations", [])
            for annotation in annotations:
                results = annotation.get("result", [])
                for result in results:
                    value = result.get("value", {})
                    decision = ReviewDecision(
                        trial_id=str(item["id"]),
                        passed=value.get("choices", ["reject"])[0] == "accept",
                        score=1.0 if value.get("choices", ["reject"])[0] == "accept" else 0.0,
                        rationale=value.get("text"),
                        reviewer_id=annotation.get("completed_by"),
                    )
                    decisions.append(decision)

        return decisions


__all__ = [
    # Base classes
    "HumanGrader",
    "ManualReviewGrader",
    # Data models
    "ReviewItem",
    "ReviewDecision",
    "ReviewBatch",
    "AgreementMetrics",
    # Utilities
    "ReviewExporter",
    "ReviewImporter",
    # Agreement functions
    "calculate_percent_agreement",
    "calculate_cohen_kappa",
    "calculate_agreement_metrics",
]
