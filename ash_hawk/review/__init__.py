"""Review workflow hooks for external human review tools.

This module provides hooks for integrating ash-hawk with external
human review platforms and workflows.

Key components:
- ReviewWorkflow: Manages the review lifecycle
- ReviewHook: Hook points in the evaluation pipeline
- format helpers for various review platforms

NOTE: This is v1 - no UI, only export/import hooks.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

import pydantic as pd

from ash_hawk.graders.human import (
    ReviewBatch,
    ReviewDecision,
    ReviewExporter,
    ReviewImporter,
    ReviewItem,
)


class ReviewStatus(StrEnum):
    """Status of a review workflow."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ReviewHook(ABC):
    """Abstract base for review workflow hooks.

    Hooks allow custom behavior at specific points in the
    review workflow lifecycle.
    """

    @abstractmethod
    def on_export(self, batch: ReviewBatch) -> None:
        """Called when a batch is exported.

        Args:
            batch: The batch being exported.
        """
        ...

    @abstractmethod
    def on_import(self, decisions: list[ReviewDecision]) -> None:
        """Called when decisions are imported.

        Args:
            decisions: The imported decisions.
        """
        ...

    @abstractmethod
    def on_complete(self, batch: ReviewBatch, decisions: list[ReviewDecision]) -> None:
        """Called when a review workflow is complete.

        Args:
            batch: The original batch.
            decisions: All collected decisions.
        """
        ...


class NullReviewHook(ReviewHook):
    """No-op hook for default behavior."""

    def on_export(self, batch: ReviewBatch) -> None:
        pass

    def on_import(self, decisions: list[ReviewDecision]) -> None:
        pass

    def on_complete(self, batch: ReviewBatch, decisions: list[ReviewDecision]) -> None:
        pass


class LoggingReviewHook(ReviewHook):
    """Hook that logs workflow events."""

    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path = log_path
        self._events: list[dict[str, Any]] = []

    def on_export(self, batch: ReviewBatch) -> None:
        self._log("export", batch_id=batch.batch_id, item_count=len(batch.items))

    def on_import(self, decisions: list[ReviewDecision]) -> None:
        self._log("import", decision_count=len(decisions))

    def on_complete(self, batch: ReviewBatch, decisions: list[ReviewDecision]) -> None:
        self._log(
            "complete",
            batch_id=batch.batch_id,
            item_count=len(batch.items),
            decision_count=len(decisions),
        )

    def _log(self, event: str, **kwargs: Any) -> None:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **kwargs,
        }
        self._events.append(entry)
        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


class ReviewWorkflow:
    """Manages the human review workflow.

    Coordinates export of trials for review and import of
    review decisions back into the evaluation pipeline.
    """

    def __init__(
        self,
        export_path: Path | None = None,
        import_path: Path | None = None,
        hook: ReviewHook | None = None,
    ) -> None:
        self._export_path = export_path
        self._import_path = import_path
        self._hook = hook or NullReviewHook()
        self._batches: dict[str, ReviewBatch] = {}
        self._pending_items: dict[str, ReviewItem] = {}
        self._completed_decisions: dict[str, ReviewDecision] = {}

    @property
    def pending_count(self) -> int:
        return len(self._pending_items)

    @property
    def completed_count(self) -> int:
        return len(self._completed_decisions)

    def create_batch(
        self,
        items: list[ReviewItem],
        batch_id: str | None = None,
    ) -> ReviewBatch:
        batch = ReviewBatch(
            batch_id=batch_id or f"batch_{datetime.now(UTC).timestamp()}",
            items=items,
        )
        self._batches[batch.batch_id] = batch
        for item in items:
            self._pending_items[item.trial_id] = item
        return batch

    def export_batch(
        self,
        batch: ReviewBatch,
        format: str = "json",
        filename: str | None = None,
    ) -> Path:
        if not self._export_path:
            raise ValueError("Export path not configured")

        output_path = self._export_path / (filename or f"{batch.batch_id}.{format}")
        ReviewExporter.export_batch(batch.items, output_path, format=format)  # type: ignore[arg-type]
        self._hook.on_export(batch)
        return output_path

    def import_decisions(self, path: Path | None = None) -> list[ReviewDecision]:
        import_path = path or self._import_path
        if not import_path:
            raise ValueError("Import path not configured")

        decisions: list[ReviewDecision] = []
        for file_path in import_path.glob("*.json"):
            file_decisions = ReviewImporter.import_from_json(file_path)
            decisions.extend(file_decisions)

        for decision in decisions:
            self._completed_decisions[decision.trial_id] = decision
            self._pending_items.pop(decision.trial_id, None)

        if decisions:
            self._hook.on_import(decisions)

        return decisions

    def get_decision(self, trial_id: str) -> ReviewDecision | None:
        return self._completed_decisions.get(trial_id)

    def complete_batch(self, batch_id: str) -> None:
        batch = self._batches.get(batch_id)
        if not batch:
            return

        decisions = [
            self._completed_decisions[item.trial_id]
            for item in batch.items
            if item.trial_id in self._completed_decisions
        ]
        self._hook.on_complete(batch, decisions)

    def status(self, batch_id: str) -> ReviewStatus:
        batch = self._batches.get(batch_id)
        if not batch:
            return ReviewStatus.CANCELLED

        reviewed = sum(1 for item in batch.items if item.trial_id in self._completed_decisions)
        total = len(batch.items)

        if reviewed == 0:
            return ReviewStatus.PENDING
        elif reviewed < total:
            return ReviewStatus.IN_PROGRESS
        else:
            return ReviewStatus.COMPLETED


ReviewCallback = Callable[[ReviewBatch], None]


def create_simple_workflow(
    export_dir: str | Path,
    import_dir: str | Path,
    enable_logging: bool = False,
) -> ReviewWorkflow:
    """Factory function to create a configured review workflow.

    Args:
        export_dir: Directory for exported review items.
        import_dir: Directory to read review decisions from.
        enable_logging: Whether to enable event logging.

    Returns:
        Configured ReviewWorkflow instance.
    """
    export_path = Path(export_dir)
    import_path = Path(import_dir)

    export_path.mkdir(parents=True, exist_ok=True)
    import_path.mkdir(parents=True, exist_ok=True)

    hook: ReviewHook
    if enable_logging:
        log_path = export_path / "review_log.jsonl"
        hook = LoggingReviewHook(log_path=log_path)
    else:
        hook = NullReviewHook()

    return ReviewWorkflow(
        export_path=export_path,
        import_path=import_path,
        hook=hook,
    )


__all__ = [
    "ReviewStatus",
    "ReviewHook",
    "NullReviewHook",
    "LoggingReviewHook",
    "ReviewWorkflow",
    "ReviewCallback",
    "create_simple_workflow",
    # Re-export from human module for convenience
    "ReviewItem",
    "ReviewDecision",
    "ReviewBatch",
    "ReviewExporter",
    "ReviewImporter",
]
