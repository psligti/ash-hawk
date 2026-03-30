"""Storage backend interface and implementations for Ash Hawk.

This module provides the storage abstraction layer for persisting evaluation
data including suites, runs, trials, and their associated metadata.

Directory structure:
    .ash-hawk/{suite_id}/suite.json              - Suite definition
    .ash-hawk/{suite_id}/runs/{run_id}/envelope.json - Run envelope
    .ash-hawk/{suite_id}/runs/{run_id}/trials/{trial_id}.json - Trial data
    .ash-hawk/{suite_id}/runs/{run_id}/summary.json - Run summary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ash_hawk.types import (
    EvalRunSummary,
    EvalSuite,
    EvalTrial,
    RunEnvelope,
    ToolSurfacePolicy,
    TrialEnvelope,
)


@dataclass
class StoredTrial:
    """Bundled trial data for storage/retrieval.

    Contains the trial, its envelope, and policy snapshot together
    for complete reproducibility.
    """

    trial: EvalTrial
    envelope: TrialEnvelope
    policy: ToolSurfacePolicy


class StorageBackend(Protocol):
    """Protocol defining the storage interface for evaluation data.

    All methods are async to support both file-based and network-based
    storage backends (e.g., databases, cloud storage).
    """

    async def save_suite(self, suite: EvalSuite) -> None:
        """Save an evaluation suite definition.

        Args:
            suite: The suite to save.
        """
        ...

    async def load_suite(self, suite_id: str) -> EvalSuite | None:
        """Load an evaluation suite by ID.

        Args:
            suite_id: The suite identifier.

        Returns:
            The suite if found, None otherwise.
        """
        ...

    async def save_run_envelope(self, suite_id: str, envelope: RunEnvelope) -> None:
        """Save a run envelope.

        Args:
            suite_id: The suite this run belongs to.
            envelope: The run envelope to save.
        """
        ...

    async def load_run_envelope(self, suite_id: str, run_id: str) -> RunEnvelope | None:
        """Load a run envelope.

        Args:
            suite_id: The suite identifier.
            run_id: The run identifier.

        Returns:
            The envelope if found, None otherwise.
        """
        ...

    async def save_trial(
        self,
        suite_id: str,
        run_id: str,
        trial: EvalTrial,
        envelope: TrialEnvelope,
        policy: ToolSurfacePolicy,
    ) -> None:
        """Save a trial with its envelope and policy.

        Args:
            suite_id: The suite identifier.
            run_id: The run identifier.
            trial: The trial to save.
            envelope: The trial envelope.
            policy: The tool surface policy snapshot.
        """
        ...

    async def load_trial(self, suite_id: str, run_id: str, trial_id: str) -> StoredTrial | None:
        """Load a stored trial with all associated data.

        Args:
            suite_id: The suite identifier.
            run_id: The run identifier.
            trial_id: The trial identifier.

        Returns:
            StoredTrial if found, None otherwise.
        """
        ...

    async def list_runs(self, suite_id: str) -> list[str]:
        """List all run IDs for a suite.

        Args:
            suite_id: The suite identifier.

        Returns:
            List of run IDs.
        """
        ...

    async def list_suites(self) -> list[str]:
        """List all suite IDs.

        Returns:
            List of suite IDs.
        """
        ...

    async def save_summary(self, suite_id: str, run_id: str, summary: EvalRunSummary) -> None:
        """Save a run summary.

        Args:
            suite_id: The suite identifier.
            run_id: The run identifier.
            summary: The run summary to save.
        """
        ...

    async def load_summary(self, suite_id: str, run_id: str) -> EvalRunSummary | None:
        """Load a run summary.

        Args:
            suite_id: The suite identifier.
            run_id: The run identifier.

        Returns:
            The summary if found, None otherwise.
        """
        ...


from ash_hawk.storage.file import FileStorage
from ash_hawk.storage.sqlite import SQLiteStorage

__all__ = [
    "FileStorage",
    "SQLiteStorage",
    "StorageBackend",
    "StoredTrial",
]
