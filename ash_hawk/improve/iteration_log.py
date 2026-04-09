from __future__ import annotations

import json
import logging
from pathlib import Path

import pydantic as pd

from ash_hawk.improve.diagnose import Diagnosis

logger = logging.getLogger(__name__)


class DiagnosisSummary(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    trial_id: str = pd.Field(description="Trial that failed")
    failure_summary: str = pd.Field(description="One-line summary")
    root_cause: str = pd.Field(description="Root cause analysis")
    target_files: list[str] = pd.Field(default_factory=list)
    confidence: float = pd.Field(ge=0.0, le=1.0)
    actionable: bool = pd.Field(default=True)
    diagnosis_mode: str = pd.Field(default="llm")
    degraded_reason: str | None = pd.Field(default=None)


class IterationLog(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    iteration: int = pd.Field(description="Iteration number (0-indexed)")
    baseline_score: float = pd.Field(ge=0.0, le=1.0, description="Mean pass rate at baseline")
    baseline_repeats: int = pd.Field(ge=1, description="Number of eval repeats for baseline")
    failures: list[str] = pd.Field(default_factory=list, description="Trial IDs that failed")
    diagnoses: list[DiagnosisSummary] = pd.Field(
        default_factory=list, description="Diagnoses generated"
    )
    hypothesis_ranked: int = pd.Field(default=0, description="Number of hypotheses after ranking")
    hypothesis_attempted: str | None = pd.Field(
        default=None, description="Trial ID of the hypothesis that reached mutation generation"
    )
    hypothesis_outcome: str | None = pd.Field(
        default=None,
        description="High-level outcome for the attempted hypothesis",
    )
    hypothesis_tested: str | None = pd.Field(
        default=None, description="Trial ID of the tested hypothesis"
    )
    hypothesis_score: float | None = pd.Field(
        default=None, ge=0.0, le=1.0, description="Score after applying hypothesis"
    )
    delta: float | None = pd.Field(default=None, description="Score change")
    kept: bool | None = pd.Field(default=None, description="Whether the change was kept")
    lesson_id: str | None = pd.Field(default=None, description="Lesson ID if one was saved")
    stop_reasons: list[str] = pd.Field(
        default_factory=list, description="Stop conditions that fired"
    )
    error: str | None = pd.Field(default=None, description="Error if iteration failed")


def diagnosis_to_summary(diagnosis: Diagnosis) -> DiagnosisSummary:
    return DiagnosisSummary(
        trial_id=diagnosis.trial_id,
        failure_summary=diagnosis.failure_summary,
        root_cause=diagnosis.root_cause,
        target_files=diagnosis.target_files,
        confidence=diagnosis.confidence,
        actionable=diagnosis.actionable,
        diagnosis_mode=diagnosis.diagnosis_mode,
        degraded_reason=diagnosis.degraded_reason,
    )


def write_iteration_log(
    log: IterationLog,
    output_dir: Path | None = None,
) -> Path:
    dir_ = output_dir or Path(".ash-hawk/improve")
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / f"iter-{log.iteration:03d}.json"

    tmp_path = path.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(
            json.dumps(log.model_dump(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    logger.debug("Iteration log written: %s", path)
    return path
