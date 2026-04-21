# type-hygiene: skip-file
from __future__ import annotations

import asyncio
import json
import logging
import math
import statistics
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pydantic as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn

from ash_hawk.agents.source_workspace import (
    commit_agent_changes,
    detect_agent_config_path,
    prepare_isolated_agent_workspace,
)
from ash_hawk.improve.diagnose import Diagnosis, diagnose_failures
from ash_hawk.improve.failure_clustering import FailureCluster, cluster_diagnoses
from ash_hawk.improve.hypothesis_ranker import HypothesisRanker
from ash_hawk.improve.iteration_log import (
    IterationLog,
    diagnosis_to_summary,
    write_iteration_log,
)
from ash_hawk.improve.lesson_store import Lesson, LessonStore
from ash_hawk.improve.memory_store import (
    EpisodeRecord,
    MemoryStore,
    WorkingSnapshot,
)
from ash_hawk.improve.patch import (
    ProposedPatch,
    propose_patch,
    propose_patch_via_agent,
)
from ash_hawk.improve.phase1_review import Phase1Review, review_summary
from ash_hawk.improve.run_bundle import ImproveRunBundle
from ash_hawk.improve.stop_condition import ScoreRecord, StopCondition, StopConditionConfig
from ash_hawk.improve.targeting import diagnosis_targets_allowed, validate_diagnosis_targets
from ash_hawk.tracing import get_telemetry

if TYPE_CHECKING:
    from ash_hawk.agents.agent_mutator import AgentMutator
    from ash_hawk.types import EvalRunSummary

logger = logging.getLogger(__name__)
SPECULATIVE_BATCH_SIZE = 3
MUTATION_TIMEOUT_RATIO = 1.2
MIN_MUTATION_TIMEOUT_SECONDS = 120.0
MAX_MUTATION_TIMEOUT_SECONDS = 360.0
RETRY_MUTATION_TIMEOUT_MULTIPLIER = 1.25
RETRY_MAX_MUTATION_TIMEOUT_SECONDS = 450.0
MAX_TRANSIENT_RETRIES_PER_HYPOTHESIS = 1
TRANSIENT_MUTATION_OUTCOMES = {"mutation_cli_timeout", "post_mutation_eval_failed"}


def _format_path_list(paths: list[str], limit: int = 3) -> str:
    if not paths:
        return "none"
    shown = paths[:limit]
    suffix = "" if len(paths) <= limit else f" (+{len(paths) - limit} more)"
    return ", ".join(shown) + suffix


def _format_seconds(value: float) -> str:
    return f"{value:.1f}s"


def _format_phase_durations_summary(phase_totals: dict[str, float]) -> str:
    phase_order = [
        "baseline_eval",
        "diagnosis",
        "mutation_generation",
        "fast_validation",
        "integrity_validation",
    ]
    ordered_keys = [key for key in phase_order if key in phase_totals] + [
        key for key in sorted(phase_totals) if key not in phase_order
    ]
    parts = [f"{key}={_format_seconds(phase_totals[key])}" for key in ordered_keys]
    return ", ".join(parts)


def _truncate_text(value: str, limit: int = 160) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _mutation_timeout_seconds(
    iteration_timeout_seconds: float,
    *,
    multiplier: float = 1.0,
    max_seconds: float = MAX_MUTATION_TIMEOUT_SECONDS,
) -> float:
    scaled = iteration_timeout_seconds * MUTATION_TIMEOUT_RATIO * multiplier
    return max(MIN_MUTATION_TIMEOUT_SECONDS, min(max_seconds, scaled))


def _retry_timeout_seconds(iteration_timeout_seconds: float) -> float:
    return _mutation_timeout_seconds(
        iteration_timeout_seconds,
        multiplier=RETRY_MUTATION_TIMEOUT_MULTIPLIER,
        max_seconds=RETRY_MAX_MUTATION_TIMEOUT_SECONDS,
    )


def _is_retry_eligible(
    outcome: str,
    *,
    changed_paths: list[str],
    mutation_llm_calls: int,
    retry_count: int,
) -> bool:
    if retry_count >= MAX_TRANSIENT_RETRIES_PER_HYPOTHESIS:
        return False
    if outcome not in TRANSIENT_MUTATION_OUTCOMES:
        return False
    return bool(changed_paths) or mutation_llm_calls > 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _compute_phase2_metrics(
    *,
    initial_score: float,
    final_score: float,
    initial_pass_rate: float,
    final_pass_rate: float,
    mutation_history: list[dict[str, Any]],
    iteration_logs: list[dict[str, Any]],
    phase1_reviews: list[dict[str, Any]],
    stop_reasons: list[str],
) -> dict[str, Any]:
    tested_count = len(mutation_history)
    kept_count = sum(1 for entry in mutation_history if entry.get("kept"))
    timeout_count = sum(
        1 for entry in mutation_history if entry.get("rejection_reason") == "mutation_cli_timeout"
    )
    noop_outcomes = {
        "no_file_changes",
        "mutation_zero_tools",
        "mutation_parse_failed",
        "mutation_cli_error",
        "mutation_cli_timeout",
    }
    noop_count = sum(
        1 for entry in mutation_history if entry.get("rejection_reason") in noop_outcomes
    )
    targeted_regression_count = sum(
        1 for entry in mutation_history if entry.get("rejection_reason") == "targeted_regression"
    )
    intolerable_regression_reject_count = sum(
        1 for entry in mutation_history if entry.get("rejection_reason") == "intolerable_regression"
    )
    net_benefits = [
        float(entry.get("net_benefit", 0.0) or 0.0)
        for entry in mutation_history
        if isinstance(entry.get("net_benefit"), int | float)
    ]
    mutation_wall_seconds_values = [
        float(entry.get("mutation_wall_seconds", 0.0) or 0.0)
        for entry in mutation_history
        if isinstance(entry.get("mutation_wall_seconds"), int | float)
    ]
    mutation_llm_calls_values = [
        int(entry.get("mutation_llm_calls", 0) or 0)
        for entry in mutation_history
        if isinstance(entry.get("mutation_llm_calls"), int | float)
    ]
    failure_bucket_counts: dict[str, int] = {}
    suspicious_count = 0
    for review in phase1_reviews:
        if bool(review.get("suspicious", False)):
            suspicious_count += 1
        bucket = review.get("failure_bucket")
        if isinstance(bucket, str) and bucket:
            failure_bucket_counts[bucket] = failure_bucket_counts.get(bucket, 0) + 1

    phase1_review_count = len(phase1_reviews)
    delta_score = final_score - initial_score
    delta_pass_rate = final_pass_rate - initial_pass_rate
    net_benefit_total = sum(net_benefits)
    generated_count = sum(
        len(log.get("diagnoses", []))
        for log in iteration_logs
        if isinstance(log.get("diagnoses", []), list)
    )
    ranked_count = sum(
        int(log.get("hypothesis_ranked", 0) or 0)
        for log in iteration_logs
        if isinstance(log.get("hypothesis_ranked", 0), int | float)
    )
    reverted_count = max(0, tested_count - kept_count)
    score_pass_divergence = (delta_score * delta_pass_rate) < -0.001

    return {
        "tested_count": tested_count,
        "kept_count": kept_count,
        "keep_rate": _safe_rate(kept_count, tested_count),
        "timeout_count": timeout_count,
        "timeout_rate": _safe_rate(timeout_count, tested_count),
        "noop_count": noop_count,
        "noop_rate": _safe_rate(noop_count, tested_count),
        "targeted_regression_count": targeted_regression_count,
        "intolerable_regression_reject_count": intolerable_regression_reject_count,
        "delta_score": delta_score,
        "delta_pass_rate": delta_pass_rate,
        "net_benefit_total": net_benefit_total,
        "net_benefit_per_tested": net_benefit_total / tested_count if tested_count else 0.0,
        "mutation_wall_seconds_total": sum(mutation_wall_seconds_values),
        "mutation_wall_seconds_mean": _mean(mutation_wall_seconds_values),
        "mutation_llm_calls_total": sum(mutation_llm_calls_values),
        "mutation_llm_calls_mean": _mean([float(value) for value in mutation_llm_calls_values]),
        "phase1_suspicious_count": suspicious_count,
        "phase1_review_count": phase1_review_count,
        "phase1_suspicious_rate": _safe_rate(suspicious_count, phase1_review_count),
        "failure_bucket_counts": failure_bucket_counts,
        "funnel": {
            "generated": generated_count,
            "ranked": ranked_count,
            "attempted": tested_count,
            "kept": kept_count,
            "reverted": reverted_count,
            "attempt_rate": _safe_rate(tested_count, ranked_count),
            "keep_from_ranked_rate": _safe_rate(kept_count, ranked_count),
        },
        "progress_ledger": {
            "claims_quality": -failure_bucket_counts.get("false_claim", 0),
            "verification_behavior": -failure_bucket_counts.get("no_verification", 0),
            "capability_signal": tested_count,
        },
        "score_pass_rate_divergence": score_pass_divergence,
        "stop_reasons": list(stop_reasons),
    }


def _cohort_key(config: dict[str, Any]) -> str:
    suite_paths = config.get("suite_paths", [])
    if isinstance(suite_paths, list):
        normalized_paths = sorted(str(path) for path in suite_paths)
    else:
        normalized_paths = [str(config.get("suite_path", ""))]
    key_payload = {
        "suite_paths": normalized_paths,
        "agent_name": str(config.get("agent_name", "")),
        "score_threshold": float(config.get("score_threshold", 0.0) or 0.0),
        "eval_repeats": int(config.get("eval_repeats", 1) or 1),
        "integrity_repeats": int(config.get("integrity_repeats", 1) or 1),
    }
    return json.dumps(key_payload, sort_keys=True)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _evaluate_phase2_gate(
    run_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(run_entries) < 3:
        return {
            "status": "insufficient_data",
            "window_size": len(run_entries),
            "required_window": 3,
            "runs": run_entries,
            "failed_criteria": ["Need at least 3 comparable runs"],
        }

    window = run_entries[-3:]
    deltas = [float(entry["phase2_metrics"].get("delta_score", 0.0) or 0.0) for entry in window]
    timeout_rates = [
        float(entry["phase2_metrics"].get("timeout_rate", 0.0) or 0.0) for entry in window
    ]
    keep_rates = [float(entry["phase2_metrics"].get("keep_rate", 0.0) or 0.0) for entry in window]
    positive_delta_runs = sum(1 for delta in deltas if delta >= 0.01)
    median_delta = _median(deltas)
    median_timeout_rate = _median(timeout_rates)
    median_keep_rate = _median(keep_rates)
    has_negative_net_with_kept = any(
        float(entry["phase2_metrics"].get("net_benefit_total", 0.0) or 0.0) < 0.0
        and int(entry["phase2_metrics"].get("kept_count", 0) or 0) > 0
        for entry in window
    )

    failed_criteria: list[str] = []
    if positive_delta_runs < 2:
        failed_criteria.append("At least 2/3 runs must have delta_score >= +0.01")
    if median_delta < 0.02:
        failed_criteria.append("Median delta_score must be >= +0.02")
    if median_timeout_rate > 0.25:
        failed_criteria.append("Median timeout_rate must be <= 0.25")
    if median_keep_rate < 0.15:
        failed_criteria.append("Median keep_rate must be >= 0.15")
    if has_negative_net_with_kept:
        failed_criteria.append(
            "No run may have net_benefit_total < 0 while also keeping at least one mutation"
        )

    return {
        "status": "pass" if not failed_criteria else "fail",
        "window_size": len(window),
        "required_window": 3,
        "runs": window,
        "criteria": {
            "positive_delta_runs": positive_delta_runs,
            "median_delta_score": median_delta,
            "median_timeout_rate": median_timeout_rate,
            "median_keep_rate": median_keep_rate,
            "has_negative_net_with_kept": has_negative_net_with_kept,
        },
        "failed_criteria": failed_criteria,
    }


def _build_phase2_gate_report(
    *,
    improve_runs_root: Path,
    current_config: dict[str, Any],
    current_run_id: str,
    current_initial_score: float,
    current_final_score: float,
    current_phase2_metrics: dict[str, Any],
) -> dict[str, Any]:
    cohort = _cohort_key(current_config)
    entries: list[dict[str, Any]] = []
    total_run_dirs = sum(1 for p in improve_runs_root.iterdir() if p.is_dir())
    for run_dir in sorted(
        [p for p in improve_runs_root.iterdir() if p.is_dir()],
        key=lambda path: path.stat().st_mtime,
    ):
        config = _load_json(run_dir / "config.json")
        run = _load_json(run_dir / "run.json")
        if config is None:
            continue
        if _cohort_key(config) != cohort:
            continue

        phase2_metrics: dict[str, Any]
        if run_dir.name == current_run_id:
            phase2_metrics = current_phase2_metrics
            initial_score = current_initial_score
            final_score = current_final_score
        elif run is None:
            continue
        else:
            raw_phase2_metrics = run.get("phase2_metrics")
            if isinstance(raw_phase2_metrics, dict):
                phase2_metrics = raw_phase2_metrics
            else:
                phase2_metrics = _compute_phase2_metrics(
                    initial_score=float(run.get("initial_score", 0.0) or 0.0),
                    final_score=float(run.get("final_score", 0.0) or 0.0),
                    initial_pass_rate=float(run.get("initial_pass_rate", 0.0) or 0.0),
                    final_pass_rate=float(run.get("final_pass_rate", 0.0) or 0.0),
                    mutation_history=list(run.get("mutation_history", [])),
                    iteration_logs=list(run.get("iteration_logs", [])),
                    phase1_reviews=list(run.get("phase1_reviews", [])),
                    stop_reasons=list(run.get("stop_reasons", [])),
                )
            initial_score = float(run.get("initial_score", 0.0) or 0.0)
            final_score = float(run.get("final_score", 0.0) or 0.0)

        entries.append(
            {
                "run_id": run_dir.name,
                "initial_score": initial_score,
                "final_score": final_score,
                "phase2_metrics": phase2_metrics,
            }
        )

    gate = _evaluate_phase2_gate(entries)
    gate["cohort_key"] = cohort
    gate["current_run_id"] = current_run_id
    gate["cohort_comparability"] = {
        "cohort_size": len(entries),
        "total_runs": total_run_dirs,
        "comparable": len(entries) >= gate.get("required_window", 3),
    }
    return gate


def _normalize_target_path(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1]


def _causality_kind(target_files: list[str], changed_paths: list[str]) -> str:
    if not target_files or not changed_paths:
        return "unknown"
    target_names = {_normalize_target_path(path) for path in target_files}
    changed_names = {_normalize_target_path(path) for path in changed_paths}
    if target_names.intersection(changed_names):
        return "direct"
    return "indirect"


class AcceptanceDecision(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    kept: bool
    net_benefit: float
    targeted_gain: float
    regression_cost: float
    tolerable_regressions: list[dict[str, float | bool | str]] = pd.Field(default_factory=list)
    intolerable_regressions: list[dict[str, float | bool | str]] = pd.Field(default_factory=list)
    rejection_reason: str | None = None


def _hypothesis_console_label(hypothesis: Any) -> str:
    diagnosis = hypothesis.diagnosis
    summary = (
        diagnosis.failure_summary.strip() if diagnosis.failure_summary else diagnosis.suggested_fix
    )
    return f"rank={hypothesis.rank}  {summary}"


def _mutation_failure_outcome(patch: ProposedPatch, changed_paths: list[str]) -> str:
    if patch.failure_reason is not None:
        return patch.failure_reason
    return "mutation_parse_failed"


def _rerank_pending_hypotheses(
    ranker: HypothesisRanker,
    pending_hypotheses: list[Any],
    *,
    resolved_families: set[str],
    resolved_target_files: set[str],
    regressed_families: set[str],
    regressed_target_files: set[str],
    allowed_target_files: set[str],
) -> list[Any]:
    if not pending_hypotheses:
        return []
    reranked = ranker.rank(
        [hyp.diagnosis for hyp in pending_hypotheses],
        filter_tried=False,
        suppress_target_files=resolved_target_files,
        suppress_families=resolved_families,
        regression_target_files=regressed_target_files,
        regression_families=regressed_families,
        allowed_target_files=allowed_target_files,
    )
    return list(reranked.hypotheses)


def _filter_actionable_diagnoses_to_agent_source(
    diagnoses: list[Diagnosis],
    agent_content: dict[str, str],
) -> tuple[list[Diagnosis], list[Diagnosis]]:
    allowed_files = set(agent_content.keys())
    kept: list[Diagnosis] = []
    filtered: list[Diagnosis] = []
    for diagnosis in diagnoses:
        validate_diagnosis_targets(diagnosis, allowed_files)
        if not diagnosis.target_files:
            filtered.append(diagnosis)
            continue
        if diagnosis_targets_allowed(diagnosis, allowed_files):
            kept.append(diagnosis)
        else:
            filtered.append(diagnosis)
    return kept, filtered


def _diagnosis_stop_reason(
    *,
    actionable_diagnoses: list[Diagnosis],
    filtered_external_diagnoses: list[Diagnosis],
    non_actionable_diagnoses: list[Diagnosis],
) -> tuple[str, str]:
    if actionable_diagnoses:
        return "", ""
    if filtered_external_diagnoses and not non_actionable_diagnoses:
        return (
            "no_agent_scoped_diagnoses",
            "  [bold yellow]⚠ Diagnoses were found, but none targeted mutable agent files.[/bold yellow]",
        )
    return (
        "no_actionable_diagnoses",
        "  [bold yellow]⚠ No actionable diagnoses this pass, so mutation testing is skipped.[/bold yellow]",
    )


def _build_improve_commit_message(
    *,
    run_id: str,
    lesson: Lesson,
    hypothesis_rank: int,
    changed_paths: list[str],
) -> tuple[str, str]:
    title = f"fix: improve {lesson.trial_id} via lesson {lesson.lesson_id}"
    target_files = ", ".join(lesson.target_files) if lesson.target_files else "none"
    applied_files = ", ".join(changed_paths) if changed_paths else "none"
    body = "\n".join(
        [
            f"Improve run: {run_id}",
            f"Trial: {lesson.trial_id}",
            f"Lesson: {lesson.lesson_id}",
            f"Iteration: {lesson.iteration}",
            f"Hypothesis rank: {hypothesis_rank}",
            f"Outcome: {lesson.outcome}",
            f"Score: {lesson.score_before:.4f} -> {lesson.score_after:.4f} (Δ {lesson.score_delta:+.4f})",
            f"Target files: {target_files}",
            f"Applied files: {applied_files}",
            "",
            f"Hypothesis: {_truncate_text(lesson.hypothesis_summary)}",
            f"Root cause: {_truncate_text(lesson.root_cause, limit=240)}",
        ]
    )
    return title, body


class ImprovementResult(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    iterations: int = pd.Field(description="Total iterations run")
    initial_score: float = pd.Field(
        ge=0.0,
        le=1.0,
        description="Initial improvement score used by the improve loop",
    )
    final_score: float = pd.Field(
        ge=0.0,
        le=1.0,
        description="Final improvement score used by the improve loop",
    )
    initial_pass_rate: float = pd.Field(
        ge=0.0,
        le=1.0,
        description="Initial suite pass rate across evaluation repeats",
    )
    final_pass_rate: float = pd.Field(
        ge=0.0,
        le=1.0,
        description="Final suite pass rate across evaluation repeats",
    )
    patches_proposed: list[dict[str, Any]] = pd.Field(
        default_factory=list, description="Serialized ProposedPatch list"
    )
    patches_applied: list[str] = pd.Field(default_factory=list)
    trace_path: Path | None = None
    mutation_history: list[dict[str, Any]] = pd.Field(default_factory=list)
    iteration_logs: list[dict[str, Any]] = pd.Field(
        default_factory=list, description="Per-iteration structured logs"
    )
    phase1_reviews: list[dict[str, Any]] = pd.Field(
        default_factory=list,
        description="Latest Phase 1 suspicious-run reviews derived from traces",
    )
    phase2_metrics: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Per-run Phase 2 reliability and mutation-economics metrics",
    )
    phase2_gate: dict[str, Any] | None = pd.Field(
        default=None,
        description="Three-run Phase 2 gate verdict for comparable runs",
    )
    memory_summary: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Four-layer memory consolidation and retrieval summary",
    )
    convergence_achieved: bool = False
    stop_reasons: list[str] = pd.Field(default_factory=list, description="Final stop reasons")


class EvalRepeatAggregate(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    selected_score: float = pd.Field(ge=0.0, le=1.0)
    mean_score: float = pd.Field(ge=0.0, le=1.0)
    median_score: float = pd.Field(ge=0.0, le=1.0)
    min_score: float = pd.Field(ge=0.0, le=1.0)
    max_score: float = pd.Field(ge=0.0, le=1.0)
    selected_pass_rate: float = pd.Field(ge=0.0, le=1.0)
    mean_pass_rate: float = pd.Field(ge=0.0, le=1.0)
    median_pass_rate: float = pd.Field(ge=0.0, le=1.0)
    min_pass_rate: float = pd.Field(ge=0.0, le=1.0)
    max_pass_rate: float = pd.Field(ge=0.0, le=1.0)
    score_stddev: float = pd.Field(ge=0.0)
    pass_rate_stddev: float = pd.Field(ge=0.0)
    repeat_count: int = pd.Field(ge=1)


def _summarize_repeat_scores(scores: list[float], pass_rates: list[float]) -> EvalRepeatAggregate:
    mean_score = sum(scores) / len(scores)
    median_score = statistics.median(scores)
    mean_pass_rate = sum(pass_rates) / len(pass_rates)
    median_pass_rate = statistics.median(pass_rates)
    return EvalRepeatAggregate(
        selected_score=median_score if len(scores) > 1 else mean_score,
        mean_score=mean_score,
        median_score=median_score,
        min_score=min(scores),
        max_score=max(scores),
        selected_pass_rate=median_pass_rate if len(pass_rates) > 1 else mean_pass_rate,
        mean_pass_rate=mean_pass_rate,
        median_pass_rate=median_pass_rate,
        min_pass_rate=min(pass_rates),
        max_pass_rate=max(pass_rates),
        score_stddev=statistics.pstdev(scores),
        pass_rate_stddev=statistics.pstdev(pass_rates),
        repeat_count=len(scores),
    )


def _format_repeat_aggregate(stats: EvalRepeatAggregate) -> str:
    if stats.repeat_count <= 1:
        return ""
    return (
        f"  [dim](median {stats.median_score:.2%}, mean {stats.mean_score:.2%}, "
        f"range {stats.min_score:.2%}-{stats.max_score:.2%}, "
        f"score σ {stats.score_stddev:.4f}; "
        f"pass mean {stats.mean_pass_rate:.2%}, pass σ {stats.pass_rate_stddev:.4f})[/dim]"
    )


def _serialize_patch(patch: ProposedPatch) -> dict[str, Any]:
    return {
        "file_path": patch.file_path,
        "description": patch.description,
        "agent_relative_path": patch.agent_relative_path,
        "rationale": patch.rationale,
        "diagnosis_trial_id": patch.diagnosis.trial_id,
    }


def _normalize_suite_paths(suite_path: str | list[str]) -> list[str]:
    raw_paths = [suite_path] if isinstance(suite_path, str) else list(suite_path)
    return [str(Path(path).resolve()) for path in raw_paths]


def _trial_scenario_path(trial: Any) -> str | None:
    snapshot = getattr(trial, "input_snapshot", None)
    if isinstance(snapshot, dict):
        scenario_path = snapshot.get("scenario_path")
        if isinstance(scenario_path, str) and scenario_path:
            return scenario_path
    return None


def _mean_score_for_paths(summary: Any, scenario_paths: list[str]) -> float | None:
    wanted = set(scenario_paths)
    scores: list[float] = []
    for trial in getattr(summary, "trials", []):
        path = _trial_scenario_path(trial)
        if path in wanted and getattr(trial, "result", None) is not None:
            scores.append(float(trial.result.aggregate_score))
    if not scores:
        return None
    return sum(scores) / len(scores)


def _scenario_outcomes(summary: Any) -> dict[str, dict[str, float | bool]]:
    outcomes: dict[str, dict[str, float | bool]] = {}
    for trial in getattr(summary, "trials", []):
        path = _trial_scenario_path(trial)
        if path is None or getattr(trial, "result", None) is None:
            continue
        outcomes[path] = {
            "passed": bool(trial.result.aggregate_passed),
            "score": float(trial.result.aggregate_score),
        }
    return outcomes


def _find_regressed_unaffected_paths(
    baseline_summary: Any,
    candidate_summary: Any,
    targeted_paths: list[str],
    *,
    max_score_drop: float,
) -> list[dict[str, float | bool | str]]:
    targeted = set(targeted_paths)
    baseline = _scenario_outcomes(baseline_summary)
    candidate = _scenario_outcomes(candidate_summary)
    regressions: list[dict[str, float | bool | str]] = []
    for path, before in baseline.items():
        if path in targeted:
            continue
        after = candidate.get(path)
        if after is None:
            continue
        before_passed = bool(before["passed"])
        after_passed = bool(after["passed"])
        before_score = float(before["score"])
        after_score = float(after["score"])
        if before_passed and not after_passed:
            regressions.append(
                {
                    "scenario_path": path,
                    "reason": "pass_to_fail",
                    "severity": "intolerable",
                    "before_passed": before_passed,
                    "after_passed": after_passed,
                    "before_score": before_score,
                    "after_score": after_score,
                    "score_drop": before_score - after_score,
                }
            )
        elif before_score - after_score > max_score_drop:
            regressions.append(
                {
                    "scenario_path": path,
                    "reason": "score_drop",
                    "severity": "tolerable",
                    "before_passed": before_passed,
                    "after_passed": after_passed,
                    "before_score": before_score,
                    "after_score": after_score,
                    "score_drop": before_score - after_score,
                }
            )
    return regressions


def _acceptance_decision(
    *,
    delta: float,
    regressions: list[dict[str, float | bool | str]],
    score_threshold: float,
) -> AcceptanceDecision:
    if delta <= 0:
        return AcceptanceDecision(
            kept=False,
            net_benefit=delta,
            targeted_gain=delta,
            regression_cost=0.0,
            rejection_reason="targeted_regression",
        )

    tolerable_regressions: list[dict[str, float | bool | str]] = []
    intolerable_regressions: list[dict[str, float | bool | str]] = []
    tolerable_cost = 0.0
    intolerable_cost = 0.0
    intolerable_drop_threshold = score_threshold * 3

    for regression in regressions:
        score_drop = float(regression.get("score_drop", 0.0))
        reason = str(regression.get("reason", "score_drop"))
        classified = dict(regression)
        if reason == "pass_to_fail" or score_drop > intolerable_drop_threshold:
            classified["severity"] = "intolerable"
            intolerable_regressions.append(classified)
            intolerable_cost += score_drop
        else:
            classified["severity"] = "tolerable"
            tolerable_regressions.append(classified)
            tolerable_cost += score_drop

    regression_cost = intolerable_cost + (tolerable_cost * 0.5)
    net_benefit = delta - regression_cost
    kept = delta > score_threshold and not intolerable_regressions and net_benefit > 0
    rejection_reason: str | None = None
    if not kept:
        if intolerable_regressions:
            rejection_reason = "intolerable_regression"
        elif net_benefit <= 0:
            rejection_reason = "net_negative"
        else:
            rejection_reason = "below_keep_threshold"

    return AcceptanceDecision(
        kept=kept,
        net_benefit=net_benefit,
        targeted_gain=delta,
        regression_cost=regression_cost,
        tolerable_regressions=tolerable_regressions,
        intolerable_regressions=intolerable_regressions,
        rejection_reason=rejection_reason,
    )


def _cluster_targeted_paths(
    cluster: FailureCluster,
    failure_by_id: dict[str, Any],
    suite_paths: list[str],
) -> list[str]:
    targeted_paths: list[str] = []
    for trial_id in cluster.trial_ids:
        failure_trial = failure_by_id.get(trial_id)
        if failure_trial is None:
            continue
        scenario_path = _trial_scenario_path(failure_trial)
        if scenario_path and scenario_path not in targeted_paths:
            targeted_paths.append(scenario_path)
    if not targeted_paths:
        return suite_paths
    nearby_path = _find_nearby_validation_path(targeted_paths, suite_paths)
    if nearby_path is not None and nearby_path not in targeted_paths:
        targeted_paths.append(nearby_path)
    return targeted_paths


def _scenario_family_key(path_str: str) -> str:
    name = Path(path_str).name.removesuffix(".scenario.yaml")
    if "__" in name:
        return name.split("__", maxsplit=1)[0]
    if name.startswith("mvp_"):
        return "mvp"
    if name.startswith("policy_"):
        return "policy"
    return name


def _find_nearby_validation_path(targeted_paths: list[str], suite_paths: list[str]) -> str | None:
    if len(suite_paths) <= 1 or not targeted_paths:
        return None
    targeted = set(targeted_paths)
    anchor = targeted_paths[0]
    anchor_family = _scenario_family_key(anchor)
    family_candidates: list[str] = []
    for candidate in suite_paths:
        if candidate in targeted:
            continue
        if _scenario_family_key(candidate) == anchor_family:
            family_candidates.append(candidate)
    if family_candidates:
        return family_candidates[0]
    return None


def find_nearby_validation_path(targeted_paths: list[str], suite_paths: list[str]) -> str | None:
    return _find_nearby_validation_path(targeted_paths, suite_paths)


def _phase1_reviews(summary: EvalRunSummary | None) -> list[Phase1Review]:
    if summary is None:
        return []
    return review_summary(summary)


def _eval_progress_labels(suite_paths: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    try:
        from ash_hawk.scenario.loader import load_scenario
    except Exception:
        load_scenario = None  # type: ignore[assignment]

    for path_str in suite_paths:
        path = Path(path_str)
        label = path.name
        if load_scenario is not None:
            try:
                scenario = load_scenario(path)
                label = scenario.id
            except Exception:
                label = path.name
        labels[label] = label
    return labels


async def _speculative_attempt(
    *,
    hypothesis: Any,
    iteration: int,
    agent_name: str,
    agent_path: Path | None,
    agent_content: dict[str, str] | None,
    suite_paths: list[str],
    targeted_paths: list[str],
    failure_by_id: dict[str, Any],
    iteration_timeout_seconds: float,
    eval_repeats: int,
    retry_count: int = 0,
    mutation_timeout_seconds_override: float | None = None,
    console: Console | None = None,
    run_bundle: ImproveRunBundle | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "hypothesis": hypothesis,
        "targeted_paths": targeted_paths,
        "kept_candidate": False,
    }
    if agent_path is None:
        result["outcome"] = "no_agent_path"
        return result

    if run_bundle is None:
        raise ValueError("run_bundle is required")

    workspace = prepare_isolated_agent_workspace(
        agent_path,
        run_id="improve",
        workspace_id=f"iter-{iteration}-rank-{hypothesis.rank}",
    )
    result["workspace"] = workspace
    hypothesis_agent_path = workspace.workspace_agent_path
    from ash_hawk.agents.agent_mutator import AgentMutator

    hypothesis_mutator = AgentMutator(
        hypothesis_agent_path,
        run_id=f"improve-iter-{iteration}-rank-{hypothesis.rank}",
    )
    result["hypothesis_mutator"] = hypothesis_mutator
    hypothesis_snapshot = hypothesis_mutator.snapshot()

    grader_details = ""
    transcript_excerpt = ""
    failure_trial = failure_by_id.get(hypothesis.diagnosis.trial_id)
    if failure_trial is not None and failure_trial.result is not None:
        grader_details = json.dumps(
            [r.model_dump() for r in failure_trial.result.grader_results], default=str
        )
        tc = failure_trial.result.transcript
        msgs = getattr(tc, "messages", [])
        tool_calls = getattr(tc, "tool_calls", [])
        lines: list[str] = []
        for m in msgs:
            if isinstance(m, dict):
                role = str(m.get("role", "?"))
                content = str(m.get("content", ""))
            else:
                role = str(getattr(m, "role", "?"))
                content = str(getattr(m, "content", ""))
            lines.append(f"[{role}]\n{content}\n")
        for tc_item in tool_calls:
            if isinstance(tc_item, dict):
                name = str(tc_item.get("name") or tc_item.get("tool") or "?")
                arguments = tc_item.get("arguments") or tc_item.get("input") or {}
            else:
                name = str(getattr(tc_item, "name", "?"))
                arguments = getattr(tc_item, "arguments", {})
            args = json.dumps(arguments, default=str)
            lines.append(f"[tool_call] {name}({args})")
        transcript_excerpt = "\n".join(lines)

    hypothesis_config_path = detect_agent_config_path(hypothesis_agent_path)
    hypothesis_repo_root = workspace.workspace_root
    mutation_timeout = (
        mutation_timeout_seconds_override
        if mutation_timeout_seconds_override is not None
        else _mutation_timeout_seconds(iteration_timeout_seconds)
    )
    retry_suffix = f"-retry-{retry_count}" if retry_count > 0 else ""

    if console is not None:
        console.print(f"    [dim]Hypothesis:[/dim] {_hypothesis_console_label(hypothesis)}")

    mutation_started_at = time.monotonic()
    try:
        patch = await propose_patch_via_agent(
            hypothesis.diagnosis,
            hypothesis_agent_path,
            agent_content,
            grader_details=grader_details,
            transcript_excerpt=transcript_excerpt,
            console=console,
            config_path=hypothesis_config_path,
            repo_root=hypothesis_repo_root,
            timeout_seconds=mutation_timeout,
            audit_bundle=run_bundle,
            audit_stem=f"iter-{iteration:03d}/rank-{hypothesis.rank}-mutate{retry_suffix}",
        )
    except ImportError:
        patch = await propose_patch(hypothesis.diagnosis, agent_content, console=console)

    mutation_wall_seconds = time.monotonic() - mutation_started_at

    result["patch"] = patch

    if patch.agent_relative_path and patch.content:
        if patch.agent_relative_path == "(agent-edited)":
            pass
        else:
            hypothesis_mutator.write_file(patch.agent_relative_path, patch.content)

    changed_paths = sorted(hypothesis_mutator.diff_since_snapshot(hypothesis_snapshot).keys())
    result["changed_paths"] = changed_paths
    result["execution_metrics"] = patch.execution_metrics
    result["mutation_wall_seconds"] = mutation_wall_seconds
    execution_metrics = patch.execution_metrics or {}
    llm_completion_count = 0
    raw_llm_count = execution_metrics.get("llm_completion_count", 0)
    if isinstance(raw_llm_count, int | float):
        llm_completion_count = int(raw_llm_count)
    result["mutation_llm_calls"] = llm_completion_count
    result["retry_count"] = retry_count

    if not (patch.agent_relative_path and patch.content) and changed_paths:
        patch.recovered_from_changed_files = True

    if not (patch.agent_relative_path and patch.content) and not changed_paths:
        result["outcome"] = _mutation_failure_outcome(patch, changed_paths)
        workspace.cleanup()
        hypothesis_mutator.cleanup()
        return result

    if not changed_paths:
        result["outcome"] = "no_file_changes"
        workspace.cleanup()
        hypothesis_mutator.cleanup()
        return result

    run_bundle.write_json(
        f"mutations/iter-{iteration:03d}/rank-{hypothesis.rank}.json",
        {
            "trial_id": hypothesis.diagnosis.trial_id,
            "rank": hypothesis.rank,
            "target_files": hypothesis.diagnosis.target_files,
            "changed_paths": changed_paths,
            "changed_file_contents": {
                path: hypothesis_mutator.read_file(path) for path in changed_paths
            },
            "execution_metrics": patch.execution_metrics,
            "patch": patch.model_dump()
            if hasattr(patch, "model_dump")
            else _serialize_patch(patch),
        },
    )

    fast_validation_started_at = time.monotonic()
    fast_validation_stats, fast_validation_summary, _ = await _run_eval_n_times(
        targeted_paths,
        agent_name,
        iteration_timeout_seconds,
        hypothesis_agent_path,
        eval_repeats,
        console=console,
        phase_label="fast validation",
        audit_bundle=run_bundle,
        audit_stem=f"iter-{iteration:03d}/rank-{hypothesis.rank}/fast-validation{retry_suffix}",
    )
    result["fast_validation_wall_seconds"] = time.monotonic() - fast_validation_started_at
    result["fast_validation_stats"] = fast_validation_stats
    result["fast_validation_summary"] = fast_validation_summary
    if fast_validation_stats is None:
        result["outcome"] = "post_mutation_eval_failed"
        workspace.cleanup()
        hypothesis_mutator.cleanup()
        return result

    result["outcome"] = "speculative_ready"
    result["kept_candidate"] = True
    return result


async def improve(
    suite_path: str | list[str],
    agent_name: str = "build",
    agent_path: Path | None = None,
    target: float = 1.0,
    max_iterations: int = 5,
    trace_dir: Path | None = None,
    output_dir: Path | None = None,
    iteration_timeout_seconds: float = 300.0,
    eval_repeats: int = 1,
    integrity_repeats: int | None = None,
    score_threshold: float = 0.02,
    lessons_dir: Path | None = None,
    stop_config: StopConditionConfig | None = None,
    overall_timeout_seconds: float | None = None,
    console: Console | None = None,
) -> ImprovementResult:
    suite_paths = _normalize_suite_paths(suite_path)
    run_bundle = ImproveRunBundle(output_dir)
    run_started_at = time.monotonic()
    iteration_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else run_bundle.path.parent.parent / "improve"
    )
    lesson_store = LessonStore(lessons_dir=lessons_dir)
    memory_base_dir = (lessons_dir.parent / "memory") if lessons_dir is not None else None
    memory_store = MemoryStore(base_dir=memory_base_dir)
    ranker = HypothesisRanker(lesson_store=lesson_store, memory_store=memory_store)
    backfilled_episodes = memory_store.backfill_from_run_artifacts(run_bundle.path.parent)
    stop_condition = StopCondition(config=stop_config)
    resolved_families: set[str] = set()
    resolved_target_files: set[str] = set()
    regressed_families: set[str] = set()
    regressed_target_files: set[str] = set()

    patches: list[dict[str, Any]] = []
    initial_score = 0.0
    final_score = 0.0
    initial_pass_rate = 0.0
    final_pass_rate = 0.0
    mutation_history: list[dict[str, Any]] = []
    iteration_logs: list[dict[str, Any]] = []
    convergence_achieved = False
    final_stop_reasons: list[str] = []
    applied_files: set[str] = set()
    actual_iterations = 0
    latest_phase1_reviews: list[Phase1Review] = []
    memory_skip_count = 0
    agent_content: dict[str, str] | None = None
    allowed_target_files: set[str] = set()
    original_mutator: AgentMutator | None = None

    if integrity_repeats is None:
        integrity_repeats = max(eval_repeats, 3)

    if agent_path is not None:
        from ash_hawk.agents.agent_mutator import AgentMutator

        uuid_hex = uuid.uuid4().hex
        original_mutator = AgentMutator(agent_path, run_id=f"improve-{uuid_hex[:8]}")
        agent_content = original_mutator.scan()
        allowed_target_files = set(agent_content.keys())

    if console is not None:
        console.print(f"[cyan]Improve run bundle:[/cyan] {run_bundle.path}")
        console.print(
            f"[cyan]Run plan:[/cyan] baseline eval x{eval_repeats}, integrity eval x{integrity_repeats}, target {target:.0%}"
        )
        if agent_content is not None:
            console.print(
                f"[cyan]Loaded agent source:[/cyan] {len(agent_content)} text file(s) ready for mutation analysis"
            )
        if backfilled_episodes:
            console.print(
                f"[cyan]Memory warm start:[/cyan] backfilled {backfilled_episodes} episode(s)"
            )
        console.print()

    run_config = {
        "suite_path": suite_path,
        "suite_paths": suite_paths,
        "agent_name": agent_name,
        "agent_path": str(agent_path) if agent_path else None,
        "target": target,
        "max_iterations": max_iterations,
        "trace_dir": str(trace_dir) if trace_dir else None,
        "output_dir": str(output_dir) if output_dir else None,
        "iteration_timeout_seconds": iteration_timeout_seconds,
        "eval_repeats": eval_repeats,
        "integrity_repeats": integrity_repeats,
        "score_threshold": score_threshold,
        "lessons_dir": str(lessons_dir) if lessons_dir else None,
        "memory_dir": str(memory_store.base_dir),
        "stop_config": stop_config.model_dump() if stop_config else None,
        "overall_timeout_seconds": overall_timeout_seconds,
    }
    run_bundle.write_json(
        "config.json",
        run_config,
    )
    run_bundle.append_event("improve.run_started", suite_path=suite_path, agent_name=agent_name)

    try:
        for i in range(max_iterations):
            if overall_timeout_seconds is not None:
                elapsed = time.monotonic() - run_started_at
                if elapsed >= overall_timeout_seconds:
                    final_stop_reasons = ["overall_timeout"]
                    logger.warning(
                        "Overall timeout reached before iteration %d (elapsed %.1fs / budget %.1fs)",
                        i,
                        elapsed,
                        overall_timeout_seconds,
                    )
                    break
            actual_iterations += 1
            phase_durations: dict[str, float] = {}
            memory_store.save_working(
                WorkingSnapshot(
                    run_id=run_bundle.run_id,
                    iteration=i,
                    active_trial_ids=[],
                    active_families=[],
                    active_target_files=[],
                    hypothesis_count=0,
                    baseline_score=None,
                    baseline_pass_rate=None,
                    last_hypothesis_outcome=None,
                    memory_skip_count=memory_skip_count,
                    stop_reasons=final_stop_reasons,
                )
            )
            if console is not None:
                console.print(
                    f"  [bold]Outer pass {i + 1}/{max_iterations}[/bold]  "
                    f"[dim]Step 1: baseline evaluation[/dim]"
                )
                console.print(
                    "  [dim]Each outer pass runs the full suite, then diagnoses only the failures from that fresh baseline run.[/dim]"
                )

            baseline_started_at = time.monotonic()
            baseline_stats, last_summary, eval_errors = await _run_eval_n_times(
                suite_paths,
                agent_name,
                iteration_timeout_seconds,
                agent_path,
                eval_repeats,
                console=console,
                phase_label="baseline evaluation",
                audit_bundle=run_bundle,
                audit_stem=f"iter-{i:03d}/baseline",
            )
            phase_durations["baseline_eval"] = time.monotonic() - baseline_started_at

            if baseline_stats is None:
                logger.warning("Iteration %d: all eval runs failed, skipping", i)
                if console is not None:
                    console.print(
                        f"  [bold red]✗ Iteration {i + 1}:[/bold red] "
                        f"All {len(eval_errors)} eval run(s) failed"
                    )
                get_telemetry().emit(
                    "improve.iteration_all_evals_failed",
                    iteration=i,
                    suite=suite_path,
                    errors=eval_errors,
                )
                iter_log = IterationLog(
                    iteration=i,
                    baseline_score=0.0,
                    baseline_repeats=eval_repeats,
                    error=f"All {len(eval_errors)} eval runs failed",
                    phase_durations=phase_durations,
                )
                memory_store.append_episode(
                    EpisodeRecord(
                        episode_id=uuid.uuid4().hex[:12],
                        run_id=run_bundle.run_id,
                        agent_name=agent_name,
                        iteration=i,
                        trial_id=f"baseline-eval-{i}",
                        diagnosis_family="baseline_eval",
                        target_files=[],
                        outcome="baseline_eval_failed",
                        attempted=False,
                        kept=False,
                        metadata={"errors": eval_errors, "source_kind": "baseline_failure"},
                    )
                )
                iteration_logs.append(iter_log.model_dump())
                write_iteration_log(iter_log, iteration_output_dir)
                continue

            if i == 0:
                initial_score = baseline_stats.selected_score
                initial_pass_rate = baseline_stats.selected_pass_rate
            final_score = baseline_stats.selected_score
            final_pass_rate = baseline_stats.selected_pass_rate

            logger.debug(
                "Iteration %d: selected_score=%.4f mean_score=%.4f median_score=%.4f (runs=%d) target=%.2f",
                i,
                baseline_stats.selected_score,
                baseline_stats.mean_score,
                baseline_stats.median_score,
                eval_repeats,
                target,
            )

            if console is not None:
                color = "green" if baseline_stats.selected_score >= target else "yellow"
                trial_info = ""
                if last_summary is not None:
                    passed = last_summary.metrics.passed_tasks
                    total = last_summary.metrics.completed_tasks
                    trial_info = f"  trials=[{passed}/{total} passed]"
                console.print(
                    f"  [bold]Baseline result:[/bold] outer pass {i + 1}/{max_iterations}  "
                    f"score=[{color}]{baseline_stats.selected_score:.2%}[/{color}]"
                    f"  pass=[{color}]{baseline_stats.selected_pass_rate:.2%}[/{color}]"
                    f"{trial_info}  "
                    f"target={target:.0%}"
                    f"{_format_repeat_aggregate(baseline_stats)}"
                )

            if baseline_stats.selected_score >= target:
                logger.info("Target reached at iteration %d", i)
                if console is not None:
                    console.print(
                        f"  [bold green]✓ Target reached: "
                        f"{baseline_stats.selected_score:.2%} >= {target:.0%}[/bold green]"
                    )
                convergence_achieved = True

                iter_log = IterationLog(
                    iteration=i,
                    baseline_score=baseline_stats.selected_score,
                    baseline_repeats=eval_repeats,
                    stop_reasons=["target_reached"],
                    phase_durations=phase_durations,
                )
                iteration_logs.append(iter_log.model_dump())
                write_iteration_log(iter_log, iteration_output_dir)
                break

            if last_summary is None:
                logger.warning("No eval summary available for iteration %d, continuing", i)
                continue

            if not last_summary.trials:
                logger.warning(
                    "No trials completed in re-evaluation for iteration %d, continuing", i
                )
                get_telemetry().emit(
                    "improve.no_trials_from_reeval",
                    iteration=i,
                    suite=suite_path,
                )
                continue

            phase1_reviews = _phase1_reviews(last_summary)
            latest_phase1_reviews = phase1_reviews
            review_by_trial = {review.trial_id: review for review in phase1_reviews}
            suspicious_reviews = [review for review in phase1_reviews if review.suspicious]
            failure_bucket_by_trial: dict[str, str] = {
                review.trial_id: review.failure_bucket
                for review in phase1_reviews
                if review.failure_bucket is not None
            }
            run_bundle.write_json(
                f"reviews/iter-{i:03d}/baseline_phase1.json",
                [review.model_dump(mode="json") for review in phase1_reviews],
            )
            if console is not None and suspicious_reviews:
                suspicious_labels = [
                    f"{review.trial_id}:{review.failure_bucket or 'review'}"
                    for review in suspicious_reviews
                ]
                console.print(
                    f"  [yellow]Phase 1 review:[/yellow] {len(suspicious_reviews)} suspicious run(s)  "
                    f"ids={_format_path_list(suspicious_labels)}"
                )

            failures = [
                t for t in last_summary.trials if t.result is None or not t.result.aggregate_passed
            ]
            failure_by_id = {t.id: t for t in failures}
            get_telemetry().emit(
                "improve.failures_detected",
                iteration=i,
                failure_count=len(failures),
                trial_ids=[t.id for t in failures],
            )
            if console is not None:
                failed_ids = [t.id for t in failures]
                console.print(
                    f"  [yellow]Failures found:[/yellow] {len(failures)}  ids={_format_path_list(failed_ids)}"
                )
            if not failures:
                logger.info("No failures found in iteration %d, stopping", i)
                if console is not None:
                    console.print("  [bold green]✓ All scenarios passing[/bold green]")
                iter_log = IterationLog(
                    iteration=i,
                    baseline_score=baseline_stats.selected_score,
                    baseline_repeats=eval_repeats,
                    suspicious_trials=[review.trial_id for review in suspicious_reviews],
                    failure_buckets=failure_bucket_by_trial,
                    stop_reasons=["all_passing"],
                    phase_durations=phase_durations,
                )
                iteration_logs.append(iter_log.model_dump())
                write_iteration_log(iter_log, iteration_output_dir)
                break

            if console is not None:
                console.print(f"  [dim]Step 2: diagnosing {len(failures)} failure(s)...[/dim]")
            try:
                diagnosis_started_at = time.monotonic()
                diagnoses = await diagnose_failures(
                    failures,
                    trace_dir,
                    agent_content,
                    agent_path=agent_path,
                    lesson_store=lesson_store,
                    personal_preferences=memory_store.format_personal_preferences_for_prompt(),
                    console=console,
                    audit_bundle=run_bundle,
                    audit_iteration=i,
                )
                phase_durations["diagnosis"] = time.monotonic() - diagnosis_started_at
            except Exception:
                logger.warning("Diagnosis failed in iteration %d", i, exc_info=True)
                if console is not None:
                    console.print(f"  [bold red]✗ Diagnosis failed in iteration {i + 1}[/bold red]")
                continue

            logger.info("Diagnosed %d failures in iteration %d", len(diagnoses), i)
            actionable_diagnoses = [d for d in diagnoses if d.actionable]
            non_actionable_diagnoses = [d for d in diagnoses if not d.actionable]
            filtered_external_diagnoses: list[Diagnosis] = []
            if agent_content:
                actionable_diagnoses, filtered_external_diagnoses = (
                    _filter_actionable_diagnoses_to_agent_source(
                        actionable_diagnoses, agent_content
                    )
                )
            if console is not None:
                console.print(
                    f"  [cyan]Diagnoses:[/cyan] "
                    f"{len(diagnoses)} generated from {len(failures)} failure(s)"
                )
                if len(diagnoses) == len(failures) == 1:
                    console.print(
                        "  [dim]One failing trial produced one diagnosis, so this pass has one hypothesis candidate to test.[/dim]"
                    )
                elif len(diagnoses) < len(failures):
                    console.print(
                        "  [dim]Some failures did not turn into diagnoses, so they will not produce hypotheses in this pass.[/dim]"
                    )
                if non_actionable_diagnoses:
                    reasons = sorted(
                        {
                            diagnosis.degraded_reason or diagnosis.diagnosis_mode
                            for diagnosis in non_actionable_diagnoses
                        }
                    )
                    console.print(
                        f"  [yellow]Non-actionable diagnoses:[/yellow] {len(non_actionable_diagnoses)}  reasons={_format_path_list(reasons)}"
                    )
                if filtered_external_diagnoses:
                    console.print(
                        f"  [yellow]Filtered non-agent diagnoses:[/yellow] {len(filtered_external_diagnoses)}"
                    )

            if not actionable_diagnoses:
                stop_reason, stop_message = _diagnosis_stop_reason(
                    actionable_diagnoses=actionable_diagnoses,
                    filtered_external_diagnoses=filtered_external_diagnoses,
                    non_actionable_diagnoses=non_actionable_diagnoses,
                )
                if console is not None:
                    console.print(stop_message)
                    console.print(
                        "  [dim]The improver will stop here because rerunning the same suite without target files or a usable diagnosis would only create more placeholder output.[/dim]"
                    )
                final_stop_reasons = [stop_reason]
                iter_log = IterationLog(
                    iteration=i,
                    baseline_score=baseline_stats.selected_score,
                    baseline_repeats=eval_repeats,
                    failures=[t.id for t in failures],
                    diagnoses=[diagnosis_to_summary(d) for d in diagnoses],
                    suspicious_trials=[review.trial_id for review in suspicious_reviews],
                    failure_buckets=failure_bucket_by_trial,
                    hypothesis_ranked=0,
                    hypothesis_outcome=stop_reason,
                    stop_reasons=final_stop_reasons,
                    phase_durations=phase_durations,
                )
                iteration_logs.append(iter_log.model_dump())
                write_iteration_log(iter_log, iteration_output_dir)
                break

            failure_clusters = cluster_diagnoses(actionable_diagnoses)
            cluster_by_id = {cluster.cluster_id: cluster for cluster in failure_clusters}
            run_bundle.write_json(
                f"hypotheses/iter-{i:03d}/clusters.json",
                [cluster.model_dump() for cluster in failure_clusters],
            )

            ranking = ranker.rank(
                [cluster.representative for cluster in failure_clusters],
                suppress_target_files=resolved_target_files,
                suppress_families=resolved_families,
                regression_target_files=regressed_target_files,
                regression_families=regressed_families,
                allowed_target_files=allowed_target_files,
            )
            run_bundle.write_json(f"hypotheses/iter-{i:03d}/ranking.json", ranking.model_dump())

            logger.debug(
                "Iteration %d: %d hypotheses ranked (%d filtered as already tried)",
                i,
                ranking.total_candidates,
                ranking.filtered_as_tried,
            )

            if console is not None and ranking.hypotheses:
                console.print(
                    f"  [cyan]Hypotheses:[/cyan] "
                    f"{ranking.total_candidates} generated, "
                    f"{ranking.filtered_as_tried} filtered (already tried)"
                )
                console.print(
                    f"  [dim]Failure clusters:[/dim] {len(failure_clusters)} grouped from {len(actionable_diagnoses)} actionable diagnoses"
                )
                top_targets = [
                    _format_path_list(h.diagnosis.target_files, limit=2)
                    for h in ranking.hypotheses[:3]
                ]
                console.print(
                    f"  [dim]Top targets:[/dim] {_format_path_list(top_targets, limit=3)}"
                )

            memory_store.save_working(
                WorkingSnapshot(
                    run_id=run_bundle.run_id,
                    iteration=i,
                    active_trial_ids=[h.diagnosis.trial_id for h in ranking.hypotheses],
                    active_families=sorted({h.diagnosis.family for h in ranking.hypotheses}),
                    active_target_files=sorted(
                        {path for h in ranking.hypotheses for path in h.diagnosis.target_files}
                    ),
                    hypothesis_count=len(ranking.hypotheses),
                    baseline_score=baseline_stats.selected_score,
                    baseline_pass_rate=baseline_stats.selected_pass_rate,
                    last_hypothesis_outcome=None,
                    memory_skip_count=memory_skip_count,
                    stop_reasons=final_stop_reasons,
                )
            )

            if not ranking.hypotheses:
                if console is not None:
                    console.print(
                        "  [bold yellow]⚠ No hypotheses remain after filtering already-tried ideas.[/bold yellow]"
                    )
                final_stop_reasons = ["no_ranked_hypotheses"]
                iter_log = IterationLog(
                    iteration=i,
                    baseline_score=baseline_stats.selected_score,
                    baseline_repeats=eval_repeats,
                    failures=[t.id for t in failures],
                    diagnoses=[diagnosis_to_summary(d) for d in diagnoses],
                    suspicious_trials=[review.trial_id for review in suspicious_reviews],
                    failure_buckets=failure_bucket_by_trial,
                    hypothesis_ranked=0,
                    hypothesis_outcome="no_ranked_hypotheses",
                    stop_reasons=final_stop_reasons,
                    phase_durations=phase_durations,
                )
                iteration_logs.append(iter_log.model_dump())
                write_iteration_log(iter_log, iteration_output_dir)
                break

            baseline_score = baseline_stats.selected_score

            iteration_kept: bool | None = None
            iteration_hypothesis_attempted: str | None = None
            iteration_hypothesis_outcome: str | None = None
            iteration_hypothesis_tested: str | None = None
            iteration_hypothesis_score: float | None = None
            iteration_delta: float | None = None
            iteration_lesson_id: str | None = None
            iteration_mutation_wall_seconds: float | None = None
            iteration_mutation_llm_calls: int | None = None
            pending_hypotheses = list(ranking.hypotheses)
            while pending_hypotheses:
                if overall_timeout_seconds is not None:
                    elapsed = time.monotonic() - run_started_at
                    if elapsed >= overall_timeout_seconds:
                        final_stop_reasons = ["overall_timeout"]
                        break
                batch = pending_hypotheses[:SPECULATIVE_BATCH_SIZE]
                remainder = pending_hypotheses[SPECULATIVE_BATCH_SIZE:]

                filtered_batch: list[Any] = []
                for hyp in batch:
                    should_skip, skip_reason, skip_metrics = memory_store.should_skip_hypothesis(
                        agent_name=agent_name,
                        diagnosis_family=hyp.diagnosis.family,
                        target_files=hyp.diagnosis.target_files,
                    )
                    if not should_skip:
                        filtered_batch.append(hyp)
                        continue

                    memory_skip_count += 1
                    iteration_hypothesis_outcome = skip_reason or "memory_skip_low_conversion"
                    if console is not None:
                        console.print(
                            f"    [yellow]↷ Memory skip rank={hyp.rank} trial={hyp.diagnosis.trial_id} "
                            f"(attempts={int(skip_metrics.get('attempts', 0))}, "
                            f"success={float(skip_metrics.get('success_rate', 0.0)):.2f})[/yellow]"
                        )

                    memory_store.append_episode(
                        EpisodeRecord(
                            episode_id=uuid.uuid4().hex[:12],
                            run_id=run_bundle.run_id,
                            agent_name=agent_name,
                            iteration=i,
                            hypothesis_rank=hyp.rank,
                            trial_id=hyp.diagnosis.trial_id,
                            diagnosis_family=hyp.diagnosis.family,
                            target_files=hyp.diagnosis.target_files,
                            outcome=iteration_hypothesis_outcome,
                            attempted=False,
                            confidence=hyp.confidence,
                            metadata={"skip_metrics": skip_metrics, **skip_metrics},
                        )
                    )

                    stop_result = stop_condition.record(
                        ScoreRecord(
                            iteration=i,
                            score=baseline_score,
                            applied=False,
                            delta=0.0,
                            mutation_outcome=iteration_hypothesis_outcome,
                        )
                    )
                    if stop_result.should_stop:
                        final_stop_reasons = stop_result.reasons
                        break

                batch = filtered_batch
                if not batch:
                    if final_stop_reasons:
                        break
                    pending_hypotheses = remainder
                    continue

                if console is not None and len(batch) > 1:
                    console.print(
                        f"  [dim]Step 3: speculative batch of {len(batch)} hypothesis/hypotheses ({len(pending_hypotheses)} remaining)[/dim]"
                    )

                def _cluster_for_hypothesis(hypothesis: Any) -> FailureCluster:
                    cluster_id = hypothesis.diagnosis.cluster_id
                    if isinstance(cluster_id, str) and cluster_id in cluster_by_id:
                        return cluster_by_id[cluster_id]
                    return FailureCluster(
                        cluster_id=cluster_id or hypothesis.diagnosis.trial_id,
                        representative=hypothesis.diagnosis,
                        diagnoses=[hypothesis.diagnosis],
                        trial_ids=[hypothesis.diagnosis.trial_id],
                        target_files=hypothesis.diagnosis.target_files,
                        family=hypothesis.diagnosis.family,
                    )

                attempt_results: list[Any] = await asyncio.gather(
                    *[
                        _speculative_attempt(
                            hypothesis=hyp,
                            iteration=i,
                            agent_name=agent_name,
                            agent_path=agent_path,
                            agent_content=agent_content,
                            suite_paths=suite_paths,
                            targeted_paths=_cluster_targeted_paths(
                                _cluster_for_hypothesis(hyp), failure_by_id, suite_paths
                            ),
                            failure_by_id=failure_by_id,
                            iteration_timeout_seconds=iteration_timeout_seconds,
                            eval_repeats=eval_repeats,
                            console=console,
                            run_bundle=run_bundle,
                        )
                        for hyp in batch
                    ],
                    return_exceptions=True,
                )

                speculative_candidates: list[dict[str, Any]] = []
                for hyp, attempt in zip(batch, attempt_results):
                    if isinstance(attempt, Exception):
                        iteration_hypothesis_outcome = "hypothesis_exception"
                        logger.warning(
                            "Hypothesis %s failed", hyp.diagnosis.trial_id, exc_info=attempt
                        )
                        if console is not None:
                            console.print(
                                f"    [bold red]✗ Hypothesis rank={hyp.rank} failed with exception[/bold red]"
                            )
                        memory_store.append_episode(
                            EpisodeRecord(
                                episode_id=uuid.uuid4().hex[:12],
                                run_id=run_bundle.run_id,
                                agent_name=agent_name,
                                iteration=i,
                                hypothesis_rank=hyp.rank,
                                trial_id=hyp.diagnosis.trial_id,
                                diagnosis_family=hyp.diagnosis.family,
                                target_files=hyp.diagnosis.target_files,
                                outcome="hypothesis_exception",
                                attempted=False,
                                confidence=hyp.confidence,
                                metadata={"error": str(attempt)},
                            )
                        )
                        continue

                    attempt_result = attempt
                    iteration_hypothesis_attempted = hyp.diagnosis.trial_id
                    patches.append(_serialize_patch(attempt_result["patch"])) if attempt_result.get(
                        "patch"
                    ) else None
                    outcome = attempt_result.get("outcome")
                    if outcome != "speculative_ready":
                        retry_count = int(attempt_result.get("retry_count", 0) or 0)
                        changed_paths = list(attempt_result.get("changed_paths", []))
                        mutation_llm_calls = int(attempt_result.get("mutation_llm_calls", 0) or 0)
                        if _is_retry_eligible(
                            str(outcome),
                            changed_paths=changed_paths,
                            mutation_llm_calls=mutation_llm_calls,
                            retry_count=retry_count,
                        ):
                            retry_timeout = _retry_timeout_seconds(iteration_timeout_seconds)
                            if console is not None:
                                console.print(
                                    f"    [yellow]↻ Retrying transient outcome '{outcome}' for rank={hyp.rank} with timeout {retry_timeout:.1f}s[/yellow]"
                                )
                            run_bundle.append_event(
                                "improve.retry_transient_outcome",
                                iteration=i,
                                hypothesis_rank=hyp.rank,
                                trial_id=hyp.diagnosis.trial_id,
                                outcome=str(outcome),
                                retry_count=retry_count + 1,
                                retry_timeout=retry_timeout,
                                changed_paths=changed_paths,
                                mutation_llm_calls=mutation_llm_calls,
                            )
                            retry_result = await _speculative_attempt(
                                hypothesis=hyp,
                                iteration=i,
                                agent_name=agent_name,
                                agent_path=agent_path,
                                agent_content=agent_content,
                                suite_paths=suite_paths,
                                targeted_paths=list(
                                    attempt_result.get("targeted_paths", suite_paths)
                                ),
                                failure_by_id=failure_by_id,
                                iteration_timeout_seconds=iteration_timeout_seconds,
                                eval_repeats=eval_repeats,
                                retry_count=retry_count + 1,
                                mutation_timeout_seconds_override=retry_timeout,
                                console=console,
                                run_bundle=run_bundle,
                            )
                            if retry_result.get("patch") is not None:
                                patches.append(_serialize_patch(retry_result["patch"]))
                            outcome = retry_result.get("outcome")
                            if outcome == "speculative_ready":
                                attempt_result = retry_result
                                run_bundle.append_event(
                                    "improve.retry_success",
                                    iteration=i,
                                    hypothesis_rank=hyp.rank,
                                    trial_id=hyp.diagnosis.trial_id,
                                    retry_count=retry_count + 1,
                                )
                            else:
                                run_bundle.append_event(
                                    "improve.retry_failed",
                                    iteration=i,
                                    hypothesis_rank=hyp.rank,
                                    trial_id=hyp.diagnosis.trial_id,
                                    retry_count=retry_count + 1,
                                    outcome=str(outcome),
                                )

                        if outcome != "speculative_ready":
                            iteration_hypothesis_outcome = str(outcome)
                            if console is not None and outcome == "no_file_changes":
                                console.print(
                                    f"    [yellow]↷ No-op mutation for rank={hyp.rank}; skipping[/yellow]"
                                )
                            stop_result = stop_condition.record(
                                ScoreRecord(
                                    iteration=i,
                                    score=baseline_score,
                                    applied=False,
                                    delta=0.0,
                                    mutation_outcome=str(outcome),
                                )
                            )
                            if stop_result.should_stop:
                                final_stop_reasons = stop_result.reasons
                                break
                            memory_store.append_episode(
                                EpisodeRecord(
                                    episode_id=uuid.uuid4().hex[:12],
                                    run_id=run_bundle.run_id,
                                    agent_name=agent_name,
                                    iteration=i,
                                    hypothesis_rank=hyp.rank,
                                    trial_id=hyp.diagnosis.trial_id,
                                    diagnosis_family=hyp.diagnosis.family,
                                    target_files=hyp.diagnosis.target_files,
                                    outcome=str(outcome),
                                    attempted=False,
                                    confidence=hyp.confidence,
                                    mutation_wall_seconds=float(
                                        attempt_result.get("mutation_wall_seconds", 0.0) or 0.0
                                    ),
                                    mutation_llm_calls=int(
                                        attempt_result.get("mutation_llm_calls", 0) or 0
                                    ),
                                    retry_count=int(attempt_result.get("retry_count", 0) or 0),
                                    metadata={
                                        "semantic_penalty": hyp.semantic_penalty,
                                        "semantic_boost": hyp.semantic_boost,
                                        "calibration_factor": hyp.calibration_factor,
                                    },
                                )
                            )
                            continue

                    if outcome != "speculative_ready":
                        continue

                    targeted_paths = attempt_result["targeted_paths"]
                    fast_validation_stats = attempt_result["fast_validation_stats"]
                    execution_metrics = attempt_result.get("execution_metrics", {})
                    mutation_wall_seconds = float(
                        attempt_result.get("mutation_wall_seconds", 0.0) or 0.0
                    )
                    phase_durations["mutation_generation"] = (
                        phase_durations.get("mutation_generation", 0.0) + mutation_wall_seconds
                    )
                    fast_validation_wall_seconds = float(
                        attempt_result.get("fast_validation_wall_seconds", 0.0) or 0.0
                    )
                    phase_durations["fast_validation"] = (
                        phase_durations.get("fast_validation", 0.0) + fast_validation_wall_seconds
                    )
                    targeted_baseline_score = _mean_score_for_paths(last_summary, targeted_paths)
                    if targeted_baseline_score is None:
                        targeted_baseline_score = baseline_score
                    fast_delta = fast_validation_stats.selected_score - targeted_baseline_score
                    attempt_result["targeted_baseline_score"] = targeted_baseline_score
                    attempt_result["fast_delta"] = fast_delta
                    attempt_result["execution_metrics"] = execution_metrics
                    if fast_delta <= 0:
                        attempt_result["outcome"] = "targeted_regression"
                        iteration_hypothesis_outcome = "targeted_regression"
                        if console is not None:
                            console.print(
                                f"    [yellow]↷ Rank={hyp.rank} rejected after fast validation (targeted delta {fast_delta:+.4f})[/yellow]"
                            )
                        stop_result = stop_condition.record(
                            ScoreRecord(
                                iteration=i,
                                score=baseline_score,
                                applied=False,
                                delta=0.0,
                                mutation_outcome="targeted_regression",
                            )
                        )
                        if stop_result.should_stop:
                            final_stop_reasons = stop_result.reasons
                            break
                        memory_store.append_episode(
                            EpisodeRecord(
                                episode_id=uuid.uuid4().hex[:12],
                                run_id=run_bundle.run_id,
                                agent_name=agent_name,
                                iteration=i,
                                hypothesis_rank=hyp.rank,
                                trial_id=hyp.diagnosis.trial_id,
                                diagnosis_family=hyp.diagnosis.family,
                                target_files=hyp.diagnosis.target_files,
                                outcome="targeted_regression",
                                attempted=True,
                                kept=False,
                                confidence=hyp.confidence,
                                score_delta=fast_delta,
                                mutation_wall_seconds=mutation_wall_seconds,
                                mutation_llm_calls=int(
                                    attempt_result.get("mutation_llm_calls", 0) or 0
                                ),
                                metadata={
                                    "semantic_penalty": hyp.semantic_penalty,
                                    "semantic_boost": hyp.semantic_boost,
                                    "calibration_factor": hyp.calibration_factor,
                                },
                            )
                        )
                        continue
                    speculative_candidates.append(attempt_result)

                speculative_candidates.sort(
                    key=lambda item: item.get("fast_delta", 0.0), reverse=True
                )

                kept_in_batch = False
                kept_hypothesis = None
                for attempt_result in speculative_candidates:
                    hyp = attempt_result["hypothesis"]
                    workspace = attempt_result["workspace"]
                    hypothesis_mutator = attempt_result["hypothesis_mutator"]
                    targeted_paths = attempt_result["targeted_paths"]
                    changed_paths = attempt_result["changed_paths"]
                    execution_metrics = attempt_result.get("execution_metrics", {})
                    mutation_wall_seconds = float(
                        attempt_result.get("mutation_wall_seconds", 0.0) or 0.0
                    )
                    mutation_llm_calls = int(attempt_result.get("mutation_llm_calls", 0) or 0)
                    fast_validation_stats = attempt_result["fast_validation_stats"]
                    targeted_baseline_score = attempt_result["targeted_baseline_score"]
                    fast_delta = attempt_result["fast_delta"]

                    if console is not None:
                        console.print(
                            f"    [dim]Promoting speculative rank={hyp.rank} with targeted delta {fast_delta:+.4f}[/dim]"
                        )
                        console.print(
                            f"    [dim]Promoted hypothesis:[/dim] {_hypothesis_console_label(hyp)}"
                        )

                    evaluated_stats = fast_validation_stats
                    evaluation_summary = attempt_result["fast_validation_summary"]
                    targeted_coverage = len(targeted_paths) / max(1, len(suite_paths))
                    integrity_validation_repeats = integrity_repeats
                    needs_full_suite_integrity = fast_delta > score_threshold and (
                        integrity_repeats > eval_repeats or len(targeted_paths) != len(suite_paths)
                    )
                    if fast_delta > score_threshold * 5 and targeted_coverage >= 0.8:
                        needs_full_suite_integrity = False
                    elif fast_delta > score_threshold * 2 and targeted_coverage >= 0.5:
                        integrity_validation_repeats = max(eval_repeats, 1)
                    if needs_full_suite_integrity:
                        if console is not None:
                            console.print(
                                f"    [dim]Step 5: integrity validation ({integrity_validation_repeats} repeats)...[/dim]"
                            )
                        integrity_started_at = time.monotonic()
                        (
                            integrity_stats,
                            integrity_summary,
                            integrity_errors,
                        ) = await _run_eval_n_times(
                            suite_paths,
                            agent_name,
                            iteration_timeout_seconds,
                            workspace.workspace_agent_path,
                            integrity_validation_repeats,
                            console=console,
                            phase_label="integrity validation",
                            audit_bundle=run_bundle,
                            audit_stem=f"iter-{i:03d}/rank-{hyp.rank}/integrity-validation",
                        )
                        if integrity_stats is None:
                            logger.warning(
                                "Integrity pass failed for hypothesis rank=%d: %s",
                                hyp.rank,
                                integrity_errors,
                            )
                            if console is not None:
                                console.print(
                                    f"    [bold red]✗ Integrity pass failed for rank={hyp.rank}[/bold red]"
                                )
                            phase_durations["integrity_validation"] = phase_durations.get(
                                "integrity_validation", 0.0
                            ) + (time.monotonic() - integrity_started_at)
                            workspace.cleanup()
                            hypothesis_mutator.cleanup()
                            continue
                        phase_durations["integrity_validation"] = phase_durations.get(
                            "integrity_validation", 0.0
                        ) + (time.monotonic() - integrity_started_at)
                        evaluated_stats = integrity_stats
                        evaluation_summary = integrity_summary

                    evaluated_score = evaluated_stats.selected_score
                    delta = evaluated_score - baseline_score
                    regressions: list[dict[str, float | bool | str]] = []
                    if (
                        evaluation_summary is not None
                        and last_summary is not None
                        and len(targeted_paths) != len(suite_paths)
                    ):
                        regressions = _find_regressed_unaffected_paths(
                            last_summary,
                            evaluation_summary,
                            targeted_paths,
                            max_score_drop=score_threshold,
                        )
                    acceptance = _acceptance_decision(
                        delta=delta,
                        regressions=regressions,
                        score_threshold=score_threshold,
                    )
                    kept = acceptance.kept
                    iteration_hypothesis_outcome = "kept" if kept else "reverted"

                    if console is not None:
                        if kept:
                            console.print(
                                f"    [green]✓ KEPT[/green]  delta=[green]{delta:+.4f}[/green]  ({baseline_score:.4f} → {evaluated_score:.4f}){_format_repeat_aggregate(evaluated_stats)}"
                            )
                        else:
                            console.print(
                                f"    [red]✗ REVERTED[/red]  delta=[red]{delta:+.4f}[/red]  ({baseline_score:.4f} → {evaluated_score:.4f}){_format_repeat_aggregate(evaluated_stats)}"
                            )
                            if (
                                acceptance.intolerable_regressions
                                or acceptance.tolerable_regressions
                            ):
                                console.print(
                                    f"    [yellow]Per-eval regression gate:[/yellow] {_format_path_list([str(r['scenario_path']) for r in acceptance.intolerable_regressions + acceptance.tolerable_regressions], limit=3)}"
                                )
                        if kept and acceptance.tolerable_regressions:
                            console.print(
                                f"    [yellow]Kept with tolerated regressions:[/yellow] {_format_path_list([str(r['scenario_path']) for r in acceptance.tolerable_regressions], limit=3)}"
                            )

                    current_review = review_by_trial.get(hyp.diagnosis.trial_id)
                    lesson = Lesson(
                        lesson_id=uuid.uuid4().hex[:12],
                        trial_id=hyp.diagnosis.trial_id,
                        hypothesis_summary=hyp.diagnosis.failure_summary,
                        root_cause=hyp.diagnosis.root_cause,
                        target_files=hyp.diagnosis.target_files,
                        outcome="kept" if kept else "reverted",
                        score_before=baseline_score,
                        score_after=evaluated_score,
                        score_delta=delta,
                        iteration=i,
                        agent_path=str(agent_path) if agent_path else None,
                        metadata={
                            "improve_run_id": run_bundle.run_id,
                            "hypothesis_rank": hyp.rank,
                            "diagnosis_family": hyp.diagnosis.family,
                            "scenario_family": (
                                current_review.task_type if current_review is not None else None
                            ),
                            "failure_bucket": (
                                current_review.failure_bucket
                                if current_review is not None
                                else None
                            ),
                            "observed_pattern": (
                                current_review.reasons[0]
                                if current_review and current_review.reasons
                                else None
                            ),
                            "causality_kind": _causality_kind(
                                hyp.diagnosis.target_files, changed_paths
                            ),
                            "before_pass_rate": final_pass_rate,
                            "after_pass_rate": evaluated_stats.selected_pass_rate,
                            "nearby_regression": bool(
                                acceptance.tolerable_regressions
                                or acceptance.intolerable_regressions
                            ),
                            "net_benefit": acceptance.net_benefit,
                            "regression_cost": acceptance.regression_cost,
                            "tolerated_regressions": acceptance.tolerable_regressions,
                            "intolerable_regressions": acceptance.intolerable_regressions,
                            "execution_metrics": execution_metrics,
                        },
                    )
                    lesson_store.save(lesson)
                    if console is not None:
                        console.print(
                            f"    [dim]Lesson saved: {lesson.lesson_id} ({'kept' if kept else 'reverted'} Δ={delta:+.4f})[/dim]"
                        )
                        console.print(
                            f"    [dim]Mutation cost: {mutation_wall_seconds:.1f}s wall-clock, {mutation_llm_calls} LLM call(s)[/dim]"
                        )

                    mutation_history.append(
                        {
                            "iteration": i,
                            "hypothesis_rank": hyp.rank,
                            "trial_id": hyp.diagnosis.trial_id,
                            "cluster_id": hyp.diagnosis.cluster_id,
                            "diagnosis_family": hyp.diagnosis.family,
                            "targeted_paths": targeted_paths,
                            "targeted_baseline_score": targeted_baseline_score,
                            "baseline_score_before": baseline_score,
                            "selected_score_after": evaluated_score,
                            "fast_selected_score_after": fast_validation_stats.selected_score,
                            "mean_score_after": evaluated_stats.mean_score,
                            "regressed_paths": regressions,
                            "tolerated_regressions": acceptance.tolerable_regressions,
                            "intolerable_regressions": acceptance.intolerable_regressions,
                            "net_benefit": acceptance.net_benefit,
                            "regression_cost": acceptance.regression_cost,
                            "median_score_after": evaluated_stats.median_score,
                            "min_score_after": evaluated_stats.min_score,
                            "max_score_after": evaluated_stats.max_score,
                            "applied_files": changed_paths,
                            "causality_kind": _causality_kind(
                                hyp.diagnosis.target_files, changed_paths
                            ),
                            "rejection_reason": None if kept else acceptance.rejection_reason,
                            "execution_metrics": execution_metrics,
                            "phase1_review": (
                                current_review.model_dump(mode="json")
                                if current_review is not None
                                else None
                            ),
                            "selected_pass_rate_after": evaluated_stats.selected_pass_rate,
                            "baseline_pass_rate_before": final_pass_rate,
                            "improvement": delta,
                            "kept": kept,
                            "lesson_id": lesson.lesson_id,
                            "mutation_wall_seconds": mutation_wall_seconds,
                            "mutation_llm_calls": mutation_llm_calls,
                        }
                    )
                    memory_store.append_episode(
                        EpisodeRecord(
                            episode_id=uuid.uuid4().hex[:12],
                            run_id=run_bundle.run_id,
                            agent_name=agent_name,
                            iteration=i,
                            hypothesis_rank=hyp.rank,
                            trial_id=hyp.diagnosis.trial_id,
                            diagnosis_family=hyp.diagnosis.family,
                            target_files=hyp.diagnosis.target_files,
                            outcome="kept" if kept else "reverted",
                            attempted=True,
                            kept=kept,
                            confidence=hyp.confidence,
                            score_delta=delta,
                            mutation_wall_seconds=mutation_wall_seconds,
                            mutation_llm_calls=mutation_llm_calls,
                            metadata={
                                "rejection_reason": acceptance.rejection_reason,
                                "net_benefit": acceptance.net_benefit,
                                "regression_cost": acceptance.regression_cost,
                                "semantic_penalty": hyp.semantic_penalty,
                                "semantic_boost": hyp.semantic_boost,
                                "calibration_factor": hyp.calibration_factor,
                            },
                        )
                    )
                    run_bundle.write_json(
                        f"hypotheses/iter-{i:03d}/rank-{hyp.rank}.json", mutation_history[-1]
                    )

                    stop_result = stop_condition.record(
                        ScoreRecord(
                            iteration=i,
                            score=evaluated_score if kept else baseline_score,
                            applied=kept,
                            delta=delta,
                            mutation_outcome=(
                                "applied_with_tolerated_regressions"
                                if kept and acceptance.tolerable_regressions
                                else "applied"
                                if kept
                                else acceptance.rejection_reason or "below_keep_threshold"
                            ),
                        )
                    )

                    iteration_hypothesis_tested = hyp.diagnosis.trial_id
                    iteration_hypothesis_score = evaluated_score
                    iteration_delta = delta
                    iteration_lesson_id = lesson.lesson_id
                    iteration_kept = kept
                    iteration_mutation_wall_seconds = mutation_wall_seconds
                    iteration_mutation_llm_calls = mutation_llm_calls

                    if kept:
                        kept_hypothesis = hyp
                        final_score = evaluated_score
                        final_pass_rate = evaluated_stats.selected_pass_rate
                        latest_phase1_reviews = _phase1_reviews(evaluation_summary)
                        synced_paths = workspace.sync_back()
                        applied_files.update(synced_paths)
                        commit_title, commit_body = _build_improve_commit_message(
                            run_id=run_bundle.run_id,
                            lesson=lesson,
                            hypothesis_rank=hyp.rank,
                            changed_paths=synced_paths,
                        )
                        try:
                            repo_commit = commit_agent_changes(
                                workspace.repo_root,
                                workspace.original_agent_path,
                                synced_paths,
                                message_title=commit_title,
                                message_body=commit_body,
                            )
                        except ValueError as exc:
                            run_bundle.append_event(
                                "improve.commit_failed",
                                iteration=i,
                                hypothesis_rank=hyp.rank,
                                trial_id=hyp.diagnosis.trial_id,
                                lesson_id=lesson.lesson_id,
                                error=str(exc),
                            )
                            workspace.cleanup()
                            hypothesis_mutator.cleanup()
                            raise

                        lesson.metadata["git_commit"] = repo_commit.commit_sha
                        lesson.metadata["git_branch"] = repo_commit.branch
                        lesson_store.save(lesson)
                        mutation_history[-1]["git_commit"] = repo_commit.commit_sha
                        mutation_history[-1]["git_branch"] = repo_commit.branch
                        mutation_history[-1]["git_commit_paths"] = repo_commit.committed_paths
                        mutation_history[-1]["git_commit_title"] = repo_commit.message_title
                        run_bundle.write_json(
                            f"hypotheses/iter-{i:03d}/rank-{hyp.rank}.json", mutation_history[-1]
                        )
                        run_bundle.write_json(
                            f"commits/iter-{i:03d}/rank-{hyp.rank}.json",
                            {
                                "iteration": i,
                                "hypothesis_rank": hyp.rank,
                                "trial_id": hyp.diagnosis.trial_id,
                                "lesson_id": lesson.lesson_id,
                                "commit_sha": repo_commit.commit_sha,
                                "branch": repo_commit.branch,
                                "committed_paths": repo_commit.committed_paths,
                                "message_title": repo_commit.message_title,
                                "message_body": repo_commit.message_body,
                            },
                        )
                        run_bundle.append_event(
                            "improve.commit_created",
                            iteration=i,
                            hypothesis_rank=hyp.rank,
                            trial_id=hyp.diagnosis.trial_id,
                            lesson_id=lesson.lesson_id,
                            commit_sha=repo_commit.commit_sha,
                            branch=repo_commit.branch,
                            committed_paths=repo_commit.committed_paths,
                        )
                        if original_mutator is not None:
                            agent_content = original_mutator.scan()
                            allowed_target_files = set(agent_content.keys())
                        baseline_score = evaluated_score
                        resolved_families.add(hyp.diagnosis.family)
                        resolved_target_files.update(hyp.diagnosis.target_files)
                        kept_in_batch = True
                        if console is not None:
                            console.print(
                                f"    [green]Step 6: synced kept mutation back to agent[/green]  {_format_path_list(synced_paths)}"
                            )
                            console.print(
                                f"    [dim]Committed kept mutation: {repo_commit.commit_sha[:12]} on {repo_commit.branch or 'detached HEAD'}[/dim]"
                            )
                            console.print(
                                f"    [dim]Kept patch raised the selected score to {evaluated_score:.2%}; restarting speculative batches from the updated baseline.[/dim]"
                            )
                        workspace.cleanup()
                        hypothesis_mutator.cleanup()
                        break
                    else:
                        if console is not None:
                            console.print(
                                "    [yellow]Discarding hypothesis worktree and trying next candidate[/yellow]"
                            )
                        if acceptance.intolerable_regressions or acceptance.tolerable_regressions:
                            regressed_families.add(hyp.diagnosis.family)
                            regressed_target_files.update(hyp.diagnosis.target_files)
                        workspace.cleanup()
                        hypothesis_mutator.cleanup()

                    if stop_result.should_stop:
                        logger.warning("Stop condition: %s", "; ".join(stop_result.reasons))
                        if console is not None:
                            console.print(
                                f"  [bold yellow]⚠ Stop: {stop_result.reasons[0]}[/bold yellow]"
                            )
                        final_stop_reasons = stop_result.reasons
                        kept_in_batch = True
                        break

                for attempt_result in speculative_candidates:
                    workspace = attempt_result.get("workspace")
                    hypothesis_mutator = attempt_result.get("hypothesis_mutator")
                    if workspace is not None and hypothesis_mutator is not None:
                        if workspace.workspace_root.exists():
                            workspace.cleanup()
                            hypothesis_mutator.cleanup()

                if final_stop_reasons:
                    break
                if kept_in_batch and kept_hypothesis is not None:
                    pending_hypotheses = [
                        hyp for hyp in batch if hyp != kept_hypothesis
                    ] + remainder
                    pending_hypotheses = _rerank_pending_hypotheses(
                        ranker,
                        pending_hypotheses,
                        resolved_families=resolved_families,
                        resolved_target_files=resolved_target_files,
                        regressed_families=regressed_families,
                        regressed_target_files=regressed_target_files,
                        allowed_target_files=allowed_target_files,
                    )
                else:
                    pending_hypotheses = _rerank_pending_hypotheses(
                        ranker,
                        remainder,
                        resolved_families=resolved_families,
                        resolved_target_files=resolved_target_files,
                        regressed_families=regressed_families,
                        regressed_target_files=regressed_target_files,
                        allowed_target_files=allowed_target_files,
                    )

            iter_log = IterationLog(
                iteration=i,
                baseline_score=baseline_stats.selected_score,
                baseline_repeats=eval_repeats,
                failures=[t.id for t in failures],
                diagnoses=[diagnosis_to_summary(d) for d in diagnoses],
                suspicious_trials=[review.trial_id for review in suspicious_reviews],
                failure_buckets=failure_bucket_by_trial,
                hypothesis_ranked=ranking.ranked_count,
                hypothesis_attempted=iteration_hypothesis_attempted,
                hypothesis_outcome=iteration_hypothesis_outcome,
                hypothesis_tested=iteration_hypothesis_tested,
                hypothesis_score=iteration_hypothesis_score,
                delta=iteration_delta,
                kept=iteration_kept,
                lesson_id=iteration_lesson_id,
                stop_reasons=final_stop_reasons,
                phase_durations=phase_durations,
                mutation_wall_seconds=iteration_mutation_wall_seconds,
                mutation_llm_calls=iteration_mutation_llm_calls,
            )
            iteration_logs.append(iter_log.model_dump())
            write_iteration_log(iter_log, iteration_output_dir)
            run_bundle.write_json(f"iteration_logs/iter-{i:03d}.json", iter_log.model_dump())
            memory_store.save_working(
                WorkingSnapshot(
                    run_id=run_bundle.run_id,
                    iteration=i,
                    active_trial_ids=[iteration_hypothesis_attempted]
                    if iteration_hypothesis_attempted
                    else [],
                    active_families=(
                        [kept_hypothesis.diagnosis.family] if kept_hypothesis is not None else []
                    ),
                    active_target_files=(
                        kept_hypothesis.diagnosis.target_files
                        if kept_hypothesis is not None
                        else []
                    ),
                    hypothesis_count=len(pending_hypotheses),
                    baseline_score=baseline_stats.selected_score,
                    baseline_pass_rate=baseline_stats.selected_pass_rate,
                    last_hypothesis_outcome=iteration_hypothesis_outcome,
                    memory_skip_count=memory_skip_count,
                    stop_reasons=final_stop_reasons,
                )
            )
            get_telemetry().emit(
                "improve.iteration_timing",
                iteration=i,
                phase_durations=phase_durations,
            )

            if console is not None:
                console.print(
                    f"  [dim]Outer pass {i + 1} complete: "
                    f"score {final_score:.2%}  "
                    f"tested={len(mutation_history)}  kept={sum(1 for m in mutation_history if m.get('kept'))}[/dim]"
                )
                if phase_durations:
                    phase_parts = [
                        f"{name}={seconds:.1f}s" for name, seconds in phase_durations.items()
                    ]
                    console.print(f"  [dim]Phase timing: {' '.join(phase_parts)}[/dim]")
                console.print()

            if final_stop_reasons:
                break

    finally:
        if original_mutator is not None:
            original_mutator.cleanup()

    if not final_stop_reasons:
        if actual_iterations >= max_iterations:
            final_stop_reasons = ["max_iterations_reached"]
        else:
            final_stop_reasons = ["completed_without_explicit_stop"]

    score_delta = final_score - initial_score
    pass_rate_delta = final_pass_rate - initial_pass_rate
    has_divergence = (score_delta * pass_rate_delta) < -0.001
    memory_consolidation = memory_store.consolidate_run(run_bundle.run_id)
    run_episodes = memory_store.load_episodes(run_id=run_bundle.run_id)
    semantic_rules = memory_store.load_semantic_rules()
    personal_preferences = memory_store.load_personal_preferences()
    observability = memory_store.observability_metrics()
    memory_summary = {
        "memory_dir": str(memory_store.base_dir),
        "episodes_this_run": len(run_episodes),
        "semantic_rule_count": len(semantic_rules),
        "personal_preference_count": len(personal_preferences),
        "memory_skips": memory_skip_count,
        "backfilled_episodes": backfilled_episodes,
        "observability": observability,
        "consolidation": memory_consolidation.model_dump(mode="json"),
    }

    if console is not None:
        console.rule("[bold]Improve Summary[/bold]")
        console.print(
            f"  [bold]Score:[/bold]     {initial_score:.2%} → "
            f"{final_score:.2%}  "
            f"[bold]Δ={score_delta:+.2%}[/bold]"
        )
        console.print(
            f"  [bold]Pass rate:[/bold] {initial_pass_rate:.2%} → "
            f"{final_pass_rate:.2%}  "
            f"[bold]Δ={pass_rate_delta:+.2%}[/bold]"
        )
        if has_divergence:
            direction = (
                "score up / pass-rate down" if score_delta > 0 else "score down / pass-rate up"
            )
            console.print(
                "  [bold yellow]⚠ Divergence:[/bold yellow] "
                f"{direction} (score Δ{score_delta:+.2%}, pass Δ{pass_rate_delta:+.2%})"
            )
        console.print(
            f"  [bold]Patches:[/bold]   {len(patches)} proposed  {len(mutation_history)} tested"
        )
        kept_count = sum(1 for m in mutation_history if m.get("kept"))
        console.print(
            f"  [bold]Results:[/bold]   [green]{kept_count} kept[/green]  "
            f"[red]{len(mutation_history) - kept_count} reverted[/red]"
        )
        if convergence_achieved:
            console.print("  [bold]Status:[/bold]    [green]✓ Converged[/green]")
        if final_stop_reasons:
            console.print(f"  [bold]Stopped:[/bold]   [yellow]{final_stop_reasons[0]}[/yellow]")

    phase2_metrics = _compute_phase2_metrics(
        initial_score=initial_score,
        final_score=final_score,
        initial_pass_rate=initial_pass_rate,
        final_pass_rate=final_pass_rate,
        mutation_history=mutation_history,
        iteration_logs=iteration_logs,
        phase1_reviews=[review.model_dump(mode="json") for review in latest_phase1_reviews],
        stop_reasons=final_stop_reasons,
    )
    phase2_gate = _build_phase2_gate_report(
        improve_runs_root=run_bundle.path.parent,
        current_config=run_config,
        current_run_id=run_bundle.run_id,
        current_initial_score=initial_score,
        current_final_score=final_score,
        current_phase2_metrics=phase2_metrics,
    )

    result = ImprovementResult(
        iterations=actual_iterations,
        initial_score=initial_score,
        final_score=final_score,
        initial_pass_rate=initial_pass_rate,
        final_pass_rate=final_pass_rate,
        patches_proposed=patches,
        patches_applied=sorted(applied_files),
        trace_path=trace_dir,
        mutation_history=mutation_history,
        iteration_logs=iteration_logs,
        phase1_reviews=[review.model_dump(mode="json") for review in latest_phase1_reviews],
        phase2_metrics=phase2_metrics,
        phase2_gate=phase2_gate,
        memory_summary=memory_summary,
        convergence_achieved=convergence_achieved,
        stop_reasons=final_stop_reasons,
    )

    run_bundle.write_json("run.json", result.model_dump(mode="json"))
    run_bundle.write_json(
        "stop_history.json",
        [record.model_dump() for record in stop_condition.history],
    )
    run_bundle.write_json("phase2_gate.json", phase2_gate)
    run_bundle.write_json("memory_summary.json", memory_summary)
    phase_totals: dict[str, float] = {}
    for iteration_log in iteration_logs:
        raw_phase_durations = iteration_log.get("phase_durations", {})
        if not isinstance(raw_phase_durations, dict):
            continue
        for phase_name, phase_seconds in raw_phase_durations.items():
            if not isinstance(phase_name, str) or not isinstance(phase_seconds, int | float):
                continue
            phase_totals[phase_name] = phase_totals.get(phase_name, 0.0) + float(phase_seconds)

    mutation_wall_seconds_values = [
        float(entry["mutation_wall_seconds"])
        for entry in mutation_history
        if isinstance(entry.get("mutation_wall_seconds"), int | float)
    ]
    mutation_llm_calls_values = [
        int(entry["mutation_llm_calls"])
        for entry in mutation_history
        if isinstance(entry.get("mutation_llm_calls"), int | float)
    ]
    total_mutation_wall_seconds = sum(mutation_wall_seconds_values)
    total_mutation_llm_calls = sum(mutation_llm_calls_values)
    avg_mutation_wall_seconds = (
        total_mutation_wall_seconds / len(mutation_wall_seconds_values)
        if mutation_wall_seconds_values
        else 0.0
    )
    avg_mutation_llm_calls = (
        total_mutation_llm_calls / len(mutation_llm_calls_values)
        if mutation_llm_calls_values
        else 0.0
    )
    total_phase_seconds = sum(phase_totals.values())
    avg_phase_seconds_per_iteration = (
        total_phase_seconds / max(1, len(iteration_logs)) if phase_totals else 0.0
    )
    phase_timing_summary = _format_phase_durations_summary(phase_totals) if phase_totals else "none"
    overall_wall_seconds = time.monotonic() - run_started_at
    gate_status = str(phase2_gate.get("status", "unknown"))
    gate_failures = list(phase2_gate.get("failed_criteria", []))
    gate_failures_str = "; ".join(str(item) for item in gate_failures) if gate_failures else "none"
    funnel = phase2_metrics.get("funnel", {})
    progress_ledger = phase2_metrics.get("progress_ledger", {})
    gate_comparability = phase2_gate.get("cohort_comparability", {})

    run_bundle.write_summary(
        "\n".join(
            [
                f"# Improve Run {run_bundle.run_id}",
                f"- Suite: `{suite_path}`",
                f"- Agent: `{agent_name}`",
                f"- Agent path: `{agent_path}`" if agent_path else "- Agent path: none",
                f"- Score: {initial_score:.2%} -> {final_score:.2%} (Δ {score_delta:+.2%})",
                f"- Pass rate: {initial_pass_rate:.2%} -> {final_pass_rate:.2%} (Δ {pass_rate_delta:+.2%})",
                (
                    "- ⚠ Divergence: "
                    f"score Δ{score_delta:+.2%} vs pass-rate Δ{pass_rate_delta:+.2%}"
                    if has_divergence
                    else "- ⚠ Divergence: none"
                ),
                f"- Phase 1 suspicious reviews: {sum(1 for review in latest_phase1_reviews if review.suspicious)}",
                f"- Iterations: {actual_iterations}",
                f"- Patches proposed: {len(patches)}",
                f"- Hypotheses tested: {len(mutation_history)}",
                (
                    "- Mutation wall-clock (tested): "
                    f"total {_format_seconds(total_mutation_wall_seconds)} "
                    f"| avg {_format_seconds(avg_mutation_wall_seconds)}"
                ),
                (
                    "- Mutation LLM calls (tested): "
                    f"total {total_mutation_llm_calls} "
                    f"| avg {avg_mutation_llm_calls:.1f}"
                ),
                (f"- Phase timing (all iterations): {phase_timing_summary}"),
                (
                    "- Phase wall-clock total: "
                    f"{_format_seconds(total_phase_seconds)} "
                    f"| avg/iteration {_format_seconds(avg_phase_seconds_per_iteration)}"
                ),
                (
                    "- Overall wall-clock: "
                    f"{_format_seconds(overall_wall_seconds)}"
                    + (
                        f" (budget {overall_timeout_seconds:.1f}s)"
                        if overall_timeout_seconds is not None
                        else ""
                    )
                ),
                (
                    "- Keep rate: "
                    f"{phase2_metrics['keep_rate']:.2%} ({phase2_metrics['kept_count']}/{phase2_metrics['tested_count']})"
                ),
                (
                    "- Timeout rate: "
                    f"{phase2_metrics['timeout_rate']:.2%} ({phase2_metrics['timeout_count']}/{phase2_metrics['tested_count']})"
                ),
                (
                    "- Net benefit: "
                    f"total {phase2_metrics['net_benefit_total']:+.4f} "
                    f"| per-tested {phase2_metrics['net_benefit_per_tested']:+.4f}"
                ),
                (f"- Memory episodes (this run): {memory_summary['episodes_this_run']}"),
                (f"- Memory semantic rules: {memory_summary['semantic_rule_count']}"),
                (f"- Personal memory preferences: {memory_summary['personal_preference_count']}"),
                (f"- Memory conversion skips: {memory_summary['memory_skips']}"),
                (f"- Memory backfilled episodes: {memory_summary['backfilled_episodes']}"),
                (
                    "- Memory observability: "
                    f"skip_precision {observability['skip_precision']:.2%}, "
                    f"promotion_hit_rate {observability['promotion_hit_rate']:.2%}, "
                    f"semantic_rule_utilization {observability['semantic_rule_utilization_rate']:.2%}"
                ),
                (
                    "- Mutation funnel: "
                    f"generated {funnel.get('generated', 0)} -> ranked {funnel.get('ranked', 0)} "
                    f"-> attempted {funnel.get('attempted', 0)} -> kept {funnel.get('kept', 0)}"
                ),
                (
                    "- Progress ledger: "
                    f"claims_quality {progress_ledger.get('claims_quality', 0)}, "
                    f"verification_behavior {progress_ledger.get('verification_behavior', 0)}, "
                    f"capability_signal {progress_ledger.get('capability_signal', 0)}"
                ),
                f"- Phase 2 gate (latest comparable window): {gate_status}",
                (
                    "- Phase 2 cohort comparability: "
                    f"cohort {gate_comparability.get('cohort_size', 0)}/"
                    f"{gate_comparability.get('total_runs', 0)} total runs "
                    f"(comparable={gate_comparability.get('comparable', False)})"
                ),
                f"- Phase 2 gate failures: {gate_failures_str}",
                f"- Stop reasons: {', '.join(final_stop_reasons) if final_stop_reasons else 'none'}",
            ]
        )
    )
    run_bundle.append_event(
        "improve.run_finished",
        score_before=initial_score,
        score_after=final_score,
        iterations=actual_iterations,
        stop_reasons=final_stop_reasons,
        phase2_gate_status=gate_status,
        phase2_keep_rate=phase2_metrics["keep_rate"],
        phase2_timeout_rate=phase2_metrics["timeout_rate"],
        memory_skip_precision=observability["skip_precision"],
        memory_promotion_hit_rate=observability["promotion_hit_rate"],
    )

    return result


def backfill_memory(
    improve_runs_root: Path,
    memory_dir: Path | None = None,
    *,
    force: bool = False,
    include_improve_cycle: bool = False,
) -> dict[str, int]:
    memory_store = MemoryStore(base_dir=memory_dir)
    imported = memory_store.backfill_from_run_artifacts(
        improve_runs_root,
        force=force,
        include_improve_cycle=include_improve_cycle,
    )
    observability = memory_store.observability_metrics()
    return {
        "imported_episodes": imported,
        "semantic_rules": len(memory_store.load_semantic_rules()),
        "skip_precision_basis": int(observability["skip_precision"] * 10000),
    }


async def _run_eval_n_times(
    suite_path: str | list[str],
    agent_name: str,
    timeout: float,
    agent_path: Path | None,
    n: int,
    console: Console | None = None,
    phase_label: str = "evaluation",
    audit_bundle: Any | None = None,
    audit_stem: str | None = None,
) -> tuple[EvalRepeatAggregate | None, EvalRunSummary | None, list[str]]:
    suite_paths = _normalize_suite_paths(suite_path)
    scores: list[float] = []
    pass_rates: list[float] = []
    errors: list[str] = []
    last_summary: EvalRunSummary | None = None

    for repeat_idx in range(n):
        if console is not None and n > 1:
            console.print(
                f"    [dim]{phase_label.capitalize()} repeat {repeat_idx + 1}/{n}...[/dim]"
            )
        try:
            if console is None:
                summary = await _run_eval(suite_paths, agent_name, timeout, agent_path)
            else:
                summary = await _run_eval(
                    suite_paths,
                    agent_name,
                    timeout,
                    agent_path,
                    console=console,
                    phase_label=phase_label,
                )
            scores.append(summary.metrics.mean_score)
            pass_rates.append(summary.metrics.pass_rate)
            last_summary = summary
            if console is not None and n > 1:
                console.print(
                    f"    [dim]{phase_label.capitalize()} repeat {repeat_idx + 1}/{n} "
                    f"score {summary.metrics.mean_score:.2%} pass {summary.metrics.pass_rate:.2%}[/dim]"
                )
            get_telemetry().emit(
                "improve.eval_repeat",
                repeat=repeat_idx,
                suite=suite_paths,
                trial_count=len(summary.trials),
                pass_rate=round(summary.metrics.pass_rate, 4),
                mean_score=round(summary.metrics.mean_score, 4),
                completed=summary.metrics.completed_tasks,
            )
            if audit_bundle is not None and audit_stem is not None:
                audit_bundle.write_json(
                    f"validations/{audit_stem}/repeat-{repeat_idx + 1}.json",
                    {"summary": summary.model_dump(mode="json"), "error": None},
                )
        except Exception as exc:
            logger.warning("Eval run failed (repeat %d)", repeat_idx, exc_info=True)
            errors.append(str(exc))
            if console is not None:
                console.print(
                    f"    [yellow]{phase_label.capitalize()} repeat {repeat_idx + 1}/{n} failed:[/yellow] {exc}"
                )
            if audit_bundle is not None and audit_stem is not None:
                audit_bundle.write_json(
                    f"validations/{audit_stem}/repeat-{repeat_idx + 1}.json",
                    {"summary": None, "error": str(exc)},
                )

    if not scores:
        return None, None, errors

    aggregate = _summarize_repeat_scores(scores, pass_rates)
    if console is not None and aggregate.repeat_count > 1:
        console.print(
            f"    [dim]{phase_label.capitalize()} aggregate uses median for selection: "
            f"{aggregate.selected_score:.2%} "
            f"(mean {aggregate.mean_score:.2%}, range {aggregate.min_score:.2%}-{aggregate.max_score:.2%}; "
            f"pass mean {aggregate.mean_pass_rate:.2%})[/dim]"
        )

    return aggregate, last_summary, errors


async def _run_eval(
    suite_path: str | list[str],
    agent_name: str,
    timeout: float,
    agent_path: Path | None = None,
    console: Console | None = None,
    phase_label: str = "evaluation",
) -> EvalRunSummary:
    from ash_hawk.scenario.runner import run_scenarios_async

    suite_paths = _normalize_suite_paths(suite_path)
    if console is None:
        return await run_scenarios_async(
            suite_paths,
            agent_path=agent_path,
            adapter_override=_resolve_adapter_override(agent_name),
        )

    labels = _eval_progress_labels(suite_paths)
    task_rows: dict[str, TaskID] = {}

    async def on_trial_progress(
        completed_count: int,
        remaining_count: int,
        total_count: int,
        task_id: str,
    ) -> None:
        del completed_count, remaining_count, total_count
        row_id = task_rows.get(task_id)
        if row_id is None:
            row_id = task_rows.get(Path(task_id).name)
        if row_id is not None:
            progress.update(
                row_id,
                completed=1,
                description=f"[green]✓[/green] {task_id}",
            )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        for task_id, label in labels.items():
            task_rows[task_id] = progress.add_task(
                f"[cyan]{phase_label}[/cyan] {label}",
                total=1,
                completed=0,
            )
        summary = await run_scenarios_async(
            suite_paths,
            agent_path=agent_path,
            adapter_override=_resolve_adapter_override(agent_name),
            on_trial_progress=on_trial_progress,
        )
        for task_id, row_id in task_rows.items():
            progress.update(row_id, completed=1, description=f"[green]✓[/green] {task_id}")
        return summary


def _resolve_adapter_override(agent_name: str) -> str | None:
    from ash_hawk.scenario.registry import get_default_adapter_registry

    registry = get_default_adapter_registry()
    candidates = [agent_name]
    normalized = agent_name.replace("-", "_")
    if normalized not in candidates:
        candidates.append(normalized)

    for candidate in candidates:
        if registry.get(candidate) is not None:
            return candidate

    return None
