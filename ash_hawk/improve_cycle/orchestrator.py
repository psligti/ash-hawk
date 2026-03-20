from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Literal, TypeVar, cast
from uuid import uuid4

from ash_hawk.improve_cycle.configuration import ImproveCyclePromotionConfig
from ash_hawk.improve_cycle.models import (
    AdversarialScenario,
    AnalystOutput,
    ChangeSet,
    CuratedLesson,
    ExperimentHistorySummary,
    ExperimentPlan,
    FailureCategory,
    FailureClassification,
    ImproveCycleCheckpoint,
    ImprovementProposal,
    KnowledgeEntry,
    PromotionDecision,
    ReviewFinding,
    RoleLifecycleEvent,
    RunArtifactBundle,
    TriageOutput,
    VerificationReport,
)
from ash_hawk.improve_cycle.promotion import PROMOTED_STATUSES, PromotionContext
from ash_hawk.improve_cycle.roles import (
    AdversaryRole,
    AnalystRole,
    ApplierRole,
    ArchitectRole,
    CoachRole,
    CompetitorRole,
    CuratorRole,
    ExperimentDesignerRole,
    HistorianRole,
    LibrarianRole,
    PromotionManagerRole,
    TranslatorRole,
    TriageRole,
    VerifierRole,
)
from ash_hawk.improve_cycle.routing import should_run_architect, should_run_coach
from ash_hawk.improve_cycle.storage import ImproveCycleStorage

TRun = TypeVar("TRun")


@dataclass
class ImproveCycleResult:
    triage: TriageOutput
    proposals: list[ImprovementProposal]
    curated_lessons: list[CuratedLesson]
    experiment_plans: list[ExperimentPlan]
    change_sets: list[ChangeSet]
    verification_reports: list[VerificationReport]
    promotion_decisions: list[PromotionDecision]
    knowledge_entries: list[KnowledgeEntry]
    history: ExperimentHistorySummary
    adversarial_scenarios: list[AdversarialScenario]


class ImproveCycleOrchestrator:
    def __init__(
        self,
        storage: ImproveCycleStorage | None = None,
        *,
        enable_competitor: bool = True,
        enable_triage: bool = True,
        enable_verifier: bool = True,
        enable_adversary: bool = True,
        min_verification_runs: int = 3,
        max_latency_delta_pct: float = 15.0,
        max_token_delta_pct: float = 10.0,
        cross_pack_eval_pack_ids: list[str] | None = None,
        promotion_scope: str = "agent-specific",
        promotion_config: ImproveCyclePromotionConfig | None = None,
    ) -> None:
        self.storage = storage or ImproveCycleStorage()
        self.enable_competitor = enable_competitor
        self.enable_triage = enable_triage
        self.enable_verifier = enable_verifier
        self.enable_adversary = enable_adversary
        self._promotion_config = promotion_config or ImproveCyclePromotionConfig()
        self.competitor = CompetitorRole()
        self.translator = TranslatorRole()
        self.analyst = AnalystRole()
        self.triage = TriageRole()
        self.coach = CoachRole()
        self.architect = ArchitectRole()
        self.curator = CuratorRole()
        self.experiment_designer = ExperimentDesignerRole(
            min_verification_runs=min_verification_runs,
            max_latency_delta_pct=max_latency_delta_pct,
            max_token_delta_pct=max_token_delta_pct,
            cross_pack_eval_pack_ids=cross_pack_eval_pack_ids,
        )
        self.applier = ApplierRole()
        self.verifier = VerifierRole()
        self.promotion_manager = PromotionManagerRole(
            config=self._promotion_config,
            default_scope=promotion_scope,
        )
        self.librarian = LibrarianRole()
        self.historian = HistorianRole()
        self.adversary = AdversaryRole()

    def _extract_refs(self, payload: Any) -> list[str]:
        refs: list[str] = []

        def collect(item: Any) -> None:
            if item is None:
                return
            if isinstance(item, (list, tuple, set)):
                return
            if isinstance(item, dict):
                for key in (
                    "run_id",
                    "experiment_id",
                    "lesson_id",
                    "proposal_id",
                    "verification_id",
                ):
                    value = cast(dict[str, Any], item).get(key)
                    if isinstance(value, str):
                        refs.append(value)
                return
            for attr in (
                "run_id",
                "experiment_id",
                "lesson_id",
                "proposal_id",
                "verification_id",
                "change_set_id",
                "experiment_plan_id",
                "decision_id",
                "finding_id",
                "scenario_id",
            ):
                value = getattr(item, attr, None)
                if isinstance(value, str):
                    refs.append(value)

        collect(payload)
        return sorted(set(refs))

    def _extract_metrics(self, output: Any) -> dict[str, float | int]:
        if isinstance(output, VerificationReport):
            metrics: dict[str, float | int] = {"regression_count": output.regression_count}
            if output.score_delta is not None:
                metrics["score_delta"] = output.score_delta
            if output.variance is not None:
                metrics["variance"] = output.variance
            return metrics
        return {}

    def _record_role_event(
        self,
        *,
        event_type: Literal["role_started", "role_completed", "role_failed"],
        role: str,
        run_bundle: RunArtifactBundle,
        status: Literal["success", "failed", "started"],
        payload: Any,
        output: Any = None,
        duration_ms: int | None = None,
        error_info: str | None = None,
    ) -> None:
        event = RoleLifecycleEvent(
            event_id=f"event-{uuid4().hex[:10]}",
            event_type=event_type,
            role=role,
            run_id=run_bundle.run_id,
            experiment_id=run_bundle.experiment_id,
            duration_ms=duration_ms,
            status=status,
            input_refs=self._extract_refs(payload),
            output_refs=self._extract_refs(output),
            metrics=self._extract_metrics(output),
            error_info=error_info,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self.storage.role_events.upsert(event, RoleLifecycleEvent)

    def _run_role(
        self,
        *,
        role_name: str,
        run_bundle: RunArtifactBundle,
        payload: Any,
        callback: Callable[[Any], TRun],
    ) -> TRun:
        self._record_role_event(
            event_type="role_started",
            role=role_name,
            run_bundle=run_bundle,
            status="started",
            payload=payload,
        )
        started = time.perf_counter()
        max_attempts = {
            "translator": 2,
            "analyst": 2,
            "triage": 2,
            "coach": 2,
            "architect": 2,
            "verifier": 2,
        }.get(role_name, 1)
        attempt = 0
        while True:
            attempt += 1
            try:
                output = callback(payload)
                break
            except Exception as exc:
                retryable = role_name != "verifier" or isinstance(exc, (TimeoutError, OSError))
                if attempt < max_attempts and retryable:
                    continue
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                self._record_role_event(
                    event_type="role_failed",
                    role=role_name,
                    run_bundle=run_bundle,
                    status="failed",
                    payload=payload,
                    duration_ms=elapsed_ms,
                    error_info=f"attempt={attempt}: {exc}",
                )
                raise
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self._record_role_event(
            event_type="role_completed",
            role=role_name,
            run_bundle=run_bundle,
            status="success",
            payload=payload,
            output=output,
            duration_ms=elapsed_ms,
        )
        return output

    def _save_checkpoint(
        self,
        *,
        run_bundle: RunArtifactBundle,
        role: Literal["analyst", "curator", "verifier", "promotion_manager", "complete"],
        state: dict[str, Any],
        status: Literal["in_progress", "completed"] = "in_progress",
    ) -> None:
        checkpoint = ImproveCycleCheckpoint(
            checkpoint_id=run_bundle.run_id,
            run_id=run_bundle.run_id,
            experiment_id=run_bundle.experiment_id,
            last_completed_role=role,
            status=status,
            state=state,
            saved_at=datetime.now(UTC).isoformat(),
        )
        self.storage.checkpoints.upsert(checkpoint, ImproveCycleCheckpoint)

    def _change_set_lessons(self, change_sets: list[ChangeSet]) -> dict[str, list[str]]:
        return {change_set.change_set_id: list(change_set.lesson_ids) for change_set in change_sets}

    def _cross_pack_validated_lessons(
        self,
        *,
        reports: list[VerificationReport],
        plans: list[ExperimentPlan],
        change_sets: list[ChangeSet],
    ) -> set[str]:
        plan_by_change_set = {
            change_set.change_set_id: plan
            for change_set, plan in zip(change_sets, plans, strict=False)
        }
        lessons_by_change_set = self._change_set_lessons(change_sets)
        cross_pack: set[str] = set()
        for report in reports:
            plan = plan_by_change_set.get(report.change_set_id)
            if plan is None:
                continue
            if len(plan.eval_pack_ids) <= 1:
                continue
            if not report.passed or report.regression_count > 0:
                continue
            cross_pack.update(lessons_by_change_set.get(report.change_set_id, []))
        return cross_pack

    def run_cycle(self, run_bundle: RunArtifactBundle) -> ImproveCycleResult:
        self.storage.runs.upsert(run_bundle, RunArtifactBundle)
        checkpoint = self.storage.checkpoints.get(run_bundle.run_id, ImproveCycleCheckpoint)
        resumed_state = (
            checkpoint.state if checkpoint and checkpoint.status == "in_progress" else {}
        )
        resume_role = (
            checkpoint.last_completed_role
            if checkpoint and checkpoint.status == "in_progress"
            else None
        )

        competitor_output = None
        translator_output = None
        analyst_output = None

        if resume_role in {"analyst", "curator", "verifier", "promotion_manager"}:
            analyst_output = AnalystOutput.model_validate(resumed_state["analyst_output"])
        else:
            competitor_output = (
                self._run_role(
                    role_name="competitor",
                    run_bundle=run_bundle,
                    payload=run_bundle,
                    callback=self.competitor.run,
                )
                if self.enable_competitor
                else None
            )
            translator_payload = (run_bundle, competitor_output)
            translator_output = self._run_role(
                role_name="translator",
                run_bundle=run_bundle,
                payload=translator_payload,
                callback=self.translator.run,
            )
            for finding in translator_output.normalized_findings:
                self.storage.findings.upsert(finding, ReviewFinding)
            analyst_output = self._run_role(
                role_name="analyst",
                run_bundle=run_bundle,
                payload=translator_output,
                callback=self.analyst.run,
            )
            for finding in analyst_output.findings:
                self.storage.findings.upsert(finding, ReviewFinding)
            self._save_checkpoint(
                run_bundle=run_bundle,
                role="analyst",
                state={"analyst_output": analyst_output.model_dump(mode="json")},
            )

        proposals: list[ImprovementProposal]
        curated_lessons: list[CuratedLesson]
        triage_output: TriageOutput

        if resume_role in {"curator", "verifier", "promotion_manager"}:
            triage_output = TriageOutput.model_validate(resumed_state["triage_output"])
            proposals = [
                ImprovementProposal.model_validate(item) for item in resumed_state["proposals"]
            ]
            curated_lessons = [
                CuratedLesson.model_validate(item) for item in resumed_state["curated_lessons"]
            ]
        else:
            if self.enable_triage:
                triage_output = self._run_role(
                    role_name="triage",
                    run_bundle=run_bundle,
                    payload=analyst_output,
                    callback=self.triage.run,
                )
            else:
                triage_output = TriageOutput(
                    primary_cause=FailureClassification(
                        category=FailureCategory.MULTI_CAUSAL,
                        confidence=0.5,
                        rationale="triage disabled",
                    ),
                    primary_owner="both",
                    notes="triage disabled; defaulting to both",
                )

            proposals = []
            if should_run_coach(triage_output):
                proposals.extend(
                    self._run_role(
                        role_name="coach",
                        run_bundle=run_bundle,
                        payload=(analyst_output, triage_output),
                        callback=self.coach.run,
                    )
                )
            if should_run_architect(triage_output):
                proposals.extend(
                    self._run_role(
                        role_name="architect",
                        run_bundle=run_bundle,
                        payload=(analyst_output, triage_output),
                        callback=self.architect.run,
                    )
                )

            for proposal in proposals:
                self.storage.proposals.upsert(proposal, ImprovementProposal)

            curated_lessons = self._run_role(
                role_name="curator",
                run_bundle=run_bundle,
                payload=proposals,
                callback=self.curator.run,
            )
            for lesson in curated_lessons:
                self.storage.lessons.upsert(lesson, CuratedLesson)
            self._save_checkpoint(
                run_bundle=run_bundle,
                role="curator",
                state={
                    "analyst_output": analyst_output.model_dump(mode="json"),
                    "triage_output": triage_output.model_dump(mode="json"),
                    "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
                    "curated_lessons": [
                        lesson.model_dump(mode="json") for lesson in curated_lessons
                    ],
                },
            )

        experiment_plans: list[ExperimentPlan]
        change_sets: list[ChangeSet]
        verification_reports: list[VerificationReport]

        if resume_role in {"verifier", "promotion_manager"}:
            experiment_plans = [
                ExperimentPlan.model_validate(item) for item in resumed_state["experiment_plans"]
            ]
            change_sets = [ChangeSet.model_validate(item) for item in resumed_state["change_sets"]]
            verification_reports = [
                VerificationReport.model_validate(item)
                for item in resumed_state["verification_reports"]
            ]
        else:
            experiment_plans = self._run_role(
                role_name="experiment_designer",
                run_bundle=run_bundle,
                payload=(curated_lessons, run_bundle),
                callback=self.experiment_designer.run,
            )
            for plan in experiment_plans:
                self.storage.experiment_plans.upsert(plan, ExperimentPlan)

            change_sets = []
            verification_reports = []
            for plan in experiment_plans:
                change_set = self._run_role(
                    role_name="applier",
                    run_bundle=run_bundle,
                    payload=(curated_lessons, plan),
                    callback=self.applier.run,
                )
                self.storage.change_sets.upsert(change_set, ChangeSet)
                change_sets.append(change_set)

                report = (
                    self._run_role(
                        role_name="verifier",
                        run_bundle=run_bundle,
                        payload=(change_set, plan),
                        callback=self.verifier.run,
                    )
                    if self.enable_verifier
                    else VerificationReport(
                        verification_id=f"verify-disabled-{change_set.change_set_id}",
                        change_set_id=change_set.change_set_id,
                        passed=True,
                        overall_summary="Verifier disabled",
                        recommendation="hold",
                    )
                )
                self.storage.verifications.upsert(report, VerificationReport)
                verification_reports.append(report)
            self._save_checkpoint(
                run_bundle=run_bundle,
                role="verifier",
                state={
                    "analyst_output": analyst_output.model_dump(mode="json"),
                    "triage_output": triage_output.model_dump(mode="json"),
                    "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
                    "curated_lessons": [
                        lesson.model_dump(mode="json") for lesson in curated_lessons
                    ],
                    "experiment_plans": [plan.model_dump(mode="json") for plan in experiment_plans],
                    "change_sets": [
                        change_set.model_dump(mode="json") for change_set in change_sets
                    ],
                    "verification_reports": [
                        report.model_dump(mode="json") for report in verification_reports
                    ],
                },
            )

        promotion_decisions: list[PromotionDecision]
        if resume_role == "promotion_manager":
            promotion_decisions = [
                PromotionDecision.model_validate(item)
                for item in resumed_state["promotion_decisions"]
            ]
        else:
            promotion_decisions = []
            lesson_success_counts = cast(
                dict[str, int], resumed_state.get("promotion_success_counts", {})
            )
            lesson_failure_counts = cast(
                dict[str, int], resumed_state.get("promotion_failure_counts", {})
            )
            lessons_by_change_set = self._change_set_lessons(change_sets)
            cross_pack_lesson_ids = self._cross_pack_validated_lessons(
                reports=verification_reports,
                plans=experiment_plans,
                change_sets=change_sets,
            )
            promoted_lessons_by_surface: dict[str, set[str]] = {}
            for existing in self.storage.promotions.list_all(PromotionDecision):
                if existing.status not in PROMOTED_STATUSES:
                    continue
                existing_lesson = self.storage.lessons.get(existing.lesson_id, CuratedLesson)
                if existing_lesson is None:
                    continue
                promoted_lessons_by_surface.setdefault(existing_lesson.target_surface, set()).add(
                    existing.lesson_id
                )
            for report in verification_reports:
                report_lesson_ids = lessons_by_change_set.get(report.change_set_id, [])
                for lesson_id in report_lesson_ids:
                    if report.passed and report.regression_count == 0:
                        lesson_success_counts[lesson_id] = (
                            lesson_success_counts.get(lesson_id, 0) + 1
                        )
                        lesson_failure_counts[lesson_id] = 0
                    else:
                        lesson_failure_counts[lesson_id] = (
                            lesson_failure_counts.get(lesson_id, 0) + 1
                        )
                        lesson_success_counts[lesson_id] = 0
                scoped_lessons = [
                    lesson for lesson in curated_lessons if lesson.lesson_id in report_lesson_ids
                ]
                if not scoped_lessons:
                    scoped_lessons = curated_lessons
                conflicting_lesson_ids: set[str] = set()
                for lesson in scoped_lessons:
                    prior_promoted = promoted_lessons_by_surface.get(lesson.target_surface, set())
                    if any(
                        other_lesson_id != lesson.lesson_id for other_lesson_id in prior_promoted
                    ):
                        conflicting_lesson_ids.add(lesson.lesson_id)
                context = PromotionContext(
                    consecutive_successes=dict(lesson_success_counts),
                    consecutive_failures=dict(lesson_failure_counts),
                    cross_pack_validated_lesson_ids=cross_pack_lesson_ids,
                    conflicting_lesson_ids=conflicting_lesson_ids,
                )
                decisions = self._run_role(
                    role_name="promotion_manager",
                    run_bundle=run_bundle,
                    payload=(report, scoped_lessons, context),
                    callback=self.promotion_manager.run,
                )
                promotion_decisions.extend(decisions)
            for decision in promotion_decisions:
                self.storage.promotions.upsert(decision, PromotionDecision)
            self._save_checkpoint(
                run_bundle=run_bundle,
                role="promotion_manager",
                state={
                    "analyst_output": analyst_output.model_dump(mode="json"),
                    "triage_output": triage_output.model_dump(mode="json"),
                    "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
                    "curated_lessons": [
                        lesson.model_dump(mode="json") for lesson in curated_lessons
                    ],
                    "experiment_plans": [plan.model_dump(mode="json") for plan in experiment_plans],
                    "change_sets": [
                        change_set.model_dump(mode="json") for change_set in change_sets
                    ],
                    "verification_reports": [
                        report.model_dump(mode="json") for report in verification_reports
                    ],
                    "promotion_decisions": [
                        decision.model_dump(mode="json") for decision in promotion_decisions
                    ],
                    "promotion_success_counts": lesson_success_counts,
                    "promotion_failure_counts": lesson_failure_counts,
                },
            )

        knowledge_entries = self._run_role(
            role_name="librarian",
            run_bundle=run_bundle,
            payload=(promotion_decisions, curated_lessons),
            callback=self.librarian.run,
        )
        for entry in knowledge_entries:
            self.storage.knowledge.upsert(entry, KnowledgeEntry)

        history = self._run_role(
            role_name="historian",
            run_bundle=run_bundle,
            payload=(
                run_bundle,
                analyst_output,
                curated_lessons,
                verification_reports,
                promotion_decisions,
            ),
            callback=self.historian.run,
        )
        self.storage.histories.upsert(history, ExperimentHistorySummary)

        adversarial_scenarios = (
            self._run_role(
                role_name="adversary",
                run_bundle=run_bundle,
                payload=(analyst_output, verification_reports),
                callback=self.adversary.run,
            )
            if self.enable_adversary
            else []
        )
        for scenario in adversarial_scenarios:
            self.storage.adversarial_scenarios.upsert(scenario, AdversarialScenario)

        self._save_checkpoint(
            run_bundle=run_bundle,
            role="complete",
            status="completed",
            state={
                "promotion_decisions": [
                    decision.model_dump(mode="json") for decision in promotion_decisions
                ]
            },
        )

        return ImproveCycleResult(
            triage=triage_output,
            proposals=proposals,
            curated_lessons=curated_lessons,
            experiment_plans=experiment_plans,
            change_sets=change_sets,
            verification_reports=verification_reports,
            promotion_decisions=promotion_decisions,
            knowledge_entries=knowledge_entries,
            history=history,
            adversarial_scenarios=adversarial_scenarios,
        )
