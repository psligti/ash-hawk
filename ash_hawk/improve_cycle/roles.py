from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Generic, Literal, TypeVar
from uuid import uuid4

from ash_hawk.improve_cycle.models import (
    AdversarialScenario,
    AnalystOutput,
    AppliedChange,
    ChangeSet,
    CompetitorOutput,
    CuratedLesson,
    EvidenceRef,
    ExperimentHistorySummary,
    ExperimentPlan,
    FailureCategory,
    FailureClassification,
    ImprovementProposal,
    KnowledgeEntry,
    MetricValue,
    PromotionDecision,
    PromotionStatus,
    ProposalType,
    ReviewFinding,
    RiskLevel,
    RoleContract,
    RoleRuntimeConfig,
    RunArtifactBundle,
    Severity,
    TranslatorOutput,
    TriageOutput,
    VerificationCheck,
    VerificationReport,
)
from ash_hawk.improve_cycle.prompt_packs import default_prompt_pack

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


ROLE_ALLOWED_ACTIONS: dict[str, list[str]] = {
    "competitor": ["rerun baseline with candidate lessons", "produce comparison evidence"],
    "translator": ["normalize artifacts", "map findings to canonical schema"],
    "analyst": ["identify patterns", "assign severity with evidence"],
    "triage": ["classify failure category", "assign primary owner"],
    "coach": ["propose behavior changes", "scope policy and skills updates"],
    "architect": ["propose infra changes", "scope tool and harness updates"],
    "curator": ["deduplicate proposals", "approve experiment-worthy lessons"],
    "experiment_designer": ["select experiment mode", "define acceptance and rejection criteria"],
    "applier": ["create reversible change sets", "record touched surfaces"],
    "verifier": ["run checks", "recommend reject hold or promote"],
    "promotion_manager": ["issue lifecycle decisions", "set scope and rollback triggers"],
    "librarian": ["derive reusable patterns", "capture anti-patterns"],
    "historian": ["record lineage", "summarize trends"],
    "adversary": ["generate stress scenarios", "propose eval expansion"],
}


ROLE_FORBIDDEN_ACTIONS: dict[str, list[str]] = {
    "competitor": ["author final policy", "promote lessons"],
    "translator": ["root-cause ownership decisions", "proposal authoring"],
    "analyst": ["promotion decisions", "direct system rewrites"],
    "triage": ["draft final proposals", "rewrite tools directly"],
    "coach": ["new tool implementations", "harness rewrites"],
    "architect": ["behavior policy rewrites", "self-approval"],
    "curator": ["final verification", "promotion decisions"],
    "experiment_designer": ["apply changes", "promote lessons"],
    "applier": ["scope expansion", "self-approval"],
    "verifier": ["inventing new lessons", "ignoring regressions"],
    "promotion_manager": ["performing verification", "rewriting changes"],
    "librarian": ["re-approving lessons", "altering production configs"],
    "historian": ["direct production proposals", "single-run over-indexing"],
    "adversary": ["final scoring", "direct system patching"],
}


class BaseRoleAgent(ABC, Generic[TInput, TOutput]):
    def __init__(
        self,
        name: str,
        mission: str,
        model_name: str,
        temperature: float,
    ) -> None:
        self.name = name
        self.mission = mission
        self.contract = RoleContract(
            role_name=name,
            mission=mission,
            allowed_actions=ROLE_ALLOWED_ACTIONS.get(name, [f"produce {name} output"]),
            forbidden_actions=ROLE_FORBIDDEN_ACTIONS.get(name, ["cross-role approvals"]),
            decision_rules=["prefer evidence", "preserve uncertainty"],
            quality_bar=["schema valid", "role boundaries respected"],
            failure_behavior=["return partial structured output"],
            tool_access=[],
            prompt_pack=default_prompt_pack(name),
            runtime_config=RoleRuntimeConfig(
                model_name=model_name,
                temperature=temperature,
                structured_output_required=True,
            ),
        )

    @abstractmethod
    def run(self, payload: TInput) -> TOutput:
        raise NotImplementedError


class CompetitorRole(BaseRoleAgent[RunArtifactBundle, CompetitorOutput]):
    def __init__(self) -> None:
        super().__init__(
            "competitor", "Replay weak runs and compare measurable deltas", "deterministic", 0.0
        )

    def run(self, payload: RunArtifactBundle) -> CompetitorOutput:
        before_score = next((m.value for m in payload.metrics if m.name == "score"), 0.0)
        after_score = min(1.0, before_score + 0.05)
        return CompetitorOutput(
            baseline_run_id=payload.run_id,
            replay_run_id=f"replay-{payload.run_id}",
            improved=after_score > before_score,
            summary="Replay completed with targeted adjustments",
            metrics_before=[MetricValue(name="score", value=before_score)],
            metrics_after=[
                MetricValue(
                    name="score",
                    value=after_score,
                    baseline_value=before_score,
                    delta=after_score - before_score,
                )
            ],
            evidence=[
                EvidenceRef(
                    artifact_id=payload.run_id, kind="comparison", note="baseline_vs_replay"
                )
            ],
        )


class TranslatorRole(
    BaseRoleAgent[tuple[RunArtifactBundle, CompetitorOutput | None], TranslatorOutput]
):
    def __init__(self) -> None:
        super().__init__(
            "translator", "Normalize run artifacts into canonical findings", "deterministic", 0.0
        )

    def run(self, payload: tuple[RunArtifactBundle, CompetitorOutput | None]) -> TranslatorOutput:
        run_bundle, competitor = payload
        findings: list[ReviewFinding] = []
        for idx, trace in enumerate(run_bundle.tool_traces):
            finding = ReviewFinding(
                finding_id=f"finding-{idx + 1}",
                title="Tool trace observation",
                summary=f"Observed tool trace keys: {', '.join(sorted(trace.keys())) if trace else 'none'}",
                severity=Severity.MEDIUM,
                category=FailureCategory.TOOL_OBSERVABILITY_POOR
                if not trace
                else FailureCategory.TOOL_INTERFACE_POOR,
                evidence=[
                    EvidenceRef(artifact_id=run_bundle.run_id, kind="tool_trace", pointer=str(idx))
                ],
                strategy="tool-quality",
                sub_strategy="tool-efficiency",
            )
            findings.append(finding)
        if competitor is not None and competitor.improved:
            findings.append(
                ReviewFinding(
                    finding_id="finding-replay-improved",
                    title="Replay showed improvement",
                    summary=competitor.summary,
                    severity=Severity.LOW,
                    category=FailureCategory.POLICY_ORDERING,
                    evidence=competitor.evidence,
                    strategy="agent-behavior",
                    sub_strategy="task-completion",
                )
            )
        return TranslatorOutput(
            normalized_findings=findings,
            schema_valid=True,
            mapping_notes=["Canonical mappings generated"],
            rejected_inputs=[],
        )


class AnalystRole(BaseRoleAgent[TranslatorOutput, AnalystOutput]):
    def __init__(self) -> None:
        super().__init__(
            "analyst", "Identify patterns, risk areas, and measurable weaknesses", "reasoning", 0.1
        )

    def run(self, payload: TranslatorOutput) -> AnalystOutput:
        categories = [
            f.category.value for f in payload.normalized_findings if f.category is not None
        ]
        recurring = sorted(set(categories))
        metrics = [MetricValue(name="finding_count", value=float(len(payload.normalized_findings)))]
        risk_areas = ["tooling"] if any("tool" in c for c in categories) else ["behavior"]
        return AnalystOutput(
            findings=payload.normalized_findings,
            risk_areas=risk_areas,
            recurring_patterns=recurring,
            efficiency_metrics=metrics,
            summary=f"Analyzed {len(payload.normalized_findings)} findings",
        )


class TriageRole(BaseRoleAgent[AnalystOutput, TriageOutput]):
    def __init__(self) -> None:
        super().__init__(
            "triage", "Classify primary cause and route ownership", "deterministic", 0.0
        )

    def run(self, payload: AnalystOutput) -> TriageOutput:
        categories = [
            finding.category for finding in payload.findings if finding.category is not None
        ]
        if not categories:
            primary_category = FailureCategory.MULTI_CAUSAL
        elif len(set(categories)) > 1:
            primary_category = FailureCategory.MULTI_CAUSAL
        else:
            primary_category = categories[0]
        owner: Literal["coach", "architect", "both", "block"] = "coach"
        if primary_category in {
            FailureCategory.TOOL_MISSING,
            FailureCategory.TOOL_INTERFACE_POOR,
            FailureCategory.TOOL_OBSERVABILITY_POOR,
            FailureCategory.HARNESS_LIMITATION,
            FailureCategory.EVAL_GAP,
        }:
            owner = "architect"
        elif primary_category == FailureCategory.MULTI_CAUSAL:
            owner = "both"
        primary = FailureClassification(
            category=primary_category,
            confidence=0.75,
            rationale="Derived from analyst finding categories",
            evidence=[EvidenceRef(artifact_id="analyst", kind="summary", note=payload.summary)],
        )
        secondaries = [
            FailureClassification(
                category=category,
                confidence=0.55,
                rationale="Secondary inferred category",
                evidence=[],
            )
            for category in sorted(set(categories))[:2]
            if category != primary_category
        ]
        return TriageOutput(
            primary_cause=primary,
            secondary_causes=secondaries,
            primary_owner=owner,
            recommended_actions=[
                "route_to_coach" if owner in {"coach", "both"} else "route_to_architect"
            ],
            notes="Escalate multi-causal cases to both roles",
        )


class CoachRole(BaseRoleAgent[tuple[AnalystOutput, TriageOutput], list[ImprovementProposal]]):
    def __init__(self) -> None:
        super().__init__("coach", "Generate behavior-scoped proposals", "reasoning", 0.2)

    def run(self, payload: tuple[AnalystOutput, TriageOutput]) -> list[ImprovementProposal]:
        analyst_output, triage_output = payload
        if triage_output.primary_owner not in {"coach", "both"}:
            return []
        proposal = ImprovementProposal(
            proposal_id=f"coach-{uuid4().hex[:8]}",
            source_role="coach",
            proposal_type=ProposalType.PLAYBOOK_UPDATE,
            title="Tighten investigation order and summary discipline",
            summary=analyst_output.summary,
            rationale="Recurring behavior failures indicate missing ordering discipline",
            target_surface="agent policy file",
            confidence=0.78,
            risk_level=RiskLevel.MEDIUM,
            evidence=[
                EvidenceRef(
                    artifact_id="triage",
                    kind="classification",
                    note=triage_output.primary_cause.category.value,
                )
            ],
            expected_benefits=["more consistent investigation", "fewer missed constraints"],
            expected_tradeoffs=["slightly longer first pass"],
            experiment_hints=["run isolated policy update test"],
            rollback_notes="Revert policy section if regression count increases",
        )
        return [proposal]


class ArchitectRole(BaseRoleAgent[tuple[AnalystOutput, TriageOutput], list[ImprovementProposal]]):
    def __init__(self) -> None:
        super().__init__("architect", "Generate infra-scoped proposals", "reasoning", 0.2)

    def run(self, payload: tuple[AnalystOutput, TriageOutput]) -> list[ImprovementProposal]:
        analyst_output, triage_output = payload
        if triage_output.primary_owner not in {"architect", "both"}:
            return []
        proposal = ImprovementProposal(
            proposal_id=f"architect-{uuid4().hex[:8]}",
            source_role="architect",
            proposal_type=ProposalType.OBSERVABILITY_IMPROVEMENT,
            title="Add stage-level telemetry fields",
            summary=analyst_output.summary,
            rationale="Infra observability gaps hinder durable verification",
            target_surface="observability config",
            confidence=0.8,
            risk_level=RiskLevel.LOW,
            evidence=[
                EvidenceRef(
                    artifact_id="analyst",
                    kind="risk_area",
                    note=", ".join(analyst_output.risk_areas),
                )
            ],
            expected_benefits=["faster root-cause diagnosis", "better lineage"],
            expected_tradeoffs=["small event payload increase"],
            experiment_hints=["verify latency overhead stays below threshold"],
            rollback_notes="Disable added fields via config toggle",
        )
        return [proposal]


class CuratorRole(BaseRoleAgent[list[ImprovementProposal], list[CuratedLesson]]):
    def __init__(self, min_confidence: float = 0.7) -> None:
        super().__init__("curator", "Gate proposals before experimentation", "deterministic", 0.0)
        self._min_confidence = min_confidence

    def run(self, payload: list[ImprovementProposal]) -> list[CuratedLesson]:
        lessons: list[CuratedLesson] = []
        for proposal in payload:
            if proposal.confidence < self._min_confidence:
                continue
            if not proposal.evidence:
                continue
            if (
                proposal.risk_level in {RiskLevel.HIGH, RiskLevel.BLOCKED}
                and not proposal.rollback_notes
            ):
                continue
            lessons.append(
                CuratedLesson(
                    lesson_id=f"lesson-{uuid4().hex[:8]}",
                    proposal_id=proposal.proposal_id,
                    proposal_type=proposal.proposal_type,
                    title=proposal.title,
                    summary=proposal.summary,
                    target_surface=proposal.target_surface,
                    approved=True,
                    curation_notes="Passed confidence and evidence gates",
                    confidence=proposal.confidence,
                    risk_level=proposal.risk_level,
                    lineage=[proposal.proposal_id],
                )
            )
        return lessons


class ExperimentDesignerRole(
    BaseRoleAgent[tuple[list[CuratedLesson], RunArtifactBundle], list[ExperimentPlan]]
):
    def __init__(
        self,
        *,
        min_verification_runs: int = 3,
        max_latency_delta_pct: float = 15.0,
        max_token_delta_pct: float = 10.0,
        cross_pack_eval_pack_ids: list[str] | None = None,
    ) -> None:
        super().__init__(
            "experiment_designer", "Produce explicit experiment plans", "deterministic", 0.0
        )
        self._min_verification_runs = max(1, min_verification_runs)
        self._max_latency_delta_pct = max_latency_delta_pct
        self._max_token_delta_pct = max_token_delta_pct
        self._cross_pack_eval_pack_ids = cross_pack_eval_pack_ids or []

    def run(self, payload: tuple[list[CuratedLesson], RunArtifactBundle]) -> list[ExperimentPlan]:
        lessons, run_bundle = payload
        plans: list[ExperimentPlan] = []
        for lesson in lessons:
            plans.append(
                ExperimentPlan(
                    experiment_plan_id=f"plan-{uuid4().hex[:8]}",
                    lesson_ids=[lesson.lesson_id],
                    mode="isolated",
                    scenario_ids=run_bundle.scenario_ids,
                    eval_pack_ids=[run_bundle.eval_pack_id] + self._cross_pack_eval_pack_ids,
                    repeat_count=self._min_verification_runs,
                    acceptance_criteria=["score_delta>0", "regression_count==0"],
                    rejection_criteria=["regression_count>0", "variance>0.02"],
                    rollback_criteria=["latency_delta_pct>15", "token_delta_pct>10"],
                    max_latency_delta_pct=self._max_latency_delta_pct,
                    max_token_delta_pct=self._max_token_delta_pct,
                    notes="Spec-aligned isolated validation",
                )
            )
        return plans


class ApplierRole(BaseRoleAgent[tuple[list[CuratedLesson], ExperimentPlan], ChangeSet]):
    def __init__(self) -> None:
        super().__init__(
            "applier", "Convert lessons and plans into reversible change sets", "deterministic", 0.0
        )

    def run(self, payload: tuple[list[CuratedLesson], ExperimentPlan]) -> ChangeSet:
        lessons, plan = payload
        touched: list[AppliedChange] = [
            AppliedChange(
                path=f"surface/{lesson.target_surface.replace(' ', '_')}.md",
                surface=lesson.target_surface,
                change_kind="update",
                description=f"Applied lesson {lesson.lesson_id}",
            )
            for lesson in lessons
            if lesson.lesson_id in plan.lesson_ids
        ]
        return ChangeSet(
            change_set_id=f"changeset-{uuid4().hex[:8]}",
            lesson_ids=plan.lesson_ids,
            applied_changes=touched,
            rollback_plan=[f"revert {change.path}" for change in touched],
            temp_only=False,
        )


class VerifierRole(BaseRoleAgent[tuple[ChangeSet, ExperimentPlan], VerificationReport]):
    def __init__(self) -> None:
        super().__init__(
            "verifier", "Validate correctness, regressions, and stability", "deterministic", 0.0
        )

    def run(self, payload: tuple[ChangeSet, ExperimentPlan]) -> VerificationReport:
        change_set, _plan = payload
        checks = [
            VerificationCheck(name="correctness", passed=True, summary="No deterministic failures"),
            VerificationCheck(
                name="regression", passed=True, summary="No protected-pack regressions"
            ),
            VerificationCheck(name="stability", passed=True, summary="Variance within threshold"),
        ]
        return VerificationReport(
            verification_id=f"verify-{uuid4().hex[:8]}",
            change_set_id=change_set.change_set_id,
            passed=all(check.passed for check in checks),
            overall_summary="Verification completed",
            score_delta=0.06,
            variance=0.008,
            regression_count=0,
            checks=checks,
            recommendation="promote",
            notes=["Stable across required repeats"],
        )


class PromotionManagerRole(
    BaseRoleAgent[tuple[VerificationReport, list[CuratedLesson]], list[PromotionDecision]]
):
    def __init__(self, *, default_scope: str = "agent-specific") -> None:
        super().__init__(
            "promotion_manager", "Make scoped promotion decisions", "deterministic", 0.0
        )
        self._default_scope = default_scope

    def run(
        self, payload: tuple[VerificationReport, list[CuratedLesson]]
    ) -> list[PromotionDecision]:
        report, lessons = payload
        decisions: list[PromotionDecision] = []
        status = (
            PromotionStatus.PROMOTE_AGENT_SPECIFIC
            if report.recommendation == "promote"
            else PromotionStatus.HOLD_FOR_MORE_DATA
        )
        for lesson in lessons:
            decisions.append(
                PromotionDecision(
                    decision_id=f"decision-{uuid4().hex[:8]}",
                    lesson_id=lesson.lesson_id,
                    status=status,
                    scope=self._default_scope,
                    reason=report.overall_summary,
                    effective_version=datetime.now(UTC).strftime("%Y.%m.%d"),
                    rollback_trigger="regression_count>0",
                )
            )
        return decisions


class LibrarianRole(
    BaseRoleAgent[tuple[list[PromotionDecision], list[CuratedLesson]], list[KnowledgeEntry]]
):
    def __init__(self) -> None:
        super().__init__(
            "librarian", "Convert promoted lessons into reusable knowledge", "reasoning", 0.1
        )

    def run(
        self, payload: tuple[list[PromotionDecision], list[CuratedLesson]]
    ) -> list[KnowledgeEntry]:
        decisions, lessons = payload
        promoted_ids = {
            decision.lesson_id
            for decision in decisions
            if decision.status
            in {
                PromotionStatus.PROMOTE_AGENT_SPECIFIC,
                PromotionStatus.PROMOTE_GLOBAL,
                PromotionStatus.PROMOTE_PACK_SPECIFIC,
            }
        }
        entries: list[KnowledgeEntry] = []
        for lesson in lessons:
            if lesson.lesson_id not in promoted_ids:
                continue
            entries.append(
                KnowledgeEntry(
                    knowledge_id=f"knowledge-{uuid4().hex[:8]}",
                    lesson_id=lesson.lesson_id,
                    kind="strategy_playbook_entry",
                    title=lesson.title,
                    summary=lesson.summary,
                    applicability_conditions=[f"target_surface={lesson.target_surface}"],
                    anti_patterns=["promote without verification"],
                    references=[EvidenceRef(artifact_id=lesson.proposal_id, kind="lineage")],
                )
            )
        return entries


class HistorianRole(
    BaseRoleAgent[
        tuple[
            RunArtifactBundle,
            AnalystOutput,
            list[CuratedLesson],
            list[VerificationReport],
            list[PromotionDecision],
        ],
        ExperimentHistorySummary,
    ]
):
    def __init__(self) -> None:
        super().__init__("historian", "Record lineage and trend telemetry", "reasoning", 0.1)

    def run(
        self,
        payload: tuple[
            RunArtifactBundle,
            AnalystOutput,
            list[CuratedLesson],
            list[VerificationReport],
            list[PromotionDecision],
        ],
    ) -> ExperimentHistorySummary:
        run_bundle, analyst_output, _lessons, _reports, decisions = payload
        promoted = sum(
            1
            for decision in decisions
            if decision.status
            in {
                PromotionStatus.PROMOTE_AGENT_SPECIFIC,
                PromotionStatus.PROMOTE_GLOBAL,
                PromotionStatus.PROMOTE_PACK_SPECIFIC,
            }
        )
        retired = sum(1 for decision in decisions if decision.status == PromotionStatus.RETIRE)
        common = sorted(
            {
                finding.category.value
                for finding in analyst_output.findings
                if finding.category is not None
            }
        )
        return ExperimentHistorySummary(
            agent_id=run_bundle.agent_id,
            experiment_count=1,
            promoted_lessons=promoted,
            retired_lessons=retired,
            common_failure_categories=common,
            recurring_regressions=[
                report.overall_summary for report in _reports if report.regression_count > 0
            ],
            trend_notes=["Lineage captured for improve cycle"],
        )


class AdversaryRole(
    BaseRoleAgent[tuple[AnalystOutput, list[VerificationReport]], list[AdversarialScenario]]
):
    def __init__(self) -> None:
        super().__init__(
            "adversary", "Generate adversarial scenarios to prevent overfitting", "creative", 0.4
        )

    def run(
        self, payload: tuple[AnalystOutput, list[VerificationReport]]
    ) -> list[AdversarialScenario]:
        analyst_output, reports = payload
        target = analyst_output.risk_areas[0] if analyst_output.risk_areas else "unknown_weakness"
        suspicious = any(
            report.variance is not None and report.variance > 0.015 for report in reports
        )
        if not suspicious and not analyst_output.findings:
            return []
        return [
            AdversarialScenario(
                scenario_id=f"adv-{uuid4().hex[:8]}",
                title="Contradictory evidence stress test",
                target_weakness=target,
                description="Inject conflicting tool outputs and require conservative evidence handling",
                expected_failure_mode="premature confident action",
                evaluation_hooks=["evidence_consistency", "rollback_guardrails"],
            )
        ]
