from __future__ import annotations

from ash_hawk.improve_cycle.configuration import ImproveCyclePromotionConfig
from ash_hawk.improve_cycle.models import (
    AnalystOutput,
    ChangeSet,
    CompetitorOutput,
    CuratedLesson,
    EvidenceRef,
    ExperimentPlan,
    FailureCategory,
    FailureClassification,
    ImprovementProposal,
    MetricValue,
    PromotionDecision,
    PromotionStatus,
    ProposalType,
    ReviewFinding,
    RiskLevel,
    RunArtifactBundle,
    Severity,
    TranslatorOutput,
    TriageOutput,
    VerificationReport,
)
from ash_hawk.improve_cycle.promotion import PromotionContext
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


def _run_bundle() -> RunArtifactBundle:
    return RunArtifactBundle(
        run_id="run-1",
        experiment_id="exp-1",
        agent_id="agent-1",
        eval_pack_id="pack-1",
        scenario_ids=["scenario-1"],
        timestamp="2026-01-01T00:00:00Z",
        tool_traces=[{"tool": "read", "status": "ok"}, {}],
        outputs=[{"text": "policy violation detected"}],
        metrics=[MetricValue(name="score", value=0.4), MetricValue(name="latency_ms", value=1200)],
    )


def _proposal() -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id="proposal-1",
        source_role="coach",
        proposal_type=ProposalType.PLAYBOOK_UPDATE,
        title="Improve checklist",
        summary="Add checklist",
        rationale="reduce misses",
        target_surface="agent policy file",
        confidence=0.8,
        risk_level=RiskLevel.LOW,
        evidence=[EvidenceRef(artifact_id="run-1", kind="finding")],
        rollback_notes="revert checklist",
    )


def _lesson() -> CuratedLesson:
    return CuratedLesson(
        lesson_id="lesson-1",
        proposal_id="proposal-1",
        proposal_type=ProposalType.PLAYBOOK_UPDATE,
        title="Lesson",
        summary="Summary",
        target_surface="agent policy file",
        approved=True,
        curation_notes="good",
        confidence=0.8,
        risk_level=RiskLevel.LOW,
    )


def test_competitor_role_outputs_metric_deltas() -> None:
    role = CompetitorRole()
    result = role.run(_run_bundle())
    assert result.improved is True
    assert any(metric.delta is not None for metric in result.metrics_after)


def test_translator_role_generates_findings_from_artifacts() -> None:
    role = TranslatorRole()
    result = role.run((_run_bundle(), None))
    assert result.normalized_findings
    assert any(
        finding.category == FailureCategory.POLICY_GUARDRAIL
        for finding in result.normalized_findings
    )


def test_analyst_role_derives_risk_areas_and_metrics() -> None:
    role = AnalystRole()
    output = TranslatorOutput(
        normalized_findings=[
            ReviewFinding(
                finding_id="f-1",
                title="Tool issue",
                summary="tool error",
                severity=Severity.HIGH,
                category=FailureCategory.TOOL_INTERFACE_POOR,
            )
        ]
    )
    result = role.run(output)
    assert "tooling" in result.risk_areas
    assert any(metric.name == "finding_count" for metric in result.efficiency_metrics)


def test_triage_role_routes_environmental_flake_to_block() -> None:
    role = TriageRole()
    analyst = AnalystOutput(
        findings=[
            ReviewFinding(
                finding_id="f-1",
                title="flake",
                summary="unstable",
                severity=Severity.HIGH,
                category=FailureCategory.ENVIRONMENTAL_FLAKE,
            )
        ],
        risk_areas=["reliability"],
        summary="flake",
    )
    result = role.run(analyst)
    assert result.primary_owner == "block"


def test_coach_role_returns_behavior_proposals_when_routed() -> None:
    role = CoachRole()
    analyst = AnalystOutput(findings=[], risk_areas=["behavior"], summary="analysis")
    triage = TriageOutput(
        primary_cause=FailureClassification(
            category=FailureCategory.POLICY_ORDERING,
            confidence=0.8,
            rationale="policy",
        ),
        primary_owner="coach",
    )
    result = role.run((analyst, triage))
    assert result
    assert result[0].source_role == "coach"


def test_architect_role_returns_infra_proposals_when_routed() -> None:
    role = ArchitectRole()
    analyst = AnalystOutput(findings=[], risk_areas=["tooling"], summary="analysis")
    triage = TriageOutput(
        primary_cause=FailureClassification(
            category=FailureCategory.TOOL_MISSING,
            confidence=0.8,
            rationale="tool",
        ),
        primary_owner="architect",
    )
    result = role.run((analyst, triage))
    assert result
    assert result[0].source_role == "architect"


def test_curator_role_filters_and_approves_best_proposals() -> None:
    role = CuratorRole()
    p1 = _proposal()
    p2 = _proposal().model_copy(update={"proposal_id": "proposal-2", "confidence": 0.9})
    lessons = role.run([p1, p2])
    assert len(lessons) == 1
    assert lessons[0].proposal_id == "proposal-2"


def test_experiment_designer_role_selects_mode_from_lesson_profile() -> None:
    role = ExperimentDesignerRole(cross_pack_eval_pack_ids=["pack-2"])
    lesson = _lesson().model_copy(update={"proposal_type": ProposalType.EVAL_EXPANSION})
    plans = role.run(([lesson], _run_bundle()))
    assert plans[0].mode == "cross_pack"
    assert "pack-2" in plans[0].eval_pack_ids


def test_applier_role_creates_reversible_changeset() -> None:
    role = ApplierRole()
    plan = ExperimentPlan(experiment_plan_id="plan-1", lesson_ids=["lesson-1"], mode="isolated")
    result = role.run(([_lesson()], plan))
    assert result.applied_changes
    assert result.rollback_plan


def test_verifier_role_computes_checks_and_recommendation() -> None:
    role = VerifierRole()
    plan = ExperimentPlan(experiment_plan_id="plan-1", lesson_ids=["lesson-1"], mode="isolated")
    change_set = ChangeSet(change_set_id="cs-1", lesson_ids=["lesson-1"], applied_changes=[])
    result = role.run((change_set, plan))
    assert result.recommendation in {"reject", "hold", "promote"}
    assert len(result.checks) == 3


def test_promotion_manager_role_uses_policy_and_context() -> None:
    role = PromotionManagerRole(
        config=ImproveCyclePromotionConfig(low_risk_success_threshold=1),
        default_scope="agent-specific",
    )
    report = VerificationReport(
        verification_id="verify-1",
        change_set_id="cs-1",
        passed=True,
        overall_summary="ok",
        score_delta=0.2,
        variance=0.01,
        regression_count=0,
        recommendation="promote",
    )
    decisions = role.run(
        (
            report,
            [_lesson()],
            PromotionContext(
                consecutive_successes={"lesson-1": 1},
                cross_pack_validated_lesson_ids={"lesson-1"},
            ),
        )
    )
    assert decisions[0].status == PromotionStatus.PROMOTE_GLOBAL


def test_librarian_role_emits_knowledge_for_promoted_lessons() -> None:
    role = LibrarianRole()
    decision = PromotionDecision(
        decision_id="decision-1",
        lesson_id="lesson-1",
        status=PromotionStatus.PROMOTE_AGENT_SPECIFIC,
        scope="agent-specific",
        reason="ok",
    )
    entries = role.run(([decision], [_lesson()]))
    assert entries
    assert entries[0].lesson_id == "lesson-1"


def test_historian_role_summarizes_lineage_and_trends() -> None:
    role = HistorianRole()
    analyst = AnalystOutput(findings=[], risk_areas=["behavior"], summary="summary")
    report = VerificationReport(
        verification_id="verify-1",
        change_set_id="cs-1",
        passed=True,
        overall_summary="ok",
        score_delta=0.1,
        variance=0.01,
        regression_count=0,
        recommendation="promote",
    )
    decision = PromotionDecision(
        decision_id="decision-1",
        lesson_id="lesson-1",
        status=PromotionStatus.PROMOTE_AGENT_SPECIFIC,
        scope="agent-specific",
        reason="ok",
    )
    summary = role.run((_run_bundle(), analyst, [_lesson()], [report], [decision]))
    assert summary.promoted_lessons == 1
    assert summary.experiment_count == 1


def test_adversary_role_generates_stress_scenarios_for_risky_signals() -> None:
    role = AdversaryRole()
    analyst = AnalystOutput(
        findings=[
            ReviewFinding(
                finding_id="f-1",
                title="multi",
                summary="multi",
                severity=Severity.HIGH,
                category=FailureCategory.MULTI_CAUSAL,
            )
        ],
        risk_areas=["reliability"],
        summary="summary",
    )
    report = VerificationReport(
        verification_id="verify-1",
        change_set_id="cs-1",
        passed=True,
        overall_summary="ok",
        score_delta=0.1,
        variance=0.02,
        regression_count=1,
        recommendation="hold",
    )
    scenarios = role.run((analyst, [report]))
    assert len(scenarios) >= 2
