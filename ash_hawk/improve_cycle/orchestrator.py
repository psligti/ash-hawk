from __future__ import annotations

from dataclasses import dataclass

from ash_hawk.improve_cycle.models import (
    AdversarialScenario,
    ChangeSet,
    CuratedLesson,
    ExperimentHistorySummary,
    ExperimentPlan,
    FailureCategory,
    FailureClassification,
    ImprovementProposal,
    KnowledgeEntry,
    PromotionDecision,
    ReviewFinding,
    RunArtifactBundle,
    TriageOutput,
    VerificationReport,
)
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
    ) -> None:
        self.storage = storage or ImproveCycleStorage()
        self.enable_competitor = enable_competitor
        self.enable_triage = enable_triage
        self.enable_verifier = enable_verifier
        self.enable_adversary = enable_adversary
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
        self.promotion_manager = PromotionManagerRole(default_scope=promotion_scope)
        self.librarian = LibrarianRole()
        self.historian = HistorianRole()
        self.adversary = AdversaryRole()

    def run_cycle(self, run_bundle: RunArtifactBundle) -> ImproveCycleResult:
        self.storage.runs.upsert(run_bundle, RunArtifactBundle)

        competitor_output = self.competitor.run(run_bundle) if self.enable_competitor else None
        translator_output = self.translator.run((run_bundle, competitor_output))
        for finding in translator_output.normalized_findings:
            self.storage.findings.upsert(finding, ReviewFinding)
        analyst_output = self.analyst.run(translator_output)
        for finding in analyst_output.findings:
            self.storage.findings.upsert(finding, ReviewFinding)
        if self.enable_triage:
            triage_output = self.triage.run(analyst_output)
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

        proposals: list[ImprovementProposal] = []
        if should_run_coach(triage_output):
            proposals.extend(self.coach.run((analyst_output, triage_output)))
        if should_run_architect(triage_output):
            proposals.extend(self.architect.run((analyst_output, triage_output)))

        for proposal in proposals:
            self.storage.proposals.upsert(proposal, ImprovementProposal)

        curated_lessons = self.curator.run(proposals)
        for lesson in curated_lessons:
            self.storage.lessons.upsert(lesson, CuratedLesson)

        experiment_plans = self.experiment_designer.run((curated_lessons, run_bundle))
        for plan in experiment_plans:
            self.storage.experiment_plans.upsert(plan, ExperimentPlan)

        change_sets: list[ChangeSet] = []
        verification_reports: list[VerificationReport] = []
        for plan in experiment_plans:
            change_set = self.applier.run((curated_lessons, plan))
            self.storage.change_sets.upsert(change_set, ChangeSet)
            change_sets.append(change_set)

            report = (
                self.verifier.run((change_set, plan))
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

        promotion_decisions: list[PromotionDecision] = []
        for report in verification_reports:
            decisions = self.promotion_manager.run((report, curated_lessons))
            promotion_decisions.extend(decisions)
        for decision in promotion_decisions:
            self.storage.promotions.upsert(decision, PromotionDecision)

        knowledge_entries = self.librarian.run((promotion_decisions, curated_lessons))
        for entry in knowledge_entries:
            self.storage.knowledge.upsert(entry, KnowledgeEntry)

        history = self.historian.run(
            (run_bundle, analyst_output, curated_lessons, verification_reports, promotion_decisions)
        )

        adversarial_scenarios = (
            self.adversary.run((analyst_output, verification_reports))
            if self.enable_adversary
            else []
        )
        for scenario in adversarial_scenarios:
            self.storage.adversarial_scenarios.upsert(scenario, AdversarialScenario)

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
