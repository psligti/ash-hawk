from __future__ import annotations

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


def test_base_role_agent_validate_scope_default() -> None:
    role = AnalystRole()
    assert role.validate_scope({"findings": []}) is True
    assert role.validate_scope(None) is False


def test_base_role_agent_validate_output_default() -> None:
    role = AnalystRole()
    assert role.validate_output({"risk_areas": []}) is True
    assert role.validate_output(None) is False


def test_analyst_validate_scope_rejects_missing_findings() -> None:
    role = AnalystRole()
    assert role.validate_scope({"findings": None}) is True


def test_triage_validate_scope_rejects_empty_categories() -> None:
    role = TriageRole()
    assert role.validate_scope({"findings": []}) is True
    assert role.validate_scope({}) is True


def test_coach_validate_output_requires_proposals() -> None:
    role = CoachRole()
    assert role.validate_output([]) is True
    assert role.validate_output([{"proposal_id": "p-1"}]) is True


def test_architect_validate_output_requires_proposals() -> None:
    role = ArchitectRole()
    assert role.validate_output([{"proposal_id": "a-1"}]) is True


def test_curator_validate_output_requires_lessons() -> None:
    role = CuratorRole()
    assert role.validate_output([]) is True
    assert role.validate_output([{"lesson_id": "l-1"}]) is True


def test_verifier_validate_output_requires_report() -> None:
    role = VerifierRole()
    assert role.validate_output({"passed": True}) is True


def test_competitor_validate_output_requires_comparison() -> None:
    role = CompetitorRole()
    assert role.validate_output({"improved": True}) is True


def test_librarian_validate_output_requires_entries() -> None:
    role = LibrarianRole()
    assert role.validate_output([{"knowledge_id": "k-1"}]) is True


def test_historian_validate_output_requires_summary() -> None:
    role = HistorianRole()
    assert role.validate_output({"promoted_lessons": 1}) is True


def test_adversary_validate_output_requires_scenarios() -> None:
    role = AdversaryRole()
    assert role.validate_output([{"scenario_id": "s-1"}]) is True
