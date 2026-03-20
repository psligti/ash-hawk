from __future__ import annotations

from pathlib import Path

import pydantic as pd
import pytest

from ash_hawk.improve_cycle.models import (
    EvidenceRef,
    FailureCategory,
    FailureClassification,
    ReviewFinding,
    RolePromptPack,
    RoleRuntimeConfig,
    Severity,
    TriageOutput,
)
from ash_hawk.improve_cycle.prompt_packs import default_prompt_pack


def test_models_forbid_extra_fields() -> None:
    with pytest.raises(pd.ValidationError):
        EvidenceRef.model_validate({"artifact_id": "a", "kind": "k", "extra_field": "x"})


def test_triage_output_schema() -> None:
    triage = TriageOutput(
        primary_cause=FailureClassification(
            category=FailureCategory.MULTI_CAUSAL,
            confidence=0.8,
            rationale="Multiple categories found",
        ),
        primary_owner="both",
    )
    assert triage.primary_owner == "both"


def test_review_finding_requires_enum_severity() -> None:
    finding = ReviewFinding(
        finding_id="f-1",
        title="Title",
        summary="Summary",
        severity=Severity.HIGH,
    )
    assert finding.severity == Severity.HIGH


def test_role_runtime_config_and_prompt_pack() -> None:
    runtime = RoleRuntimeConfig(model_name="deterministic", temperature=0.0)
    pack = RolePromptPack(
        system_prompt_path="system.md",
        task_template_path="task.md",
        rubric_path="rubric.md",
    )
    assert runtime.structured_output_required is True
    assert pack.rubric_path == "rubric.md"


def test_required_role_prompt_packs_exist() -> None:
    required_roles = ["triage", "coach", "architect", "curator", "verifier", "adversary"]
    for role in required_roles:
        pack = default_prompt_pack(role)
        assert Path(pack.system_prompt_path).exists()
        assert Path(pack.task_template_path).exists()
        assert Path(pack.rubric_path).exists()
        for example_path in pack.example_paths:
            assert Path(example_path).exists()


def test_shared_prompt_assets_exist() -> None:
    base = Path("ash_hawk/improve_cycle/shared_prompts")
    assert (base / "failure_taxonomy.md").exists()
    assert (base / "evidence_handling.md").exists()
    assert (base / "proposal_rubric.md").exists()
    assert (base / "risk_rubric.md").exists()
    assert (base / "verification_standards.md").exists()
