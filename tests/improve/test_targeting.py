from __future__ import annotations

from ash_hawk.improve.diagnose import Diagnosis
from ash_hawk.improve.targeting import diagnosis_targets_allowed, validate_diagnosis_targets


def _diagnosis(
    *,
    target_files: list[str],
    anchor_files: list[str] | None = None,
) -> Diagnosis:
    return Diagnosis(
        trial_id="trial-1",
        failure_summary="failed",
        root_cause="bug",
        suggested_fix="fix it",
        target_files=target_files,
        anchor_files=anchor_files or [],
        confidence=0.8,
    )


class TestDiagnosisTargetsAllowed:
    def test_agent_root_relative_existing_file_is_allowed(self) -> None:
        diagnosis = _diagnosis(target_files=["agent/execute.py"])

        validated, rejected = validate_diagnosis_targets(
            diagnosis,
            {"agent/execute.py", "agent/coding_agent.py"},
        )

        assert rejected == []
        assert diagnosis_targets_allowed(validated, {"agent/execute.py", "agent/coding_agent.py"})
        assert validated.target_files == ["agent/execute.py"]
        assert validated.actionable is True

    def test_unanchored_new_file_is_not_allowed(self) -> None:
        diagnosis = _diagnosis(target_files=["tools/verification_retry.py"])

        assert (
            diagnosis_targets_allowed(diagnosis, {"tool_dispatcher.py", "tools/edit.py"}) is False
        )

    def test_anchored_new_file_matches_validation_rule(self) -> None:
        diagnosis = _diagnosis(
            target_files=["bolt_merlin/agent/tools/verification_retry.py"],
            anchor_files=["bolt_merlin/agent/tool_dispatcher.py"],
        )

        allowed_files = {"tool_dispatcher.py", "tools/edit.py"}
        validated, _ = validate_diagnosis_targets(diagnosis, allowed_files)

        assert diagnosis_targets_allowed(validated, allowed_files) is True
        assert validated.target_files == ["tools/verification_retry.py"]

    def test_anchored_new_file_with_agent_prefixed_manifest_is_allowed(self) -> None:
        diagnosis = _diagnosis(
            target_files=["agent/tools/verification_retry.py"],
            anchor_files=["agent/tool_dispatcher.py"],
        )

        allowed_files = {"agent/tool_dispatcher.py", "agent/tools/edit.py"}
        validated, rejected = validate_diagnosis_targets(diagnosis, allowed_files)

        assert rejected == []
        assert diagnosis_targets_allowed(validated, allowed_files) is True
        assert validated.target_files == ["agent/tools/verification_retry.py"]
        assert validated.anchor_files == ["agent/tool_dispatcher.py"]
