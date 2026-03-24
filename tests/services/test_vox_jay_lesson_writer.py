from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from ash_hawk.contracts import CuratedLesson
from ash_hawk.curation.experiment_store import ExperimentStore
from ash_hawk.services.vox_jay_lesson_writer import VoxJayLessonWriter
from ash_hawk.strategies import Strategy, SubStrategy


def _lesson(
    lesson_id: str,
    lesson_type: Literal["policy", "skill", "tool", "harness", "eval"],
    title: str,
    description: str,
    *,
    sub_strategies: list[SubStrategy] | None = None,
    lesson_payload: dict[str, object] | None = None,
    status: Literal["approved", "deprecated", "rolled_back"] = "approved",
) -> CuratedLesson:
    payload = lesson_payload
    if payload is None:
        payload = {"details": title}
        if lesson_type == "policy":
            payload = {
                "rule_name": title,
                "rule_type": "guardrail",
                "condition": "default-condition",
                "action": "default-action",
                "priority": 1,
            }

    return CuratedLesson(
        lesson_id=lesson_id,
        source_proposal_id=f"proposal-{lesson_id}",
        applies_to_agents=["vox-jay"],
        lesson_type=lesson_type,
        title=title,
        description=description,
        lesson_payload=payload,
        validation_status=status,
        version=1,
        created_at=datetime.now(UTC),
        strategy=Strategy.SKILL_QUALITY if lesson_type == "skill" else Strategy.POLICY_QUALITY,
        sub_strategies=sub_strategies or [],
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_apply_writes_mapped_lessons_to_expected_targets(tmp_path: Path) -> None:
    base_path = tmp_path / ".ash-hawk"
    vox_root = tmp_path / "vox-jay"
    store = ExperimentStore(base_path=base_path)
    exp_id = "exp-vox"

    store.store(
        _lesson(
            "lesson-voice",
            "skill",
            "Improve voice tone",
            "Adjust voice and tone guidelines.",
            sub_strategies=[SubStrategy.VOICE_TONE],
        ),
        exp_id,
    )
    store.store(
        _lesson(
            "lesson-playbook",
            "skill",
            "Strengthen playbook",
            "Refine playbook adherence.",
            sub_strategies=[SubStrategy.PLAYBOOK_ADHERENCE],
        ),
        exp_id,
    )
    store.store(
        _lesson("lesson-policy", "policy", "Policy update", "Update policy constraints."),
        exp_id,
    )

    writer = VoxJayLessonWriter(
        experiments_root=base_path / "experiments",
        vox_jay_root=vox_root,
    )
    result = writer.apply(experiment_id=exp_id)

    assert result.approved_lessons_seen == 3
    assert result.lessons_written == 3
    assert result.lessons_skipped_unmapped == 0
    assert result.writes_by_target["src/vox_jay/assets/voice.md"] == 1
    assert result.writes_by_target["src/vox_jay/assets/strategy.md"] == 1
    assert result.writes_by_target["src/vox_jay/assets/policy.md"] == 1

    voice_content = _read(vox_root / "src/vox_jay/assets/voice.md")
    strategy_content = _read(vox_root / "src/vox_jay/assets/strategy.md")
    policy_content = _read(vox_root / "src/vox_jay/assets/policy.md")

    assert "ash-hawk-lesson:lesson-voice" in voice_content
    assert "ash-hawk-lesson:lesson-playbook" in strategy_content
    assert "ash-hawk-lesson:lesson-policy" in policy_content


def test_apply_uses_latest_experiment_by_default(tmp_path: Path) -> None:
    base_path = tmp_path / ".ash-hawk"
    vox_root = tmp_path / "vox-jay"
    store = ExperimentStore(base_path=base_path)

    store.store(_lesson("lesson-older", "policy", "Older", "Old policy lesson."), "exp-older")
    store.store(_lesson("lesson-newer", "policy", "Newer", "New policy lesson."), "exp-newer")

    old_dir = base_path / "experiments" / "exp-older"
    new_dir = base_path / "experiments" / "exp-newer"
    old_ts = 1_700_000_000
    new_ts = 1_800_000_000
    os.utime(old_dir, (old_ts, old_ts))
    os.utime(new_dir, (new_ts, new_ts))

    writer = VoxJayLessonWriter(
        experiments_root=base_path / "experiments",
        vox_jay_root=vox_root,
    )
    result = writer.apply()

    assert result.experiments_read == ["exp-newer"]
    content = _read(vox_root / "src/vox_jay/assets/policy.md")
    assert "ash-hawk-lesson:lesson-newer" in content
    assert "ash-hawk-lesson:lesson-older" not in content


def test_apply_dry_run_does_not_write_files(tmp_path: Path) -> None:
    base_path = tmp_path / ".ash-hawk"
    vox_root = tmp_path / "vox-jay"
    store = ExperimentStore(base_path=base_path)
    exp_id = "exp-vox"
    store.store(
        _lesson("lesson-policy", "policy", "Policy update", "Update policy constraints."), exp_id
    )

    writer = VoxJayLessonWriter(
        experiments_root=base_path / "experiments",
        vox_jay_root=vox_root,
    )
    result = writer.apply(experiment_id=exp_id, dry_run=True)

    assert result.lessons_written == 1
    assert not (vox_root / "src/vox_jay/assets/policy.md").exists()


def test_apply_is_idempotent_and_skips_existing_markers(tmp_path: Path) -> None:
    base_path = tmp_path / ".ash-hawk"
    vox_root = tmp_path / "vox-jay"
    store = ExperimentStore(base_path=base_path)
    exp_id = "exp-vox"
    store.store(
        _lesson("lesson-policy", "policy", "Policy update", "Update policy constraints."), exp_id
    )

    writer = VoxJayLessonWriter(
        experiments_root=base_path / "experiments",
        vox_jay_root=vox_root,
    )

    first_result = writer.apply(experiment_id=exp_id)
    second_result = writer.apply(experiment_id=exp_id)

    assert first_result.lessons_written == 1
    assert first_result.lessons_skipped_existing == 0
    assert second_result.lessons_written == 0
    assert second_result.lessons_skipped_existing == 1


def test_apply_skips_unmapped_lesson_types(tmp_path: Path) -> None:
    base_path = tmp_path / ".ash-hawk"
    vox_root = tmp_path / "vox-jay"
    store = ExperimentStore(base_path=base_path)
    exp_id = "exp-vox"
    store.store(_lesson("lesson-tool", "tool", "Tool lesson", "Tool changes."), exp_id)

    writer = VoxJayLessonWriter(
        experiments_root=base_path / "experiments",
        vox_jay_root=vox_root,
    )
    result = writer.apply(experiment_id=exp_id)

    assert result.approved_lessons_seen == 1
    assert result.lessons_written == 0
    assert result.lessons_skipped_unmapped == 1
    assert result.writes_by_target == {}


def test_apply_policy_write_excludes_diagnostic_payload_json(tmp_path: Path) -> None:
    base_path = tmp_path / ".ash-hawk"
    vox_root = tmp_path / "vox-jay"
    store = ExperimentStore(base_path=base_path)
    exp_id = "exp-vox"

    store.store(
        _lesson(
            "lesson-policy",
            "policy",
            "Policy update",
            "Update policy constraints.",
            lesson_payload={
                "experiment_steps": ["step 1", "step 2"],
                "focus_signals": ["signal 1"],
                "hypothesis": "diagnostic hypothesis",
                "signal_count": 3,
            },
        ),
        exp_id,
    )

    writer = VoxJayLessonWriter(
        experiments_root=base_path / "experiments",
        vox_jay_root=vox_root,
    )
    result = writer.apply(experiment_id=exp_id)

    assert result.lessons_written == 0
    assert result.lessons_skipped_non_actionable == 1
    assert not (vox_root / "src/vox_jay/assets/policy.md").exists()


def test_apply_policy_write_renders_actionable_fields_only(tmp_path: Path) -> None:
    base_path = tmp_path / ".ash-hawk"
    vox_root = tmp_path / "vox-jay"
    store = ExperimentStore(base_path=base_path)
    exp_id = "exp-vox"

    store.store(
        _lesson(
            "lesson-policy",
            "policy",
            "Tool policy_apply failed 1 time(s)",
            "Diagnostic summary that should not be rendered.",
            lesson_payload={
                "rule_name": "NoEmptyPolicyApply",
                "rule_type": "guardrail",
                "condition": "approved_policy_lessons_count == 0",
                "action": "skip_policy_apply_and_emit_warning",
                "priority": 10,
                "focus_signals": ["Tool policy_apply failed"],
            },
        ),
        exp_id,
    )

    writer = VoxJayLessonWriter(
        experiments_root=base_path / "experiments",
        vox_jay_root=vox_root,
    )
    result = writer.apply(experiment_id=exp_id)

    assert result.lessons_written == 1
    content = _read(vox_root / "src/vox_jay/assets/policy.md")
    assert "## Policy Rule Update" in content
    assert "rule_name: `NoEmptyPolicyApply`" in content
    assert "focus_signals" not in content
    assert "Tool policy_apply failed 1 time(s)" not in content
