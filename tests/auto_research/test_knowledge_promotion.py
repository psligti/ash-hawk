"""Tests for KnowledgePromoter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ash_hawk.auto_research.knowledge_promotion import KnowledgePromoter, PromotionCriteria
from ash_hawk.auto_research.types import (
    CycleResult,
    IntentPatterns,
    IterationResult,
    PromotedLesson,
    PromotionStatus,
    TargetType,
)


def _make_iteration(
    num: int,
    before: float,
    after: float,
    applied: bool = True,
) -> IterationResult:
    return IterationResult(
        iteration_num=num,
        score_before=before,
        score_after=after,
        improvement_text=f"Iteration {num} improvement",
        applied=applied,
    )


def _make_lesson(
    lesson_id: str = "lesson-1",
    score_delta: float = 0.10,
    target_type: TargetType = TargetType.SKILL,
) -> PromotedLesson:
    return PromotedLesson(
        lesson_id=lesson_id,
        source_experiment="exp-001",
        target_name="test-target",
        target_type=target_type,
        improvement_text="Improved prompt structure for better accuracy",
        score_delta=score_delta,
    )


@pytest.fixture
def cycle_result() -> CycleResult:
    return CycleResult(
        agent_name="test-agent",
        target_path="skills/test",
        target_type=TargetType.SKILL,
        initial_score=0.50,
        final_score=0.70,
    )


@pytest.fixture
def promoter() -> KnowledgePromoter:
    return KnowledgePromoter(note_lark_enabled=False)


class TestShouldPromote:
    """Tests for should_promote decision logic."""

    async def test_rejects_below_min_improvement(
        self, promoter: KnowledgePromoter, cycle_result: CycleResult
    ) -> None:
        iteration = _make_iteration(1, 0.50, 0.51)
        should, reason = await promoter.should_promote(iteration, [iteration], cycle_result)

        assert should is False
        assert "below min_improvement" in reason

    async def test_rejects_unapplied_iteration(
        self, promoter: KnowledgePromoter, cycle_result: CycleResult
    ) -> None:
        iteration = _make_iteration(1, 0.50, 0.60, applied=False)
        should, reason = await promoter.should_promote(iteration, [iteration], cycle_result)

        assert should is False
        assert "not applied" in reason

    async def test_rejects_insufficient_consecutive_successes(
        self, promoter: KnowledgePromoter, cycle_result: CycleResult
    ) -> None:
        iterations = [
            _make_iteration(1, 0.50, 0.56),
            _make_iteration(2, 0.56, 0.62),
        ]
        should, reason = await promoter.should_promote(iterations[-1], iterations, cycle_result)

        assert should is False
        assert "consecutive successes" in reason

    async def test_accepts_with_enough_consecutive_successes(
        self, promoter: KnowledgePromoter, cycle_result: CycleResult
    ) -> None:
        iterations = [
            _make_iteration(1, 0.50, 0.56),
            _make_iteration(2, 0.56, 0.62),
            _make_iteration(3, 0.62, 0.68),
        ]
        should, reason = await promoter.should_promote(iterations[-1], iterations, cycle_result)

        assert should is True
        assert "consecutive" in reason

    async def test_rejects_when_recent_regression_exceeds_max(
        self, cycle_result: CycleResult
    ) -> None:
        promoter = KnowledgePromoter(
            criteria=PromotionCriteria(require_stability=True, max_regression=0.02),
            note_lark_enabled=False,
        )
        iterations = [
            _make_iteration(1, 0.50, 0.56),
            _make_iteration(2, 0.56, 0.53),
            _make_iteration(3, 0.53, 0.59),
            _make_iteration(4, 0.59, 0.65),
            _make_iteration(5, 0.65, 0.71),
        ]
        should, reason = await promoter.should_promote(iterations[-1], iterations, cycle_result)

        assert should is False
        assert "regression" in reason.lower()

    async def test_accepts_when_stability_disabled(self, cycle_result: CycleResult) -> None:
        promoter = KnowledgePromoter(
            criteria=PromotionCriteria(require_stability=False),
            note_lark_enabled=False,
        )
        iterations = [
            _make_iteration(1, 0.50, 0.56),
            _make_iteration(2, 0.56, 0.50),
            _make_iteration(3, 0.50, 0.56),
            _make_iteration(4, 0.56, 0.62),
            _make_iteration(5, 0.62, 0.68),
        ]
        should, reason = await promoter.should_promote(iterations[-1], iterations, cycle_result)

        assert should is True


class TestPromoteLesson:
    """Tests for local lesson promotion."""

    async def test_saves_lesson_locally(self, tmp_path: Path) -> None:
        promoter = KnowledgePromoter(note_lark_enabled=False)
        lesson = _make_lesson()

        with patch.object(promoter, "_save_local", wraps=promoter._save_local) as mock_save:
            mock_save.side_effect = lambda lesson, path: promoter.__class__._save_local(
                promoter, lesson, tmp_path
            )
            result = await promoter.promote_lesson(lesson)

        assert result is True
        assert lesson.promotion_status == PromotionStatus.PROMOTED

    async def test_returns_false_on_local_save_failure(self) -> None:
        promoter = KnowledgePromoter(note_lark_enabled=False)
        lesson = _make_lesson()

        with patch.object(promoter, "_save_local", side_effect=OSError("disk full")):
            result = await promoter.promote_lesson(lesson)

        assert result is False
        assert lesson.promotion_status == PromotionStatus.FAILED

    async def test_attempts_note_lark_after_local_save(self, tmp_path: Path) -> None:
        promoter = KnowledgePromoter(note_lark_enabled=True)
        lesson = _make_lesson()

        with (
            patch.object(promoter, "_save_local", return_value=tmp_path / "lesson-1.json"),
            patch.object(
                promoter, "promote_to_note_lark", new_callable=AsyncMock, return_value="note-abc"
            ) as mock_nl,
        ):
            result = await promoter.promote_lesson(lesson)

        assert result is True
        assert lesson.note_id == "note-abc"
        mock_nl.assert_awaited_once()


class TestSaveLocal:
    """Tests for _save_local file persistence."""

    @pytest.mark.asyncio
    async def test_creates_directory_and_writes_json(self, tmp_path: Path) -> None:
        promoter = KnowledgePromoter(note_lark_enabled=False)
        lesson = _make_lesson(lesson_id="local-test-1")
        storage = tmp_path / "lessons"

        filepath = await promoter._save_local(lesson, storage)

        assert filepath.exists()
        assert filepath.name == "local-test-1.json"
        data = json.loads(filepath.read_text())
        assert data["lesson_id"] == "local-test-1"
        assert data["score_delta"] == 0.10
        assert data["target_type"] == "skill"

    @pytest.mark.asyncio
    async def test_serializes_datetime_as_iso(self, tmp_path: Path) -> None:
        promoter = KnowledgePromoter(note_lark_enabled=False)
        lesson = _make_lesson()
        storage = tmp_path / "lessons"

        filepath = await promoter._save_local(lesson, storage)
        data = json.loads(filepath.read_text())

        assert isinstance(data["promoted_at"], str)
        assert "T" in data["promoted_at"]


class TestPromoteToNoteLark:
    """Tests for note-lark MCP integration."""

    async def test_returns_none_when_disabled(self) -> None:
        promoter = KnowledgePromoter(note_lark_enabled=False)
        lesson = _make_lesson()

        result = await promoter.promote_to_note_lark(lesson)

        assert result is None

    async def test_returns_none_on_import_error(self) -> None:
        promoter = KnowledgePromoter(note_lark_enabled=True)
        lesson = _make_lesson()

        result = await promoter.promote_to_note_lark(lesson)

        assert result is None

    async def test_calls_mcp_with_correct_payload(self) -> None:
        promoter = KnowledgePromoter(note_lark_enabled=True, project_name="test-project")
        lesson = _make_lesson(score_delta=0.15, target_type=TargetType.AGENT)

        mock_mcp = AsyncMock(return_value={"note_id": "note-xyz"})
        with patch.object(promoter, "_call_note_lark_mcp", mock_mcp):
            result = await promoter.promote_to_note_lark(lesson)

        assert result == "note-xyz"
        payload = mock_mcp.call_args[0][0]
        assert payload["memory_type"] == "procedural"
        assert payload["project"] == "test-project"
        assert payload["confidence"] == min(0.95, 0.15 / 0.2)
        assert "auto-research" in payload["tags"]

    async def test_tool_target_maps_to_reference(self) -> None:
        promoter = KnowledgePromoter(note_lark_enabled=True)
        lesson = _make_lesson(target_type=TargetType.TOOL)

        mock_mcp = AsyncMock(return_value={"note_id": "note-ref"})
        with patch.object(promoter, "_call_note_lark_mcp", mock_mcp):
            result = await promoter.promote_to_note_lark(lesson)

        assert result == "note-ref"
        payload = mock_mcp.call_args[0][0]
        assert payload["memory_type"] == "reference"

    async def test_handles_mcp_exception_gracefully(self) -> None:
        promoter = KnowledgePromoter(note_lark_enabled=True)
        lesson = _make_lesson()

        mock_mcp = AsyncMock(side_effect=RuntimeError("MCP unavailable"))
        with patch.object(promoter, "_call_note_lark_mcp", mock_mcp):
            result = await promoter.promote_to_note_lark(lesson)

        assert result is None


class TestMemoryTypeMapping:
    """Tests for target type → memory type mapping."""

    def test_agent_maps_to_procedural(self) -> None:
        assert KnowledgePromoter._get_memory_type(TargetType.AGENT) == "procedural"

    def test_skill_maps_to_procedural(self) -> None:
        assert KnowledgePromoter._get_memory_type(TargetType.SKILL) == "procedural"

    def test_tool_maps_to_reference(self) -> None:
        assert KnowledgePromoter._get_memory_type(TargetType.TOOL) == "reference"


class TestBuildNoteBody:
    """Tests for note body construction."""

    def test_includes_improvement_and_impact(self) -> None:
        lesson = _make_lesson(score_delta=0.12)
        body = KnowledgePromoter._build_note_body(lesson)

        assert "# Improvement" in body
        assert "## Impact" in body
        assert "+0.1200" in body
        assert lesson.target_name in body

    def test_includes_intent_patterns_when_provided(self) -> None:
        lesson = _make_lesson()
        patterns = IntentPatterns(
            inferred_intent="Agent prefers file-based context",
            dominant_tools=["read", "grep"],
        )
        body = KnowledgePromoter._build_note_body(lesson, patterns)

        assert "## Intent Context" in body
        assert "file-based context" in body
        assert "read, grep" in body

    def test_omits_intent_section_when_no_patterns(self) -> None:
        lesson = _make_lesson()
        body = KnowledgePromoter._build_note_body(lesson)

        assert "Intent Context" not in body


class TestConfidenceCalculation:
    """Tests for confidence = min(0.95, score_delta / 0.2)."""

    async def test_confidence_caps_at_095(self) -> None:
        promoter = KnowledgePromoter(note_lark_enabled=True)
        lesson = _make_lesson(score_delta=0.50)

        mock_mcp = AsyncMock(return_value={"note_id": "note-cap"})
        with patch.object(promoter, "_call_note_lark_mcp", mock_mcp):
            await promoter.promote_to_note_lark(lesson)

        payload = mock_mcp.call_args[0][0]
        assert payload["confidence"] == 0.95

    async def test_confidence_proportional_for_small_delta(self) -> None:
        promoter = KnowledgePromoter(note_lark_enabled=True)
        lesson = _make_lesson(score_delta=0.10)

        mock_mcp = AsyncMock(return_value={"note_id": "note-prop"})
        with patch.object(promoter, "_call_note_lark_mcp", mock_mcp):
            await promoter.promote_to_note_lark(lesson)

        payload = mock_mcp.call_args[0][0]
        assert payload["confidence"] == pytest.approx(0.5)
