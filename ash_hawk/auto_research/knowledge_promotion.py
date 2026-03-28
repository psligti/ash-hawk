"""Knowledge promotion for auto-research improvement cycles.

Evaluates whether improvements should be promoted to persistent storage,
saves lessons locally under .ash-hawk/lessons/, and optionally promotes
to note-lark via MCP tools.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ash_hawk.auto_research.types import (
    CycleResult,
    IntentPatterns,
    IterationResult,
    PromotedLesson,
    PromotionStatus,
    TargetType,
)

logger = logging.getLogger(__name__)


@dataclass
class PromotionCriteria:
    """Criteria for deciding whether an improvement should be promoted."""

    min_improvement: float = 0.05
    min_consecutive_successes: int = 3
    max_regression: float = 0.02
    require_stability: bool = True


class KnowledgePromoter:
    """Promotes validated lessons to persistent knowledge storage.

    Evaluates iteration results against promotion criteria, saves lessons
    locally under .ash-hawk/lessons/, and optionally promotes to note-lark
    via MCP tools.
    """

    def __init__(
        self,
        criteria: PromotionCriteria | None = None,
        note_lark_enabled: bool = True,
        project_name: str = "ash-hawk",
    ) -> None:
        self._criteria = criteria or PromotionCriteria()
        self._note_lark_enabled = note_lark_enabled
        self._project_name = project_name

    async def should_promote(
        self,
        iteration: IterationResult,
        all_iterations: list[IterationResult],
        cycle_result: CycleResult,
    ) -> tuple[bool, str]:
        """Determine if an improvement should be promoted.

        Args:
            iteration: The current iteration to evaluate.
            all_iterations: All iterations in the cycle so far.
            cycle_result: The overall cycle result.

        Returns:
            Tuple of (should_promote, reason).
        """
        if iteration.delta < self._criteria.min_improvement:
            return (
                False,
                f"Delta {iteration.delta:.4f} below min_improvement "
                f"{self._criteria.min_improvement:.4f}",
            )

        if not iteration.applied:
            return False, "Iteration was not applied"

        consecutive = self._count_consecutive_successes(all_iterations)
        if consecutive < self._criteria.min_consecutive_successes:
            return (
                False,
                f"Only {consecutive} consecutive successes, "
                f"need {self._criteria.min_consecutive_successes}",
            )

        if self._criteria.require_stability:
            recent_regression = self._find_recent_regression(all_iterations)
            if recent_regression > self._criteria.max_regression:
                return (
                    False,
                    f"Recent regression {recent_regression:.4f} exceeds "
                    f"max_regression {self._criteria.max_regression:.4f}",
                )

        overall_delta = cycle_result.improvement_delta
        return (
            True,
            f"Improvement {iteration.delta:.4f} with {consecutive} consecutive "
            f"successes, overall delta {overall_delta:.4f}",
        )

    async def promote_lesson(
        self,
        lesson: PromotedLesson,
        intent_patterns: IntentPatterns | None = None,
    ) -> bool:
        """Promote a lesson to persistent storage.

        Saves locally first (guaranteed), then attempts note-lark promotion
        if enabled. Local save is the fallback — note-lark failures are logged
        but do not fail the operation.

        Args:
            lesson: The lesson to promote.
            intent_patterns: Optional intent patterns to include in the lesson body.

        Returns:
            True if at least local save succeeded.
        """
        storage_path = Path(".ash-hawk") / "lessons"

        try:
            local_path = self._save_local(lesson, storage_path)
            lesson.promotion_status = PromotionStatus.PROMOTED
            logger.info("Saved lesson %s locally to %s", lesson.lesson_id, local_path)
        except Exception:
            lesson.promotion_status = PromotionStatus.FAILED
            lesson.error_message = "Failed to save locally"
            logger.exception("Failed to save lesson %s locally", lesson.lesson_id)
            return False

        if self._note_lark_enabled:
            note_id = await self.promote_to_note_lark(lesson, intent_patterns)
            if note_id:
                lesson.note_id = note_id
                logger.info(
                    "Promoted lesson %s to note-lark (note_id=%s)",
                    lesson.lesson_id,
                    note_id,
                )

        return True

    async def promote_to_note_lark(
        self,
        lesson: PromotedLesson,
        intent_patterns: IntentPatterns | None = None,
    ) -> str | None:
        """Promote lesson to note-lark knowledge base.

        Args:
            lesson: The lesson to promote.
            intent_patterns: Optional intent patterns to include in the body.

        Returns:
            note_id if successful, None otherwise.
        """
        if not self._note_lark_enabled:
            return None

        confidence = min(0.95, lesson.score_delta / 0.2)
        memory_type = self._get_memory_type(lesson.target_type)

        body = self._build_note_body(lesson, intent_patterns)

        payload: dict[str, Any] = {
            "title": f"Auto-discovered: {lesson.improvement_text[:80]}",
            "memory_type": memory_type,
            "scope": "project",
            "project": self._project_name,
            "status": "validated",
            "confidence": confidence,
            "evidence_count": 1,
            "tags": ["auto-research", "improvement", lesson.target_type.value],
            "body": body,
        }

        try:
            result = await self._call_note_lark_mcp(payload)
            if isinstance(result, dict):
                return result.get("note_id") or result.get("id")
            return str(result) if result else None
        except Exception:
            logger.exception(
                "Failed to promote lesson %s to note-lark",
                lesson.lesson_id,
            )
            return None

    async def _call_note_lark_mcp(self, payload: dict[str, Any]) -> Any:
        """Call note-lark MCP tool. Override in tests or subclasses."""
        try:
            from note_lark_memory_structured import (
                note_lark_memory_structured,
            )

            return await note_lark_memory_structured(payload=payload)
        except ImportError:
            logger.warning("note-lark MCP not available; skipping promotion")
            return None

    def _save_local(self, lesson: PromotedLesson, storage_path: Path) -> Path:
        """Save lesson to local .ash-hawk/lessons/ directory.

        Args:
            lesson: The lesson to save.
            storage_path: Base path for lesson storage.

        Returns:
            Path to the saved lesson file.
        """
        storage_path.mkdir(parents=True, exist_ok=True)

        filename = f"{lesson.lesson_id}.json"
        filepath = storage_path / filename

        lesson_data = asdict(lesson)
        for key, value in lesson_data.items():
            if isinstance(value, datetime):
                lesson_data[key] = value.isoformat()

        with open(filepath, "w") as f:
            json.dump(lesson_data, f, indent=2, default=str)

        return filepath

    def _count_consecutive_successes(
        self,
        iterations: list[IterationResult],
    ) -> int:
        """Count consecutive applied iterations with positive delta from the end."""
        count = 0
        for it in reversed(iterations):
            if it.applied and it.delta > 0:
                count += 1
            else:
                break
        return count

    def _find_recent_regression(
        self,
        iterations: list[IterationResult],
        window: int = 5,
    ) -> float:
        """Find the worst regression in the recent window.

        Returns:
            The magnitude of the worst regression (positive value), or 0.0.
        """
        recent = iterations[-window:] if len(iterations) >= window else iterations
        worst = 0.0
        for it in recent:
            if it.delta < 0:
                worst = max(worst, abs(it.delta))
        return worst

    @staticmethod
    def _get_memory_type(target_type: TargetType) -> str:
        """Map target type to note-lark memory_type.

        AGENT/SKILL → "procedural", TOOL → "reference".
        """
        if target_type in (TargetType.AGENT, TargetType.SKILL):
            return "procedural"
        return "reference"

    @staticmethod
    def _build_note_body(
        lesson: PromotedLesson,
        intent_patterns: IntentPatterns | None = None,
    ) -> str:
        """Build the markdown body for a note-lark promotion."""
        sections: list[str] = [
            f"# Improvement\n\n{lesson.improvement_text}",
            f"## Impact\n\n- Score delta: {lesson.score_delta:+.4f}\n"
            f"- Target: {lesson.target_name} ({lesson.target_type.value})\n"
            f"- Source experiment: {lesson.source_experiment}",
        ]

        if intent_patterns and intent_patterns.inferred_intent:
            intent_section = f"## Intent Context\n\n{intent_patterns.inferred_intent}"
            if intent_patterns.dominant_tools:
                tools_str = ", ".join(intent_patterns.dominant_tools)
                intent_section += f"\n\n**Dominant tools:** {tools_str}"
            sections.append(intent_section)

        return "\n\n".join(sections)
