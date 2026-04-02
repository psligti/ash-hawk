from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ash_hawk.research.types import TargetSurface

if TYPE_CHECKING:
    from ash_hawk.auto_research.types import IterationResult

logger = logging.getLogger(__name__)


def _empty_str_list() -> list[str]:
    return []


def _as_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) else default


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _as_list_str(value: object) -> list[str]:
    if isinstance(value, list):
        items = cast(list[object], value)
        return [item for item in items if isinstance(item, str)]
    return []


@dataclass
class StrategyPattern:
    pattern_id: str
    name: str
    description: str
    trigger_condition: str
    action: str
    success_count: int = 0
    total_applications: int = 0
    avg_score_delta: float = 0.0
    affected_surfaces: list[str] = field(default_factory=_empty_str_list)
    status: str = "candidate"
    promoted_at: str | None = None

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_applications if self.total_applications > 0 else 0.0


@dataclass
class PromotedStrategy:
    strategy_id: str
    source_pattern_id: str
    name: str
    description: str
    trigger_condition: str
    action_template: str
    success_rate: float
    affected_surfaces: list[str] = field(default_factory=_empty_str_list)
    promoted_at: str = ""
    artifact_path: str | None = None


class StrategyPromoter:
    """Promotes recurring successful patterns into reusable strategies."""

    PROMOTION_THRESHOLD_SUCCESS_COUNT = 3
    PROMOTION_THRESHOLD_SUCCESS_RATE = 0.7
    PROMOTION_MIN_SURFACES = 2

    def __init__(self, storage_path: Path | None = None) -> None:
        self._patterns: dict[str, StrategyPattern] = {}
        self._promoted: dict[str, PromotedStrategy] = {}
        self._storage_path = storage_path or Path(".ash-hawk/research")

    @classmethod
    def load(cls, storage_path: Path) -> StrategyPromoter:
        """Load patterns and promoted strategies from storage."""
        promoter = cls(storage_path=storage_path)
        file_path = storage_path / "strategies.json"
        if not file_path.exists():
            return promoter

        try:
            raw_data = file_path.read_text(encoding="utf-8")
            raw_value = json.loads(raw_data)
        except Exception:
            logger.exception("Failed to load strategies from %s", file_path)
            return promoter

        if not isinstance(raw_value, dict):
            return promoter

        data = cast(dict[str, object], raw_value)

        patterns_payload = data.get("patterns")
        if isinstance(patterns_payload, dict):
            typed_patterns = cast(dict[str, dict[str, object]], patterns_payload)
            for pattern_id, payload_data in typed_patterns.items():
                pattern = StrategyPattern(
                    pattern_id=pattern_id,
                    name=_as_str(payload_data.get("name")),
                    description=_as_str(payload_data.get("description")),
                    trigger_condition=_as_str(payload_data.get("trigger_condition")),
                    action=_as_str(payload_data.get("action")),
                    success_count=_as_int(payload_data.get("success_count")),
                    total_applications=_as_int(payload_data.get("total_applications")),
                    avg_score_delta=_as_float(payload_data.get("avg_score_delta")),
                    affected_surfaces=_as_list_str(payload_data.get("affected_surfaces")),
                    status=_as_str(payload_data.get("status"), "candidate"),
                    promoted_at=_as_optional_str(payload_data.get("promoted_at")),
                )
                promoter._patterns[pattern_id] = pattern

        promoted_payload = data.get("promoted")
        if isinstance(promoted_payload, dict):
            typed_promoted = cast(dict[str, dict[str, object]], promoted_payload)
            for strategy_id, payload_data in typed_promoted.items():
                strategy = PromotedStrategy(
                    strategy_id=strategy_id,
                    source_pattern_id=_as_str(payload_data.get("source_pattern_id")),
                    name=_as_str(payload_data.get("name")),
                    description=_as_str(payload_data.get("description")),
                    trigger_condition=_as_str(payload_data.get("trigger_condition")),
                    action_template=_as_str(payload_data.get("action_template")),
                    success_rate=_as_float(payload_data.get("success_rate")),
                    affected_surfaces=_as_list_str(payload_data.get("affected_surfaces")),
                    promoted_at=_as_str(payload_data.get("promoted_at")),
                    artifact_path=_as_optional_str(payload_data.get("artifact_path")),
                )
                promoter._promoted[strategy_id] = strategy

        return promoter

    async def save(self) -> None:
        """Persist to {storage_path}/strategies.json via asyncio.to_thread."""
        self._storage_path.mkdir(parents=True, exist_ok=True)
        file_path = self._storage_path / "strategies.json"
        payload = {
            "patterns": {key: asdict(value) for key, value in self._patterns.items()},
            "promoted": {key: asdict(value) for key, value in self._promoted.items()},
        }

        def _write_file() -> None:
            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)

        await asyncio.to_thread(_write_file)

    def detect_patterns(self, iterations: list[IterationResult]) -> list[StrategyPattern]:
        """Detect recurring patterns from iteration results.

        Group iterations by similar improvement_text (first 50 chars as key).
        For each group with 2+ entries:
        - Create a StrategyPattern with the group's aggregate stats
        - If similar pattern already exists, update its counts

        Returns newly detected patterns only.
        """
        grouped: dict[str, list[IterationResult]] = {}
        for iteration in iterations:
            key = iteration.improvement_text[:50].strip()
            if not key:
                continue
            grouped.setdefault(key, []).append(iteration)

        detected: list[StrategyPattern] = []
        surface_values = {surface.value for surface in TargetSurface}

        for key, group in grouped.items():
            if len(group) < 2:
                continue

            pattern_id = f"pattern_{str(abs(hash(key)))[:8]}"
            success_count = sum(1 for it in group if it.applied and it.delta > 0)
            total_applications = sum(1 for it in group if it.applied)
            avg_score_delta = sum(it.delta for it in group) / len(group)
            surfaces: set[str] = set()
            for iteration in group:
                if iteration.category_scores:
                    for surface in iteration.category_scores:
                        if surface in surface_values:
                            surfaces.add(surface)
            if not surfaces:
                surfaces.add(TargetSurface.PROMPT.value)

            if pattern_id in self._patterns:
                existing = self._patterns[pattern_id]
                combined_total = existing.total_applications + total_applications
                combined_delta = (
                    existing.avg_score_delta * existing.total_applications
                    + avg_score_delta * total_applications
                )
                existing.total_applications = combined_total
                existing.success_count += success_count
                existing.avg_score_delta = (
                    combined_delta / combined_total
                    if combined_total > 0
                    else existing.avg_score_delta
                )
                existing.affected_surfaces = sorted(set(existing.affected_surfaces).union(surfaces))
                continue

            name = key
            description = group[0].improvement_text
            pattern = StrategyPattern(
                pattern_id=pattern_id,
                name=name,
                description=description,
                trigger_condition=key,
                action=description,
                success_count=success_count,
                total_applications=total_applications,
                avg_score_delta=avg_score_delta,
                affected_surfaces=sorted(surfaces),
            )
            self._patterns[pattern_id] = pattern
            detected.append(pattern)

        return detected

    def should_promote(self, pattern: StrategyPattern) -> bool:
        """Check if pattern meets promotion criteria:
        - success_count >= 3
        - success_rate > 0.7
        - affected_surfaces spans at least 2 different surfaces (for full design)
        - For MVP: relax the 2-surfaces requirement since we have limited data
        """
        if pattern.success_count < self.PROMOTION_THRESHOLD_SUCCESS_COUNT:
            return False
        if pattern.success_rate < self.PROMOTION_THRESHOLD_SUCCESS_RATE:
            return False
        return True

    async def promote(self, pattern: StrategyPattern) -> PromotedStrategy:
        """Promote a pattern to a reusable strategy.

        1. Create PromotedStrategy from pattern
        2. Save locally under .ash-hawk/research/strategies/
        3. Optionally delegate to KnowledgePromoter for note-lark integration
        4. Update pattern status to 'promoted'
        """
        promoted_at = datetime.now(UTC).isoformat()
        strategy_id = f"strategy_{pattern.pattern_id.split('_')[-1]}"
        strategy = PromotedStrategy(
            strategy_id=strategy_id,
            source_pattern_id=pattern.pattern_id,
            name=pattern.name,
            description=pattern.description,
            trigger_condition=pattern.trigger_condition,
            action_template=pattern.action,
            success_rate=pattern.success_rate,
            affected_surfaces=list(pattern.affected_surfaces),
            promoted_at=promoted_at,
        )

        strategies_dir = self._storage_path / "strategies"
        strategies_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = strategies_dir / f"{strategy.strategy_id}.json"

        def _write_strategy() -> None:
            with open(artifact_path, "w", encoding="utf-8") as handle:
                json.dump(asdict(strategy), handle, indent=2, default=str)

        await asyncio.to_thread(_write_strategy)
        strategy.artifact_path = str(artifact_path)
        self._promoted[strategy.strategy_id] = strategy

        try:
            from ash_hawk.auto_research.knowledge_promotion import KnowledgePromoter
            from ash_hawk.auto_research.types import PromotedLesson, PromotionStatus, TargetType

            promoter = KnowledgePromoter()
            lesson = PromotedLesson(
                lesson_id=strategy.strategy_id,
                improvement_text=pattern.description,
                score_delta=pattern.avg_score_delta,
                target_type=TargetType.POLICY,
                target_name=pattern.name,
                source_experiment="strategy_promotion",
                promotion_status=PromotionStatus.PENDING,
            )
            await promoter.promote_lesson(lesson)
        except ImportError:
            logger.debug("KnowledgePromoter not available; skipping note-lark promotion")
        except Exception:
            logger.exception("Failed to promote strategy %s via KnowledgePromoter", strategy_id)

        pattern.status = "promoted"
        pattern.promoted_at = promoted_at
        await self.save()
        return strategy

    def get_promoted_strategies(self) -> list[PromotedStrategy]:
        """Return all promoted strategies."""
        return list(self._promoted.values())

    def get_candidate_patterns(self) -> list[StrategyPattern]:
        """Return patterns that are candidates for promotion."""
        return [p for p in self._patterns.values() if p.status == "candidate"]
