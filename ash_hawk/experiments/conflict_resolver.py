"""Lesson conflict resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ash_hawk.contracts import CuratedLesson


class ResolutionStrategy(StrEnum):
    """Strategies for resolving lesson conflicts."""

    KEEP_BOTH = "keep_both"
    MERGE = "merge"
    PREFER_NEWER = "prefer_newer"
    PREFER_HIGHER_CONFIDENCE = "prefer_higher_confidence"
    PREFER_EXPLICIT = "prefer_explicit"


@dataclass
class ConflictInfo:
    """Information about a detected conflict."""

    lesson_a: CuratedLesson
    lesson_b: CuratedLesson
    conflict_type: str  # "same_type", "overlapping_applicability", "contradictory_rules"
    severity: str  # "low", "medium", "high"
    details: str = ""


@dataclass
class ResolutionResult:
    """Result of conflict resolution."""

    keep: list[CuratedLesson] = field(default_factory=list)
    merge: list[tuple[CuratedLesson, CuratedLesson]] = field(default_factory=list)
    reject: list[CuratedLesson] = field(default_factory=list)
    conflicts_detected: list[ConflictInfo] = field(default_factory=list)
    resolution_notes: list[str] = field(default_factory=list)


class ConflictResolver:
    """Resolves conflicts between overlapping lessons.

    Detects lessons that apply to the same agent with similar applicability
    and resolves using configurable strategies.
    """

    def __init__(
        self,
        default_strategy: ResolutionStrategy = ResolutionStrategy.PREFER_NEWER,
    ) -> None:
        self._default_strategy = default_strategy

    def detect_conflicts(
        self,
        lessons: list[CuratedLesson],
    ) -> list[ConflictInfo]:
        """Detect all conflicts between lessons."""
        conflicts: list[ConflictInfo] = []

        # Group by agent
        by_agent: dict[str, list[CuratedLesson]] = {}
        for lesson in lessons:
            for agent in lesson.applies_to_agents:
                if agent not in by_agent:
                    by_agent[agent] = []
                by_agent[agent].append(lesson)

        for agent, agent_lessons in by_agent.items():
            # Check for same-type conflicts
            by_type: dict[str, list[CuratedLesson]] = {}
            for lesson in agent_lessons:
                if lesson.lesson_type not in by_type:
                    by_type[lesson.lesson_type] = []
                by_type[lesson.lesson_type].append(lesson)

            for lesson_type, type_lessons in by_type.items():
                if len(type_lessons) > 1:
                    # Check for overlapping applicability
                    for i, lesson_a in enumerate(type_lessons):
                        for lesson_b in type_lessons[i + 1 :]:
                            overlap = self._check_applicability_overlap(lesson_a, lesson_b)
                            if overlap:
                                conflicts.append(
                                    ConflictInfo(
                                        lesson_a=lesson_a,
                                        lesson_b=lesson_b,
                                        conflict_type="overlapping_applicability",
                                        severity="medium",
                                        details=f"Both lessons apply to {agent} with type {lesson_type}",
                                    )
                                )

                            contradiction = self._check_contradiction(lesson_a, lesson_b)
                            if contradiction:
                                conflicts.append(
                                    ConflictInfo(
                                        lesson_a=lesson_a,
                                        lesson_b=lesson_b,
                                        conflict_type="contradictory_rules",
                                        severity="high",
                                        details=contradiction,
                                    )
                                )

        return conflicts

    def _check_applicability_overlap(
        self,
        lesson_a: CuratedLesson,
        lesson_b: CuratedLesson,
    ) -> bool:
        """Check if two lessons have overlapping applicability rules."""
        rules_a = set(lesson_a.applicability_rules)
        rules_b = set(lesson_b.applicability_rules)

        # If both have no rules, they overlap (both apply universally)
        if not rules_a and not rules_b:
            return True

        # Check for any common rules
        return bool(rules_a & rules_b)

    def _check_contradiction(
        self,
        lesson_a: CuratedLesson,
        lesson_b: CuratedLesson,
    ) -> str | None:
        """Check if two lessons have contradictory payloads."""
        payload_a = lesson_a.lesson_payload
        payload_b = lesson_b.lesson_payload

        # Check for timeout contradictions
        if "timeout_override" in payload_a and "timeout_override" in payload_b:
            if payload_a["timeout_override"] != payload_b["timeout_override"]:
                return f"Conflicting timeouts: {payload_a['timeout_override']} vs {payload_b['timeout_override']}"

        # Check for rule contradictions
        if "action" in payload_a and "action" in payload_b:
            if payload_a.get("condition") == payload_b.get("condition"):
                if payload_a["action"] != payload_b["action"]:
                    return f"Same condition but different actions: {payload_a['action']} vs {payload_b['action']}"

        return None

    def resolve(
        self,
        lessons: list[CuratedLesson],
        strategy: ResolutionStrategy | None = None,
    ) -> ResolutionResult:
        """Resolve conflicts in a list of lessons."""
        use_strategy = strategy or self._default_strategy

        keep: list[CuratedLesson] = []
        merge: list[tuple[CuratedLesson, CuratedLesson]] = []
        reject: list[CuratedLesson] = []
        notes: list[str] = []

        # Detect all conflicts
        conflicts = self.detect_conflicts(lessons)

        # Track which lessons have been decided
        decided: set[str] = set()

        # Group conflicts by lesson pairs
        conflict_pairs: dict[tuple[str, str], list[ConflictInfo]] = {}
        for conflict in conflicts:
            key = tuple(sorted([conflict.lesson_a.lesson_id, conflict.lesson_b.lesson_id]))
            if key not in conflict_pairs:
                conflict_pairs[key] = []
            conflict_pairs[key].append(conflict)

        # Resolve each conflict pair
        for (id_a, id_b), pair_conflicts in conflict_pairs.items():
            lesson_a = next(lesson for lesson in lessons if lesson.lesson_id == id_a)
            lesson_b = next(lesson for lesson in lessons if lesson.lesson_id == id_b)

            winner = self._apply_resolution_strategy(
                lesson_a,
                lesson_b,
                use_strategy,
                pair_conflicts,
            )

            if winner == lesson_a:
                if lesson_a.lesson_id not in decided:
                    keep.append(lesson_a)
                    decided.add(lesson_a.lesson_id)
                if lesson_b.lesson_id not in decided:
                    reject.append(lesson_b)
                    decided.add(lesson_b.lesson_id)
                notes.append(
                    f"Kept {lesson_a.lesson_id}, rejected {lesson_b.lesson_id} (strategy: {use_strategy})"
                )
            elif winner == lesson_b:
                if lesson_b.lesson_id not in decided:
                    keep.append(lesson_b)
                    decided.add(lesson_b.lesson_id)
                if lesson_a.lesson_id not in decided:
                    reject.append(lesson_a)
                    decided.add(lesson_a.lesson_id)
                notes.append(
                    f"Kept {lesson_b.lesson_id}, rejected {lesson_a.lesson_id} (strategy: {use_strategy})"
                )
            else:
                # Merge case
                merge.append((lesson_a, lesson_b))
                decided.add(lesson_a.lesson_id)
                decided.add(lesson_b.lesson_id)
                notes.append(f"Marked {lesson_a.lesson_id} and {lesson_b.lesson_id} for merge")

        # Add non-conflicting lessons to keep
        for lesson in lessons:
            if lesson.lesson_id not in decided:
                keep.append(lesson)
                decided.add(lesson.lesson_id)

        return ResolutionResult(
            keep=keep,
            merge=merge,
            reject=reject,
            conflicts_detected=conflicts,
            resolution_notes=notes,
        )

    def _apply_resolution_strategy(
        self,
        lesson_a: CuratedLesson,
        lesson_b: CuratedLesson,
        strategy: ResolutionStrategy,
        conflicts: list[ConflictInfo],
    ) -> CuratedLesson | None:
        """Apply resolution strategy to determine winner. Returns None if merge needed."""
        if strategy == ResolutionStrategy.KEEP_BOTH:
            return lesson_a  # Arbitrarily keep first (both will be kept in practice)

        if strategy == ResolutionStrategy.MERGE:
            return None  # Signal merge needed

        if strategy == ResolutionStrategy.PREFER_NEWER:
            if lesson_a.created_at and lesson_b.created_at:
                return lesson_a if lesson_a.created_at > lesson_b.created_at else lesson_b
            return lesson_a  # Default to first if no timestamps

        if strategy == ResolutionStrategy.PREFER_HIGHER_CONFIDENCE:
            # Use version as proxy for confidence (higher version = more validated)
            if lesson_a.version > lesson_b.version:
                return lesson_a
            elif lesson_b.version > lesson_a.version:
                return lesson_b
            # Fall back to newer
            if lesson_a.created_at and lesson_b.created_at:
                return lesson_a if lesson_a.created_at > lesson_b.created_at else lesson_b
            return lesson_a

        if strategy == ResolutionStrategy.PREFER_EXPLICIT:
            # Prefer lessons with more specific applicability rules
            rules_a = len(lesson_a.applicability_rules)
            rules_b = len(lesson_b.applicability_rules)
            if rules_a > rules_b:
                return lesson_a
            elif rules_b > rules_a:
                return lesson_b
            # Fall back to newer
            if lesson_a.created_at and lesson_b.created_at:
                return lesson_a if lesson_a.created_at > lesson_b.created_at else lesson_b
            return lesson_a

        return lesson_a  # Default

    def merge_lessons(
        self,
        lesson_a: CuratedLesson,
        lesson_b: CuratedLesson,
    ) -> CuratedLesson:
        """Merge two lessons into one. The merged lesson takes the best of both."""
        # Combine applicability rules
        merged_rules = list(set(lesson_a.applicability_rules + lesson_b.applicability_rules))

        # Merge payloads (lesson_b overrides lesson_a for conflicts)
        merged_payload: dict[str, Any] = {
            **lesson_a.lesson_payload,
            **lesson_b.lesson_payload,
        }

        # Use the higher version
        merged_version = max(lesson_a.version, lesson_b.version)

        # Combine agent applicability
        merged_agents = list(set(lesson_a.applies_to_agents + lesson_b.applies_to_agents))

        # Create merged lesson
        return CuratedLesson(
            lesson_id=f"{lesson_a.lesson_id}-merged",
            source_proposal_id=lesson_a.source_proposal_id,
            applies_to_agents=merged_agents,
            lesson_type=lesson_a.lesson_type,
            title=f"[Merged] {lesson_a.title}",
            description=f"Merged from {lesson_a.lesson_id} and {lesson_b.lesson_id}",
            lesson_payload=merged_payload,
            validation_status="approved",
            version=merged_version + 1,
            parent_lesson_id=lesson_a.lesson_id,
            applicability_rules=merged_rules,
            created_at=lesson_a.created_at,
        )
