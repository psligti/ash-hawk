"""Tests for RollbackManager."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ash_hawk.contracts import CuratedLesson
from ash_hawk.curation.rollback import RollbackManager


@pytest.fixture
def rollback_manager() -> RollbackManager:
    return RollbackManager()


@pytest.fixture
def lesson() -> CuratedLesson:
    return CuratedLesson(
        lesson_id="lesson-rollback-001",
        source_proposal_id="prop-rollback-001",
        applies_to_agents=["test-agent"],
        lesson_type="policy",
        title="Test lesson for rollback",
        description="Testing rollback functionality",
        lesson_payload={"key": "value"},
        validation_status="approved",
        version=1,
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def lesson_v2(lesson: CuratedLesson) -> CuratedLesson:
    return CuratedLesson(
        lesson_id=lesson.lesson_id,
        source_proposal_id=lesson.source_proposal_id,
        applies_to_agents=lesson.applies_to_agents,
        lesson_type=lesson.lesson_type,
        title=lesson.title,
        description="Updated description",
        lesson_payload={"key": "updated_value"},
        validation_status="approved",
        version=2,
        created_at=lesson.created_at,
        updated_at=datetime.now(UTC),
    )


class TestRollbackManagerSnapshot:
    def test_snapshot_creates_history(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        history = rollback_manager.get_history(lesson.lesson_id)
        assert len(history) == 1
        assert history[0].lesson_id == lesson.lesson_id

    def test_snapshot_copies_lesson(self, rollback_manager: RollbackManager, lesson: CuratedLesson):
        rollback_manager.snapshot(lesson)
        history = rollback_manager.get_history(lesson.lesson_id)
        assert history[0] is not lesson

    def test_snapshot_multiple_versions(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson, lesson_v2: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        rollback_manager.snapshot(lesson_v2)
        history = rollback_manager.get_history(lesson.lesson_id)
        assert len(history) == 2

    def test_snapshot_preserves_payload(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        history = rollback_manager.get_history(lesson.lesson_id)
        assert history[0].lesson_payload == lesson.lesson_payload


class TestRollbackManagerRollback:
    def test_rollback_returns_previous_version(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson, lesson_v2: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        rollback_manager.snapshot(lesson_v2)
        rolled_back = rollback_manager.rollback(lesson.lesson_id)
        assert rolled_back is not None
        assert rolled_back.description == lesson.description
        assert rolled_back.lesson_payload == lesson.lesson_payload

    def test_rollback_increments_version(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson, lesson_v2: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        rollback_manager.snapshot(lesson_v2)
        rolled_back = rollback_manager.rollback(lesson.lesson_id)
        assert rolled_back is not None
        assert rolled_back.version == lesson.version + 1

    def test_rollback_sets_rolled_back_status(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson, lesson_v2: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        rollback_manager.snapshot(lesson_v2)
        rolled_back = rollback_manager.rollback(lesson.lesson_id)
        assert rolled_back is not None
        assert rolled_back.validation_status == "rolled_back"

    def test_rollback_sets_rollback_of(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson, lesson_v2: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        rollback_manager.snapshot(lesson_v2)
        rolled_back = rollback_manager.rollback(lesson.lesson_id)
        assert rolled_back is not None
        assert rolled_back.rollback_of == lesson.lesson_id

    def test_rollback_returns_none_with_no_history(self, rollback_manager: RollbackManager):
        rolled_back = rollback_manager.rollback("unknown-lesson")
        assert rolled_back is None

    def test_rollback_returns_none_with_single_snapshot(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        rolled_back = rollback_manager.rollback(lesson.lesson_id)
        assert rolled_back is None

    def test_rollback_to_specific_version(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        lesson_v2 = CuratedLesson(
            lesson_id=lesson.lesson_id,
            source_proposal_id=lesson.source_proposal_id,
            applies_to_agents=lesson.applies_to_agents,
            lesson_type=lesson.lesson_type,
            title=lesson.title,
            description="Version 2",
            lesson_payload={"v": 2},
            validation_status="approved",
            version=2,
            created_at=lesson.created_at,
        )
        lesson_v3 = CuratedLesson(
            lesson_id=lesson.lesson_id,
            source_proposal_id=lesson.source_proposal_id,
            applies_to_agents=lesson.applies_to_agents,
            lesson_type=lesson.lesson_type,
            title=lesson.title,
            description="Version 3",
            lesson_payload={"v": 3},
            validation_status="approved",
            version=3,
            created_at=lesson.created_at,
        )
        rollback_manager.snapshot(lesson_v2)
        rollback_manager.snapshot(lesson_v3)

        rolled_back = rollback_manager.rollback(lesson.lesson_id, target_version=1)
        assert rolled_back is not None
        assert rolled_back.lesson_payload == {"v": 1}

    def test_rollback_adds_to_history(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson, lesson_v2: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        rollback_manager.snapshot(lesson_v2)
        rollback_manager.rollback(lesson.lesson_id)
        history = rollback_manager.get_history(lesson.lesson_id)
        assert len(history) == 3


class TestRollbackManagerGetHistory:
    def test_get_history_returns_copy(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        history1 = rollback_manager.get_history(lesson.lesson_id)
        history2 = rollback_manager.get_history(lesson.lesson_id)
        assert history1 is not history2

    def test_get_history_empty_for_unknown(self, rollback_manager: RollbackManager):
        history = rollback_manager.get_history("unknown-lesson")
        assert history == []


class TestRollbackManagerGetRollbackChain:
    def test_get_rollback_chain_empty(self, rollback_manager: RollbackManager):
        chain = rollback_manager.get_rollback_chain("unknown-lesson")
        assert chain == []

    def test_get_rollback_chain_single_rollback(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson, lesson_v2: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        rollback_manager.snapshot(lesson_v2)
        rolled_back = rollback_manager.rollback(lesson.lesson_id)
        assert rolled_back is not None
        chain = rollback_manager.get_rollback_chain(rolled_back.lesson_id)
        assert lesson.lesson_id in chain

    def test_get_rollback_chain_multiple_rollbacks(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson, lesson_v2: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        rollback_manager.snapshot(lesson_v2)
        rolled_back1 = rollback_manager.rollback(lesson.lesson_id)
        assert rolled_back1 is not None
        rollback_manager.snapshot(rolled_back1)
        rolled_back2 = rollback_manager.rollback(lesson.lesson_id)
        assert rolled_back2 is not None

        chain = rollback_manager.get_rollback_chain(rolled_back2.lesson_id)
        assert len(chain) >= 1


class TestRollbackManagerCanRollback:
    def test_can_rollback_false_with_no_history(self, rollback_manager: RollbackManager):
        assert rollback_manager.can_rollback("unknown-lesson") is False

    def test_can_rollback_false_with_single_snapshot(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        assert rollback_manager.can_rollback(lesson.lesson_id) is False

    def test_can_rollback_true_with_multiple_snapshots(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson, lesson_v2: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        rollback_manager.snapshot(lesson_v2)
        assert rollback_manager.can_rollback(lesson.lesson_id) is True


class TestRollbackManagerEdgeCases:
    def test_rollback_preserves_agents_list(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson
    ):
        rollback_manager.snapshot(lesson)
        lesson_v2 = CuratedLesson(
            lesson_id=lesson.lesson_id,
            source_proposal_id=lesson.source_proposal_id,
            applies_to_agents=["agent-1", "agent-2", "agent-3"],
            lesson_type=lesson.lesson_type,
            title=lesson.title,
            description="Updated",
            lesson_payload={},
            validation_status="approved",
            version=2,
            created_at=lesson.created_at,
        )
        rollback_manager.snapshot(lesson_v2)
        rolled_back = rollback_manager.rollback(lesson.lesson_id)
        assert rolled_back is not None
        assert rolled_back.applies_to_agents == lesson.applies_to_agents

    def test_multiple_lessons_tracked_separately(
        self, rollback_manager: RollbackManager, lesson: CuratedLesson
    ):
        lesson2 = CuratedLesson(
            lesson_id="lesson-rollback-002",
            source_proposal_id="prop-rollback-002",
            applies_to_agents=["test-agent"],
            lesson_type="skill",
            title="Second lesson",
            description="Another lesson",
            lesson_payload={},
            validation_status="approved",
            version=1,
            created_at=datetime.now(UTC),
        )
        rollback_manager.snapshot(lesson)
        rollback_manager.snapshot(lesson2)

        history1 = rollback_manager.get_history(lesson.lesson_id)
        history2 = rollback_manager.get_history(lesson2.lesson_id)

        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0].lesson_id != history2[0].lesson_id
