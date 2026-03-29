"""Tests for LessonStore."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ash_hawk.contracts import CuratedLesson
from ash_hawk.curation.store import LessonStore


@pytest.fixture
def store(tmp_path) -> LessonStore:
    return LessonStore(storage_path=tmp_path / "lessons")


@pytest.fixture
def lesson() -> CuratedLesson:
    return CuratedLesson(
        lesson_id="lesson-store-001",
        source_proposal_id="prop-store-001",
        applies_to_agents=["agent-alpha", "agent-beta"],
        lesson_type="policy",
        title="Store test lesson",
        description="Test lesson for store",
        lesson_payload={"key": "value"},
        validation_status="approved",
        version=1,
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def lesson2() -> CuratedLesson:
    return CuratedLesson(
        lesson_id="lesson-store-002",
        source_proposal_id="prop-store-002",
        applies_to_agents=["agent-alpha"],
        lesson_type="skill",
        title="Second lesson",
        description="Another lesson",
        lesson_payload={"other": "data"},
        validation_status="approved",
        version=1,
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def deprecated_lesson() -> CuratedLesson:
    return CuratedLesson(
        lesson_id="lesson-deprecated-001",
        source_proposal_id="prop-deprecated-001",
        applies_to_agents=["agent-gamma"],
        lesson_type="policy",
        title="Deprecated lesson",
        description="This is deprecated",
        lesson_payload={},
        validation_status="deprecated",
        version=1,
        created_at=datetime.now(UTC),
    )


class TestLessonStoreStore:
    def test_store_returns_lesson_id(self, store: LessonStore, lesson: CuratedLesson):
        lesson_id = store.store(lesson)
        assert lesson_id == lesson.lesson_id

    def test_store_indexes_by_proposal(self, store: LessonStore, lesson: CuratedLesson):
        store.store(lesson)
        retrieved = store.get_by_proposal(lesson.source_proposal_id)
        assert retrieved is not None
        assert retrieved.lesson_id == lesson.lesson_id

    def test_store_indexes_by_agent(self, store: LessonStore, lesson: CuratedLesson):
        store.store(lesson)
        for agent in lesson.applies_to_agents:
            agent_lessons = store.get_for_agent(agent)
            assert any(stored.lesson_id == lesson.lesson_id for stored in agent_lessons)

    def test_store_multiple_lessons(
        self, store: LessonStore, lesson: CuratedLesson, lesson2: CuratedLesson
    ):
        store.store(lesson)
        store.store(lesson2)
        all_lessons = store.list_all()
        assert len(all_lessons) == 2


class TestLessonStoreGet:
    def test_get_returns_stored_lesson(self, store: LessonStore, lesson: CuratedLesson):
        store.store(lesson)
        retrieved = store.get(lesson.lesson_id)
        assert retrieved is not None
        assert retrieved.lesson_id == lesson.lesson_id

    def test_get_returns_none_for_unknown_id(self, store: LessonStore):
        retrieved = store.get("nonexistent-lesson")
        assert retrieved is None

    def test_get_by_proposal_returns_lesson(self, store: LessonStore, lesson: CuratedLesson):
        store.store(lesson)
        retrieved = store.get_by_proposal(lesson.source_proposal_id)
        assert retrieved is not None
        assert retrieved.source_proposal_id == lesson.source_proposal_id

    def test_get_by_proposal_returns_none_for_unknown(self, store: LessonStore):
        retrieved = store.get_by_proposal("nonexistent-proposal")
        assert retrieved is None


class TestLessonStoreGetForAgent:
    def test_get_for_agent_returns_approved_lessons(
        self, store: LessonStore, lesson: CuratedLesson
    ):
        store.store(lesson)
        agent_lessons = store.get_for_agent("agent-alpha")
        assert len(agent_lessons) == 1
        assert agent_lessons[0].lesson_id == lesson.lesson_id

    def test_get_for_agent_excludes_deprecated(
        self, store: LessonStore, deprecated_lesson: CuratedLesson
    ):
        store.store(deprecated_lesson)
        agent_lessons = store.get_for_agent("agent-gamma")
        assert len(agent_lessons) == 0

    def test_get_for_agent_returns_empty_for_unknown(self, store: LessonStore):
        agent_lessons = store.get_for_agent("unknown-agent")
        assert agent_lessons == []

    def test_get_for_agent_multiple_lessons(
        self, store: LessonStore, lesson: CuratedLesson, lesson2: CuratedLesson
    ):
        store.store(lesson)
        store.store(lesson2)
        agent_lessons = store.get_for_agent("agent-alpha")
        assert len(agent_lessons) == 2


class TestLessonStoreListAll:
    def test_list_all_returns_all_lessons(
        self, store: LessonStore, lesson: CuratedLesson, lesson2: CuratedLesson
    ):
        store.store(lesson)
        store.store(lesson2)
        all_lessons = store.list_all()
        assert len(all_lessons) == 2

    def test_list_all_filters_by_status(
        self, store: LessonStore, lesson: CuratedLesson, deprecated_lesson: CuratedLesson
    ):
        store.store(lesson)
        store.store(deprecated_lesson)
        approved = store.list_all(status="approved")
        assert len(approved) == 1
        assert approved[0].validation_status == "approved"

    def test_list_all_filters_by_type(
        self, store: LessonStore, lesson: CuratedLesson, lesson2: CuratedLesson
    ):
        store.store(lesson)
        store.store(lesson2)
        policies = store.list_all(lesson_type="policy")
        assert len(policies) == 1
        assert policies[0].lesson_type == "policy"

    def test_list_all_empty_store(self, store: LessonStore):
        all_lessons = store.list_all()
        assert all_lessons == []


class TestLessonStoreUpdateStatus:
    def test_update_status_changes_status(self, store: LessonStore, lesson: CuratedLesson):
        store.store(lesson)
        updated = store.update_status(lesson.lesson_id, "deprecated")
        assert updated is not None
        assert updated.validation_status == "deprecated"

    def test_update_status_sets_updated_at(self, store: LessonStore, lesson: CuratedLesson):
        store.store(lesson)
        updated = store.update_status(lesson.lesson_id, "deprecated")
        assert updated is not None
        assert updated.updated_at is not None

    def test_update_status_preserves_other_fields(self, store: LessonStore, lesson: CuratedLesson):
        store.store(lesson)
        updated = store.update_status(lesson.lesson_id, "rolled_back")
        assert updated is not None
        assert updated.lesson_id == lesson.lesson_id
        assert updated.title == lesson.title
        assert updated.version == lesson.version

    def test_update_status_returns_none_for_unknown(self, store: LessonStore):
        updated = store.update_status("nonexistent", "deprecated")
        assert updated is None

    def test_update_status_to_rolled_back(self, store: LessonStore, lesson: CuratedLesson):
        store.store(lesson)
        updated = store.update_status(lesson.lesson_id, "rolled_back")
        assert updated is not None
        assert updated.validation_status == "rolled_back"


class TestLessonStoreDelete:
    def test_delete_removes_lesson(self, store: LessonStore, lesson: CuratedLesson):
        store.store(lesson)
        result = store.delete(lesson.lesson_id)
        assert result is True
        assert store.get(lesson.lesson_id) is None

    def test_delete_removes_from_proposal_index(self, store: LessonStore, lesson: CuratedLesson):
        store.store(lesson)
        store.delete(lesson.lesson_id)
        assert store.get_by_proposal(lesson.source_proposal_id) is None

    def test_delete_removes_from_agent_index(self, store: LessonStore, lesson: CuratedLesson):
        store.store(lesson)
        store.delete(lesson.lesson_id)
        for agent in lesson.applies_to_agents:
            agent_lessons = store.get_for_agent(agent)
            assert not any(stored.lesson_id == lesson.lesson_id for stored in agent_lessons)

    def test_delete_returns_false_for_unknown(self, store: LessonStore):
        result = store.delete("nonexistent-lesson")
        assert result is False

    def test_delete_multiple_lessons(
        self, store: LessonStore, lesson: CuratedLesson, lesson2: CuratedLesson
    ):
        store.store(lesson)
        store.store(lesson2)
        store.delete(lesson.lesson_id)
        assert len(store.list_all()) == 1
