from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ash_hawk.contracts import CuratedLesson, ImprovementProposal, ReviewRequest, RunArtifact
from ash_hawk.curation.persistent_store import PersistentLessonStore
from ash_hawk.pipeline.curator import CurationConfig, CuratorRole, QualityGate


def _make_artifact(run_id: str, agent: str, outcome: str) -> RunArtifact:
    return RunArtifact(
        run_id=run_id,
        agent_name=agent,
        outcome=outcome,
        tool_calls=[],
        steps=[],
        messages=[],
    )


def _make_proposal(proposal_id: str, agent: str, proposal_type: str) -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id=proposal_id,
        origin_run_id=f"run-{proposal_id}",
        target_agent=agent,
        proposal_type=proposal_type,  # type: ignore[arg-type]
        title=f"Test proposal {proposal_id}",
        rationale="Test rationale",
        expected_benefit="Test benefit",
        risk_level="low",
        evidence_refs=["test_evidence_1", "test_evidence_2"],
        created_at=datetime.now(UTC),
    )


class TestParallelTrialIsolation:
    @pytest.mark.asyncio
    async def test_lessons_scoped_by_experiment(self, tmp_path: Path) -> None:
        store = PersistentLessonStore(db_path=tmp_path / "lessons.db")
        await store._init_schema()

        lesson_a = CuratedLesson(
            lesson_id="lesson-exp-a",
            source_proposal_id="prop-a",
            applies_to_agents=["bolt-merlin"],
            lesson_type="policy",
            title="Lesson for Experiment A",
            description="Only applies to experiment A",
            lesson_payload={"rule": "timeout-30"},
            validation_status="approved",
            version=1,
            experiment_id="exp-a",
            created_at=datetime.now(UTC),
        )

        lesson_b = CuratedLesson(
            lesson_id="lesson-exp-b",
            source_proposal_id="prop-b",
            applies_to_agents=["bolt-merlin"],
            lesson_type="policy",
            title="Lesson for Experiment B",
            description="Only applies to experiment B",
            lesson_payload={"rule": "timeout-60"},
            validation_status="approved",
            version=1,
            experiment_id="exp-b",
            created_at=datetime.now(UTC),
        )

        lesson_global = CuratedLesson(
            lesson_id="lesson-global",
            source_proposal_id="prop-global",
            applies_to_agents=["bolt-merlin"],
            lesson_type="policy",
            title="Global Lesson",
            description="Applies to all experiments",
            lesson_payload={"rule": "max-retries-3"},
            validation_status="approved",
            version=1,
            created_at=datetime.now(UTC),
        )

        await store.store(lesson_a)
        await store.store(lesson_b)
        await store.store(lesson_global)

        lessons_for_a = await store.get_for_agent("bolt-merlin", experiment_id="exp-a")
        lessons_for_b = await store.get_for_agent("bolt-merlin", experiment_id="exp-b")
        lessons_global = await store.get_for_agent("bolt-merlin")

        lesson_ids_a = {lesson.lesson_id for lesson in lessons_for_a}
        lesson_ids_b = {lesson.lesson_id for lesson in lessons_for_b}
        lesson_ids_global = {lesson.lesson_id for lesson in lessons_global}

        assert "lesson-exp-a" in lesson_ids_a
        assert "lesson-exp-b" in lesson_ids_b
        assert "lesson-global" in lesson_ids_global
        assert "lesson-global" not in lesson_ids_a
        assert "lesson-global" not in lesson_ids_b

        await store.close()

    @pytest.mark.asyncio
    async def test_concurrent_stores_no_collision(self, tmp_path: Path) -> None:
        db_path = tmp_path / "lessons.db"
        store = PersistentLessonStore(db_path=db_path)

        num_lessons = 10
        tasks = []

        for i in range(num_lessons):
            lesson = CuratedLesson(
                lesson_id=f"lesson-{i}",
                source_proposal_id=f"prop-{i}",
                applies_to_agents=["bolt-merlin"],
                lesson_type="policy",
                title=f"Concurrent Lesson {i}",
                description=f"Lesson {i} from concurrent store",
                lesson_payload={"experiment_id": f"exp-{i % 3}", "index": i},
                validation_status="approved",
                version=1,
                created_at=datetime.now(UTC),
            )
            tasks.append(store.store(lesson))

        await asyncio.gather(*tasks)

        all_lessons = await store.list_all()
        assert len(all_lessons) == num_lessons

        lesson_ids = {lesson.lesson_id for lesson in all_lessons}
        for i in range(num_lessons):
            assert f"lesson-{i}" in lesson_ids

        await store.close()

    @pytest.mark.asyncio
    async def test_unique_proposal_constraint(self, tmp_path: Path) -> None:
        store = PersistentLessonStore(db_path=tmp_path / "lessons.db")
        await store._init_schema()

        lesson = CuratedLesson(
            lesson_id="lesson-1",
            source_proposal_id="prop-unique",
            applies_to_agents=["bolt-merlin"],
            lesson_type="policy",
            title="Unique Proposal Lesson",
            description="Should only exist once",
            lesson_payload={},
            validation_status="approved",
            version=1,
            created_at=datetime.now(UTC),
        )

        await store.store(lesson)

        duplicate = CuratedLesson(
            lesson_id="lesson-2",
            source_proposal_id="prop-unique",
            applies_to_agents=["bolt-merlin"],
            lesson_type="policy",
            title="Duplicate Proposal Lesson",
            description="Should replace original",
            lesson_payload={},
            validation_status="approved",
            version=2,
            created_at=datetime.now(UTC),
        )

        await store.store(duplicate)

        all_lessons = await store.list_all()
        assert len(all_lessons) == 1
        assert all_lessons[0].lesson_id == "lesson-2"
        assert all_lessons[0].version == 2

        await store.close()

    def test_curation_quality_gates_prevent_auto_approval(self) -> None:
        config = CurationConfig(
            auto_approve=True,
            min_confidence=0.8,
            require_evidence=True,
            require_rationale=True,
        )
        curator = CuratorRole(config=config)

        high_quality_proposal = ImprovementProposal(
            proposal_id="prop-high",
            origin_run_id="run-1",
            target_agent="bolt-merlin",
            proposal_type="policy",
            title="High Quality Proposal",
            rationale="This has a clear rationale with evidence",
            expected_benefit="Expected 20% improvement in timeout handling",
            risk_level="low",
            confidence=0.9,
            evidence_refs=["tool_failure_rate", "timeout_distribution"],
            created_at=datetime.now(UTC),
        )

        low_quality_proposal = ImprovementProposal(
            proposal_id="prop-low",
            origin_run_id="run-1",
            target_agent="bolt-merlin",
            proposal_type="policy",
            title="Low Quality Proposal",
            rationale="",
            expected_benefit="",
            risk_level="high",
            confidence=0.5,
            evidence_refs=[],
            created_at=datetime.now(UTC),
        )

        lessons_high = curator.curate([high_quality_proposal])
        lessons_low = curator.curate([low_quality_proposal])

        assert len(lessons_high) == 1
        assert lessons_high[0].source_proposal_id == "prop-high"

        assert len(lessons_low) == 0

    def test_quality_gate_checks(self) -> None:
        gate = QualityGate(
            min_confidence=0.7,
            require_evidence=True,
            require_rationale=True,
            allowed_risk_levels=["low", "medium"],
        )

        good_proposal = ImprovementProposal(
            proposal_id="prop-good",
            origin_run_id="run-1",
            target_agent="bolt-merlin",
            proposal_type="policy",
            title="Good Proposal",
            rationale="Has rationale",
            expected_benefit="Benefit",
            risk_level="low",
            confidence=0.8,
            evidence_refs=["ref1"],
            created_at=datetime.now(UTC),
        )

        bad_proposal = ImprovementProposal(
            proposal_id="prop-bad",
            origin_run_id="run-1",
            target_agent="bolt-merlin",
            proposal_type="policy",
            title="Bad Proposal",
            rationale="",
            expected_benefit="",
            risk_level="high",
            confidence=0.5,
            evidence_refs=[],
            created_at=datetime.now(UTC),
        )

        assert gate.check(good_proposal).passed is True
        assert gate.check(bad_proposal).passed is False

    @pytest.mark.asyncio
    async def test_experiment_isolation_end_to_end(self, tmp_path: Path) -> None:
        store = PersistentLessonStore(db_path=tmp_path / "lessons.db")

        proposals_exp_a = [_make_proposal(f"prop-a-{i}", "bolt-merlin", "policy") for i in range(3)]
        proposals_exp_b = [_make_proposal(f"prop-b-{i}", "iron-rook", "skill") for i in range(3)]

        curator = CuratorRole(config=CurationConfig(auto_approve=True))

        lessons_a = curator.curate(proposals_exp_a)
        lessons_b = curator.curate(proposals_exp_b)

        for lesson in lessons_a:
            lesson.experiment_id = "exp-a"
            await store.store(lesson)

        for lesson in lessons_b:
            lesson.experiment_id = "exp-b"
            await store.store(lesson)

        bolt_merlin_exp_a = await store.get_for_agent("bolt-merlin", experiment_id="exp-a")
        iron_rook_exp_b = await store.get_for_agent("iron-rook", experiment_id="exp-b")

        bolt_merlin_exp_b = await store.get_for_agent("bolt-merlin", experiment_id="exp-b")
        iron_rook_exp_a = await store.get_for_agent("iron-rook", experiment_id="exp-a")

        assert len(bolt_merlin_exp_a) == 3
        assert len(iron_rook_exp_b) == 3

        assert len(bolt_merlin_exp_b) == 0
        assert len(iron_rook_exp_a) == 0

        await store.close()


class TestReviewRequestIsolation:
    def test_review_request_experiment_fields(self) -> None:
        request_a = ReviewRequest(
            run_artifact_id="run-exp-a",
            target_agent="bolt-merlin",
            experiment_id="exp-a",
            variant="control",
        )

        request_b = ReviewRequest(
            run_artifact_id="run-exp-b",
            target_agent="bolt-merlin",
            experiment_id="exp-b",
            variant="treatment",
        )

        assert request_a.experiment_id == "exp-a"
        assert request_b.experiment_id == "exp-b"
        assert request_a.variant == "control"
        assert request_b.variant == "treatment"

    def test_review_request_defaults(self) -> None:
        request = ReviewRequest(
            run_artifact_id="run-1",
            target_agent="bolt-merlin",
        )

        assert request.experiment_id is None
        assert request.variant is None
        assert request.review_mode == "standard"
        assert request.persistence_mode == "propose"
