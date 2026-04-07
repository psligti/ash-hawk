"""Smoke tests for transcript propagation through the research loop.

Regression tests for: _execute_fix must forward evaluation transcripts
to generate_improvement instead of passing an empty list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ash_hawk.research.research_loop import EvaluationSnapshot, ResearchLoop
from ash_hawk.types import EvalTranscript


def _snapshot_with_transcripts(
    transcripts: list[EvalTranscript] | None = None,
) -> EvaluationSnapshot:
    """Helper to build a minimal EvaluationSnapshot."""
    return EvaluationSnapshot(
        mean_score=0.5,
        eval_results={"task-1": 0.5},
        trace_events=[],
        scores={
            "iteration": 0,
            "mean_score": 0.5,
            "score": 0.5,
            "previous_score": 0.5,
            "score_delta": 0.0,
        },
        category_scores={},
        previous_score=0.5,
        score_delta=0.0,
        grader_details=None,
        transcripts=transcripts,
    )


class TestEvaluationSnapshot:
    def test_transcripts_default_none(self) -> None:
        snap = EvaluationSnapshot(
            mean_score=0.0,
            eval_results={},
            trace_events=[],
            scores={},
            category_scores={},
            previous_score=0.0,
            score_delta=0.0,
            grader_details=None,
        )
        assert snap.transcripts is None

    def test_transcripts_stored(self) -> None:
        t = EvalTranscript(messages=[{"role": "user", "content": "hello"}])
        snap = _snapshot_with_transcripts([t])
        assert snap.transcripts is not None
        assert len(snap.transcripts) == 1
        assert snap.transcripts[0].messages[0]["content"] == "hello"


class TestBuildSnapshotTranscripts:
    def test_build_snapshot_stores_transcripts(self, tmp_path: Path) -> None:
        loop = ResearchLoop(storage_path=tmp_path)
        t = EvalTranscript(
            messages=[{"role": "assistant", "content": "did a thing"}],
            tool_calls=[{"name": "bash", "arguments": {"command": "ls"}}],
        )

        snap = loop._build_snapshot(
            mean_score=0.6,
            eval_results={"task-1": 0.6},
            trace_events=[],
            category_scores={"tool_usage": 0.4},
            grader_details=None,
            iteration=0,
            transcripts=[t],
        )

        assert snap.transcripts is not None
        assert len(snap.transcripts) == 1
        assert snap.transcripts[0].tool_calls[0]["name"] == "bash"

    def test_build_snapshot_empty_transcripts(self, tmp_path: Path) -> None:
        loop = ResearchLoop(storage_path=tmp_path)

        snap = loop._build_snapshot(
            mean_score=0.0,
            eval_results={},
            trace_events=[],
            category_scores={},
            grader_details=None,
            iteration=0,
            transcripts=[],
        )

        assert snap.transcripts == []

    def test_build_snapshot_none_transcripts(self, tmp_path: Path) -> None:
        loop = ResearchLoop(storage_path=tmp_path)

        snap = loop._build_snapshot(
            mean_score=0.0,
            eval_results={},
            trace_events=[],
            category_scores={},
            grader_details=None,
            iteration=0,
        )

        assert snap.transcripts == []


class TestExecuteFixTranscriptForwarding:
    @pytest.mark.asyncio
    async def test_execute_fix_uses_evaluation_transcripts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_execute_fix must forward evaluation transcripts to generate_improvement."""
        loop = ResearchLoop(storage_path=tmp_path)
        loop._project_root = tmp_path

        # Set up a target file for the fix to find
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_file = skills_dir / "test-skill.md"
        skill_file.write_text("---\nname: test-skill\n---\n# Test Skill\nDo things.\n")

        # Plant transcripts in the latest evaluation
        transcript = EvalTranscript(
            messages=[{"role": "assistant", "content": "I should use tools"}],
            tool_calls=[],
        )
        loop._latest_eval = _snapshot_with_transcripts([transcript])
        loop._score_history = [0.5]

        received_transcripts: list[list[EvalTranscript]] = []

        async def _fake_generate_improvement(llm_client, current_content, transcripts, **kwargs):
            received_transcripts.append(transcripts)
            return None

        monkeypatch.setattr(
            "ash_hawk.auto_research.llm.generate_improvement",
            _fake_generate_improvement,
        )

        class FakeTarget:
            name = "test-skill"
            target_type = type("TT", (), {"value": "skill"})()

            def read_content(self) -> str:
                return skill_file.read_text()

        class FakeDiscovery:
            def discover_all_targets(self):
                return [FakeTarget()]

        import ash_hawk.auto_research.target_discovery as td_mod

        monkeypatch.setattr(td_mod, "TargetDiscovery", lambda _root: FakeDiscovery())

        # Register a target so _execute_fix picks it up
        from ash_hawk.research.target_registry import TargetEntry, TargetSurface

        loop._target_registry.register(
            TargetEntry(name="test-skill", surface=TargetSurface.PROMPT, description="test")
        )

        from datetime import UTC, datetime

        from ash_hawk.research.diagnosis import DiagnosisReport
        from ash_hawk.research.types import ResearchAction, ResearchDecision

        decision = ResearchDecision(
            action=ResearchAction.FIX,
            rationale="fix",
            target="test-skill",
            expected_info_gain=0.2,
            confidence=0.8,
        )
        diagnosis = DiagnosisReport(
            diagnosis_id="diag-1",
            run_id="run-1",
            timestamp=datetime.now(UTC).isoformat(),
            uncertainty_level=0.2,
            recommended_action="fix",
        )

        result = type("R", (), {"decisions": [], "strategies_promoted": []})()
        await loop._execute_decision(decision, [], tmp_path, 0, diagnosis, result)

        # The key assertion: transcripts were forwarded, not empty
        assert len(received_transcripts) == 1
        assert len(received_transcripts[0]) == 1
        assert received_transcripts[0][0].messages[0]["content"] == "I should use tools"
