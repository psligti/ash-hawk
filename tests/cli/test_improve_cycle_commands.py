from __future__ import annotations

from datetime import UTC, datetime

import pytest
from click.testing import CliRunner

from ash_hawk.cli.main import cli
from ash_hawk.contracts import CuratedLesson


def test_new_top_level_commands_are_registered() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "lessons" in result.output
    assert "promotions" in result.output
    assert "history" in result.output
    assert "adversary" in result.output


def test_lessons_list_command_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["lessons", "list"])
    assert result.exit_code == 0


def test_improve_rollback_executes_real_deactivation(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()

    class FakeLessonService:
        def deactivate_lesson(self, lesson_id: str) -> CuratedLesson | None:
            return CuratedLesson(
                lesson_id=lesson_id,
                source_proposal_id="proposal-1",
                applies_to_agents=["bolt-merlin"],
                lesson_type="policy",
                title="Rollback target",
                description="",
                lesson_payload={},
                validation_status="rolled_back",
                version=1,
                created_at=datetime.now(UTC),
            )

    monkeypatch.setattr("ash_hawk.services.lesson_service.LessonService", FakeLessonService)

    result = runner.invoke(
        cli,
        ["improve", "rollback", "lesson-123", "--reason", "regression observed"],
    )

    assert result.exit_code == 0
    assert "Lesson rolled back" in result.output
