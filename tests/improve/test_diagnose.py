# type-hygiene: skip-file
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ash_hawk.improve.agentic_diagnoser import (
    _derive_suggested_inspection_paths,
    investigate_trial_with_explorer,
)
from ash_hawk.improve.diagnose import (
    DIAGNOSIS_MESSAGE_LIMIT,
    DIAGNOSIS_TOOL_CALL_LIMIT,
    DIAGNOSIS_TRACE_EVENT_LIMIT,
    _format_transcript_excerpt,
    _parse_diagnosis_response,
    diagnose_failures,
)
from ash_hawk.types import EvalOutcome, EvalStatus, EvalTranscript, EvalTrial, TrialResult


def _make_trial(trial_id: str = "trial-1") -> EvalTrial:
    return EvalTrial(
        id=trial_id,
        task_id="task-1",
        status=EvalStatus.ERROR,
        attempt_number=1,
        input_snapshot="test",
        result=TrialResult(
            trial_id=trial_id,
            outcome=EvalOutcome.failure("agent_error", "err"),
        ),
    )


class TestParseDiagnosis:
    def test_valid_json_response(self):
        response = (
            '{"ideas": ['
            '{"failure_summary": "bad", "root_cause": "bug", '
            '"suggested_fix": "fix it", "target_files": ["a.py"], "confidence": 0.9}'
            "]}"
        )
        result = _parse_diagnosis_response("trial-1", response)
        assert len(result) == 1
        assert result[0].trial_id == "trial-1"
        assert result[0].failure_summary == "bad"
        assert result[0].root_cause == "bug"
        assert result[0].confidence == 0.9

    def test_missing_required_field(self):
        response = '{"failure_summary": "bad"}'
        result = _parse_diagnosis_response("trial-1", response)
        assert result == []

    def test_invalid_json(self):
        result = _parse_diagnosis_response("trial-1", "not json at all")
        assert result == []

    def test_default_confidence(self):
        response = '{"failure_summary": "bad", "root_cause": "bug", "suggested_fix": "fix"}'
        result = _parse_diagnosis_response("trial-1", response)
        assert len(result) == 1
        assert result[0].confidence == 0.0

    def test_assigns_family_and_dedupes_similar_ideas(self):
        response = (
            '{"ideas": ['
            '{"failure_summary": "registered zero tools", '
            '"root_cause": "tool loader returned empty list", '
            '"suggested_fix": "fix loader", '
            '"target_files": ["tools/loader.py"], '
            '"confidence": 0.9}, '
            '{"failure_summary": "loader still empty", '
            '"root_cause": "tool loader returned empty list", '
            '"suggested_fix": "fix loader again", '
            '"target_files": ["tools/loader.py"], '
            '"confidence": 0.8}'
            "]}"
        )

        result = _parse_diagnosis_response("trial-1", response)

        assert len(result) == 1
        assert result[0].family == "tool_loader"

    def test_can_label_explorer_diagnosis_mode(self):
        response = (
            '{"ideas": ['
            '{"failure_summary": "bad", "root_cause": "bug", '
            '"suggested_fix": "fix it", "target_files": ["a.py"], "confidence": 0.9}'
            "]}"
        )
        result = _parse_diagnosis_response("trial-1", response, diagnosis_mode="explorer")
        assert len(result) == 1
        assert result[0].diagnosis_mode == "explorer"

    def test_nested_json_target_files(self):
        response = (
            "Here is my diagnosis:\n"
            '{"ideas": ['
            '{"failure_summary": "bad", "root_cause": "nested issue", '
            '"suggested_fix": "fix it", "target_files": ["a.py", "b.py"], '
            '"anchor_files": ["loader.py"], '
            '"confidence": 0.7, "extra": {"nested": true}}'
            "]}\n"
            "End of analysis."
        )
        result = _parse_diagnosis_response("trial-1", response)
        assert len(result) == 1
        assert result[0].target_files == ["a.py", "b.py"]
        assert result[0].anchor_files == ["loader.py"]
        assert result[0].confidence == 0.7

    def test_ignores_earlier_empty_object_and_prefers_ideas_object(self):
        response = (
            "The trace mentions file_change_index: {} before the real diagnosis.\n\n"
            "```json\n"
            '{"ideas": ['
            '{"failure_summary": "bad", "root_cause": "bug", '
            '"suggested_fix": "fix it", "target_files": ["a.py"], "confidence": 0.9}'
            "]}\n"
            "```"
        )
        result = _parse_diagnosis_response("trial-1", response, diagnosis_mode="explorer")
        assert len(result) == 1
        assert result[0].target_files == ["a.py"]
        assert result[0].diagnosis_mode == "explorer"

    def test_parses_multiple_unique_ideas(self):
        response = (
            '{"ideas": ['
            '{"failure_summary": "bad1", "root_cause": "bug1", '
            '"suggested_fix": "fix1", "target_files": ["a.py"], "confidence": 0.9},'
            '{"failure_summary": "bad2", "root_cause": "bug2", '
            '"suggested_fix": "fix2", "target_files": ["b.py"], "confidence": 0.8}'
            "]}"
        )
        result = _parse_diagnosis_response("trial-1", response)
        assert len(result) == 2
        assert [d.target_files for d in result] == [["a.py"], ["b.py"]]

    def test_dedupes_duplicate_ideas(self):
        response = (
            '{"ideas": ['
            '{"failure_summary": "bad", "root_cause": "bug", '
            '"suggested_fix": "fix1", "target_files": ["a.py"], "confidence": 0.9},'
            '{"failure_summary": "bad", "root_cause": "bug", '
            '"suggested_fix": "fix2", "target_files": ["a.py"], "confidence": 0.8}'
            "]}"
        )
        result = _parse_diagnosis_response("trial-1", response)
        assert len(result) == 1

    def test_drops_overly_broad_target_sets(self):
        response = (
            '{"ideas": ['
            '{"failure_summary": "broad", "root_cause": "bug", '
            '"suggested_fix": "fix many", '
            '"target_files": ["a.py", "b.py", "c.py", "d.py"], '
            '"confidence": 0.9}'
            "]}"
        )

        result = _parse_diagnosis_response("trial-1", response)

        assert result == []


class TestDiagnoseFailures:
    def test_format_transcript_excerpt_uses_expanded_context_windows(self):
        messages = [
            {"role": "assistant", "content": f"message-{index}"}
            for index in range(DIAGNOSIS_MESSAGE_LIMIT + 3)
        ]
        tool_calls = [
            {"name": f"tool-{index}", "arguments": {"index": index}}
            for index in range(DIAGNOSIS_TOOL_CALL_LIMIT + 4)
        ]
        trace_events = [
            {"event_type": f"event-{index}", "data": {"index": index}}
            for index in range(DIAGNOSIS_TRACE_EVENT_LIMIT + 5)
        ]
        trial = _make_trial()
        assert trial.result is not None
        trial.result.transcript = EvalTranscript(
            messages=messages,
            tool_calls=tool_calls,
            trace_events=trace_events,
        )

        transcript_excerpt, tool_calls_excerpt, trace_excerpt = _format_transcript_excerpt(trial)

        assert "message-0" not in transcript_excerpt
        assert f"message-{len(messages) - 1}" in transcript_excerpt
        assert transcript_excerpt.count("[assistant]") == DIAGNOSIS_MESSAGE_LIMIT

        assert "tool-0" not in tool_calls_excerpt
        assert (
            f'tool-{len(tool_calls) - 1}({{"index": {len(tool_calls) - 1}}})' in tool_calls_excerpt
        )
        assert len(tool_calls_excerpt.splitlines()) == DIAGNOSIS_TOOL_CALL_LIMIT

        assert "event-0" not in trace_excerpt
        assert (
            f'event-{len(trace_events) - 1}: {{"index": {len(trace_events) - 1}}}' in trace_excerpt
        )
        assert len(trace_excerpt.splitlines()) == DIAGNOSIS_TRACE_EVENT_LIMIT

    @pytest.mark.asyncio
    async def test_llm_returns_none_returns_fallback(self):
        trial = _make_trial()
        with patch(
            "ash_hawk.improve.diagnose._call_llm", new_callable=AsyncMock, return_value=None
        ):
            results = await diagnose_failures([trial])
        assert len(results) == 1
        d = results[0]
        assert d.trial_id == "trial-1"
        assert d.confidence == 0.1
        assert d.target_files == []
        assert "unavailable" in d.root_cause.lower()
        assert d.actionable is False
        assert d.diagnosis_mode == "fallback_llm_unavailable"
        assert d.degraded_reason == "diagnosis_llm_unavailable"

    @pytest.mark.asyncio
    async def test_unparseable_response_returns_non_actionable_fallback(self):
        trial = _make_trial()
        with patch(
            "ash_hawk.improve.diagnose._call_llm",
            new_callable=AsyncMock,
            return_value="not valid json",
        ):
            results = await diagnose_failures([trial])

        assert len(results) == 1
        d = results[0]
        assert d.actionable is False
        assert d.diagnosis_mode == "fallback_parse_failure"
        assert d.degraded_reason == "diagnosis_parse_failure"

    @pytest.mark.asyncio
    async def test_diagnosis_prompt_includes_transcript_context(self):
        captured_prompt: dict[str, str] = {}
        trial = _make_trial()
        assert trial.result is not None
        trial.result.transcript = EvalTranscript(
            messages=[{"role": "assistant", "content": "I should inspect the repo."}],
            tool_calls=[{"name": "grep", "arguments": {"pattern": "bug"}}],
            trace_events=[{"event_type": "ToolCallEvent", "data": {"name": "grep"}}],
            agent_response="Need more context",
            error_trace="Traceback line",
        )

        async def fake_call_llm(prompt: str) -> str:
            captured_prompt["value"] = prompt
            return (
                '{"ideas": ['
                '{"failure_summary": "bad", "root_cause": "bug", '
                '"suggested_fix": "fix it", "target_files": ["a.py"], "confidence": 0.8}'
                "]}"
            )

        with patch("ash_hawk.improve.diagnose._call_llm", side_effect=fake_call_llm):
            results = await diagnose_failures([trial])

        assert len(results) == 1
        prompt = captured_prompt["value"]
        assert "Transcript Excerpt" in prompt
        assert "I should inspect the repo." in prompt
        assert 'grep({"pattern": "bug"})' in prompt
        assert "ToolCallEvent" in prompt

    @pytest.mark.asyncio
    async def test_diagnosis_prompt_includes_agent_file_manifest(self):
        captured_prompt: dict[str, str] = {}
        trial = _make_trial()

        async def fake_call_llm(prompt: str) -> str:
            captured_prompt["value"] = prompt
            return (
                '{"ideas": ['
                '{"failure_summary": "bad", "root_cause": "bug", '
                '"suggested_fix": "fix it", "target_files": ["coding_agent.py"], "confidence": 0.8}'
                "]}"
            )

        with patch("ash_hawk.improve.diagnose._call_llm", side_effect=fake_call_llm):
            await diagnose_failures(
                [trial],
                agent_content={"coding_agent.py": "x", "tools/edit.py": "y"},
            )

        prompt = captured_prompt["value"]
        assert "Mutable Agent Files" in prompt
        assert "- coding_agent.py" in prompt
        assert "- tools/edit.py" in prompt
        assert "Prefer executable code-path fixes first" in prompt
        assert "shared prompt or skill files" in prompt
        assert "last" in prompt
        assert "resort" in prompt

    @pytest.mark.asyncio
    async def test_flattens_multiple_ideas_from_single_failure(self):
        trial = _make_trial()

        with patch(
            "ash_hawk.improve.diagnose._call_llm",
            new_callable=AsyncMock,
            return_value=(
                '{"ideas": ['
                '{"failure_summary": "bad1", "root_cause": "bug1", '
                '"suggested_fix": "fix1", "target_files": ["a.py"], "confidence": 0.9},'
                '{"failure_summary": "bad2", "root_cause": "bug2", '
                '"suggested_fix": "fix2", "target_files": ["b.py"], "confidence": 0.8}'
                "]}"
            ),
        ):
            results = await diagnose_failures([trial])

        assert len(results) == 2
        assert [d.target_files for d in results] == [["a.py"], ["b.py"]]

    @pytest.mark.asyncio
    async def test_normalizes_repo_relative_target_files_against_agent_manifest(self):
        trial = _make_trial()

        with patch(
            "ash_hawk.improve.diagnose._call_llm",
            new_callable=AsyncMock,
            return_value=(
                '{"ideas": ['
                '{"failure_summary": "bad", "root_cause": "bug", '
                '"suggested_fix": "fix it", '
                '"target_files": ["bolt_merlin/agent/coding_agent.py"], '
                '"confidence": 0.8}'
                "]}"
            ),
        ):
            results = await diagnose_failures([trial], agent_content={"coding_agent.py": "x"})

        assert len(results) == 1
        assert results[0].target_files == ["coding_agent.py"]
        assert results[0].actionable is True

    @pytest.mark.asyncio
    async def test_keeps_agent_root_relative_target_files_against_agent_manifest(self):
        trial = _make_trial()

        with patch(
            "ash_hawk.improve.diagnose._call_llm",
            new_callable=AsyncMock,
            return_value=(
                '{"ideas": ['
                '{"failure_summary": "bad", "root_cause": "bug", '
                '"suggested_fix": "fix it", '
                '"target_files": ["agent/execute.py"], '
                '"anchor_files": ["agent/coding_agent.py"], '
                '"confidence": 0.8}'
                "]}"
            ),
        ):
            results = await diagnose_failures(
                [trial],
                agent_content={"agent/execute.py": "x", "agent/coding_agent.py": "y"},
            )

        assert len(results) == 1
        assert results[0].target_files == ["agent/execute.py"]
        assert results[0].anchor_files == ["agent/coding_agent.py"]
        assert results[0].actionable is True

    @pytest.mark.asyncio
    async def test_keeps_anchored_new_file_under_existing_architecture(self):
        trial = _make_trial()

        with patch(
            "ash_hawk.improve.diagnose._call_llm",
            new_callable=AsyncMock,
            return_value=(
                '{"ideas": ['
                '{"failure_summary": "bad", "root_cause": "bug", '
                '"suggested_fix": "fix it", '
                '"target_files": ["bolt_merlin/agent/tools/verification_retry.py"], '
                '"anchor_files": ["bolt_merlin/agent/tool_dispatcher.py"], '
                '"confidence": 0.8}'
                "]}"
            ),
        ):
            results = await diagnose_failures(
                [trial],
                agent_content={"tool_dispatcher.py": "x", "tools/edit.py": "y"},
            )

        assert len(results) == 1
        assert results[0].target_files == ["tools/verification_retry.py"]
        assert results[0].anchor_files == ["tool_dispatcher.py"]
        assert results[0].actionable is True

    @pytest.mark.asyncio
    async def test_new_file_without_anchor_becomes_non_actionable(self):
        trial = _make_trial()

        with patch(
            "ash_hawk.improve.diagnose._call_llm",
            new_callable=AsyncMock,
            return_value=(
                '{"ideas": ['
                '{"failure_summary": "bad", "root_cause": "bug", '
                '"suggested_fix": "fix it", '
                '"target_files": ["bolt_merlin/agent/tools/verification_retry.py"], '
                '"confidence": 0.8}'
                "]}"
            ),
        ):
            results = await diagnose_failures(
                [trial],
                agent_content={"tool_dispatcher.py": "x", "tools/edit.py": "y"},
            )

        assert len(results) == 1
        assert results[0].actionable is False
        assert results[0].degraded_reason == "diagnosis_new_file_missing_anchor"

    @pytest.mark.asyncio
    async def test_prefers_agentic_diagnosis_when_agent_path_available(self, tmp_path):
        trial = _make_trial()
        agent_dir = tmp_path / "bolt_merlin" / "agent"
        agent_dir.mkdir(parents=True)

        with (
            patch(
                "ash_hawk.improve.diagnose.investigate_trial_with_explorer",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(
                    response=(
                        '{"ideas": ['
                        '{"failure_summary": "agentic", "root_cause": "repo search", '
                        '"suggested_fix": "change runtime", "target_files": ["runtime/x.py"], '
                        '"confidence": 0.9}'
                        "]}"
                    ),
                    error=None,
                    tool_calls_used=2,
                    tool_calls_max=5,
                    file_reads_used=1,
                    file_reads_max=2,
                    search_calls_used=1,
                    search_calls_max=3,
                ),
            ),
            patch(
                "ash_hawk.improve.diagnose._call_llm",
                new_callable=AsyncMock,
                return_value='{"ideas": []}',
            ) as mock_llm,
        ):
            results = await diagnose_failures([trial], agent_path=agent_dir)

        assert len(results) == 1
        assert results[0].target_files == ["runtime/x.py"]
        assert results[0].diagnosis_mode == "explorer"
        assert mock_llm.await_count == 0

    @pytest.mark.asyncio
    async def test_explorer_error_result_falls_back_to_llm(self, tmp_path):
        trial = _make_trial()
        agent_dir = tmp_path / "bolt_merlin" / "agent"
        agent_dir.mkdir(parents=True)

        with (
            patch(
                "ash_hawk.improve.diagnose.investigate_trial_with_explorer",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(
                    response="",
                    error="bolt-merlin CLI timed out after 120.0s",
                ),
            ),
            patch(
                "ash_hawk.improve.diagnose._call_llm",
                new_callable=AsyncMock,
                return_value=(
                    '{"ideas": ['
                    '{"failure_summary": "fallback", "root_cause": "llm path", '
                    '"suggested_fix": "fix it", "target_files": ["a.py"], "confidence": 0.8}'
                    "]}"
                ),
            ) as mock_llm,
        ):
            results = await diagnose_failures([trial], agent_path=agent_dir)

        assert len(results) == 1
        assert results[0].target_files == ["a.py"]
        assert results[0].diagnosis_mode == "llm"
        assert mock_llm.await_count == 1


class TestAgenticDiagnoserPaths:
    @pytest.mark.asyncio
    async def test_external_lessons_dir_does_not_crash_relative_path_handling(self, tmp_path):
        repo_root = tmp_path / "bolt-merlin"
        agent_dir = repo_root / "bolt_merlin" / "agent"
        agent_dir.mkdir(parents=True)
        config_dir = repo_root / ".dawn-kestrel"
        config_dir.mkdir(parents=True)
        (config_dir / "explorer_config.yaml").write_text("tools: [grep]\n", encoding="utf-8")

        external_lessons = tmp_path / "external-lessons"
        external_lessons.mkdir()
        lesson_store = SimpleNamespace(_lessons_dir=external_lessons)

        trial = _make_trial("trial-external-lessons")

        with patch(
            "ash_hawk.improve.patch.run_agent_cli",
            new_callable=AsyncMock,
            return_value=(
                json.dumps({"response": '{"ideas": []}', "metadata": {}}),
                None,
            ),
        ) as mock_run:
            result = await investigate_trial_with_explorer(trial, agent_dir, lesson_store)

        assert result is not None
        assert result.response == '{"ideas": []}'
        prompt = mock_run.await_args.kwargs["prompt"]
        assert str(external_lessons) in prompt
        assert "Prefer executable code-path fixes first" in prompt
        assert "Treat `prompts/*` and `skills/*` as last-resort shared surfaces" in prompt

    @pytest.mark.asyncio
    async def test_personal_preferences_are_injected_into_prompt(self, tmp_path):
        trial = _make_trial("trial-personal-pref")

        with patch(
            "ash_hawk.improve.diagnose._call_llm",
            new_callable=AsyncMock,
            return_value=(
                '{"ideas": ['
                '{"failure_summary": "fallback", "root_cause": "llm path", '
                '"suggested_fix": "fix it", "target_files": ["a.py"], "confidence": 0.8}'
                "]}"
            ),
        ) as mock_llm:
            await diagnose_failures(
                [trial],
                personal_preferences="## Personal Memory\n- prefer_small_changes: true",
            )

        prompt = mock_llm.await_args.args[0]
        assert "## Personal Memory" in prompt
        assert "prefer_small_changes: true" in prompt

    def test_suggested_inspection_paths_prefer_code_paths_before_prompts(self):
        trial = _make_trial("trial-suggested-paths")
        assert trial.result is not None
        trial.result.transcript = EvalTranscript(messages=[], tool_calls=[])

        suggested = _derive_suggested_inspection_paths(trial, ["src/auth.py"])

        assert suggested == [
            "bolt_merlin/agent/execute.py",
            "bolt_merlin/agent/coding_agent.py",
            "bolt_merlin/agent/tool_dispatcher.py",
        ]

    @pytest.mark.asyncio
    async def test_explorer_exception_falls_back_to_llm(self, tmp_path):
        trial = _make_trial()
        agent_dir = tmp_path / "bolt_merlin" / "agent"
        agent_dir.mkdir(parents=True)

        with (
            patch(
                "ash_hawk.improve.diagnose.investigate_trial_with_explorer",
                new_callable=AsyncMock,
                side_effect=RuntimeError("explorer blew up"),
            ),
            patch(
                "ash_hawk.improve.diagnose._call_llm",
                new_callable=AsyncMock,
                return_value=(
                    '{"ideas": ['
                    '{"failure_summary": "fallback", "root_cause": "llm path", '
                    '"suggested_fix": "fix it", "target_files": ["a.py"], "confidence": 0.8}'
                    "]}"
                ),
            ) as mock_llm,
        ):
            results = await diagnose_failures([trial], agent_path=agent_dir)

        assert len(results) == 1
        assert results[0].target_files == ["a.py"]
        assert mock_llm.await_count == 1
