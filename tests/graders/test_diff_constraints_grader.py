import pytest

from ash_hawk.graders.diff_constraints import DiffConstraintsGrader
from ash_hawk.types import EvalTranscript, EvalTrial, GraderSpec


class TestDiffConstraintsGrader:
    def test_name(self):
        assert DiffConstraintsGrader().name == "diff_constraints"

    @pytest.mark.asyncio
    async def test_passes_when_no_constraints(self):
        """No constraints configured means it always passes."""
        grader = DiffConstraintsGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "DiffEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {
                        "patch_text": "diff --git a/file.py b/file.py\n+++ b/file.py\n+new line"
                    },
                }
            ]
        )
        spec = GraderSpec(grader_type="diff_constraints", config={})

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_passes_when_allowed_paths_satisfied(self):
        """Passes when all changed paths match allowed patterns."""
        grader = DiffConstraintsGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "DiffEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {
                        "patch_text": "diff --git a/src/file.py b/src/file.py\n+++ b/src/file.py\n+new line"
                    },
                }
            ]
        )
        spec = GraderSpec(
            grader_type="diff_constraints",
            config={"allowed_paths": ["src/*", "tests/*"]},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["violations"] == []

    @pytest.mark.asyncio
    async def test_fails_when_disallowed_paths_modified(self):
        """Fails when paths not in allowed_paths are modified."""
        grader = DiffConstraintsGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "DiffEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {
                        "patch_text": "diff --git a/secrets/key.pem b/secrets/key.pem\n+++ b/secrets/key.pem\n+secret"
                    },
                }
            ]
        )
        spec = GraderSpec(
            grader_type="diff_constraints",
            config={"allowed_paths": ["src/*", "tests/*"]},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert len(result.details["violations"]) == 1
        assert result.details["violations"][0]["error"] == "disallowed_paths"
        assert "secrets/key.pem" in result.details["violations"][0]["paths"]

    @pytest.mark.asyncio
    async def test_fails_when_max_files_exceeded(self):
        """Fails when more files changed than max_files allows."""
        grader = DiffConstraintsGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "DiffEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {
                        "patch_text": """diff --git a/file1.py b/file1.py
+++ b/file1.py
+line1
diff --git a/file2.py b/file2.py
+++ b/file2.py
+line2
diff --git a/file3.py b/file3.py
+++ b/file3.py
+line3"""
                    },
                }
            ]
        )
        spec = GraderSpec(
            grader_type="diff_constraints",
            config={"max_files": 2},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert len(result.details["violations"]) == 1
        assert result.details["violations"][0]["error"] == "max_files_exceeded"
        assert result.details["violations"][0]["max_files"] == 2
        assert result.details["violations"][0]["changed_files"] == 3

    @pytest.mark.asyncio
    async def test_passes_when_under_max_files(self):
        """Passes when number of files is under max_files limit."""
        grader = DiffConstraintsGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "DiffEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {
                        "patch_text": """diff --git a/file1.py b/file1.py
+++ b/file1.py
+line1"""
                    },
                }
            ]
        )
        spec = GraderSpec(
            grader_type="diff_constraints",
            config={"max_files": 5},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_fails_when_secrets_detected_default_patterns(self):
        """Fails when default secret patterns (AWS keys, private keys) are detected."""
        grader = DiffConstraintsGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "DiffEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {
                        "patch_text": """diff --git a/config.py b/config.py
+++ b/config.py
+AWS_KEY = "AKIAIOSFODNN7EXAMPLE\""""
                    },
                }
            ]
        )
        spec = GraderSpec(grader_type="diff_constraints", config={})

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert len(result.details["violations"]) == 1
        assert result.details["violations"][0]["error"] == "secret_patterns_detected"

    @pytest.mark.asyncio
    async def test_fails_when_secrets_detected_custom_patterns(self):
        """Fails when custom secret patterns are detected."""
        grader = DiffConstraintsGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "DiffEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {
                        "patch_text": """diff --git a/config.py b/config.py
+++ b/config.py
+MY_SECRET = "super_secret_value_123\""""
                    },
                }
            ]
        )
        spec = GraderSpec(
            grader_type="diff_constraints",
            config={"secret_patterns": [r"super_secret"]},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert len(result.details["violations"]) == 1
        assert result.details["violations"][0]["error"] == "secret_patterns_detected"

    @pytest.mark.asyncio
    async def test_fails_when_max_loc_exceeded(self):
        """Fails when added lines exceed max_loc limit."""
        grader = DiffConstraintsGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "DiffEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {
                        "patch_text": """diff --git a/file.py b/file.py
+++ b/file.py
+line1
+line2
+line3
+line4
+line5"""
                    },
                }
            ]
        )
        spec = GraderSpec(
            grader_type="diff_constraints",
            config={"max_loc": 3},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert len(result.details["violations"]) == 1
        assert result.details["violations"][0]["error"] == "max_loc_exceeded"
        assert result.details["violations"][0]["max_loc"] == 3
        assert result.details["violations"][0]["added_lines"] == 5

    @pytest.mark.asyncio
    async def test_passes_when_under_max_loc(self):
        """Passes when added lines are under max_loc limit."""
        grader = DiffConstraintsGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "DiffEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {
                        "patch_text": """diff --git a/file.py b/file.py
+++ b/file.py
+line1
+line2"""
                    },
                }
            ]
        )
        spec = GraderSpec(
            grader_type="diff_constraints",
            config={"max_loc": 10},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_uses_trial_result_transcript_if_available(self):
        """Uses transcript from trial.result if available."""
        grader = DiffConstraintsGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        result_transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "DiffEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {"patch_text": "diff --git a/file.py b/file.py\n+++ b/file.py\n+new"},
                }
            ]
        )
        from ash_hawk.types import EvalOutcome, EvalStatus, TrialResult

        trial.result = TrialResult(trial_id="t1", outcome=EvalOutcome(status=EvalStatus.COMPLETED), transcript=result_transcript)

        # Empty transcript passed to grade, should use trial.result.transcript
        transcript = EvalTranscript()
        spec = GraderSpec(
            grader_type="diff_constraints",
            config={"max_files": 1},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_collects_diff_from_tool_calls(self):
        """Can collect diff text from tool call results."""
        grader = DiffConstraintsGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            tool_calls=[
                {
                    "name": "run_bash",
                    "result": "diff --git a/file.py b/file.py\n+++ b/file.py\n+line",
                }
            ]
        )
        spec = GraderSpec(
            grader_type="diff_constraints",
            config={"max_files": 1},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.details["diffs_checked"] == 1
