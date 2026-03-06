import pytest

from ash_hawk.graders.diff_constraints import DiffConstraintsGrader
from ash_hawk.types import EvalTranscript, EvalTrial, GraderSpec


class TestDiffConstraintsGrader:
    def test_name(self):
        assert DiffConstraintsGrader().name == "diff_constraints"

    @pytest.mark.asyncio
    async def test_grade_passes_with_allowed_paths(self):
        grader = DiffConstraintsGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        diff_text = """
diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1,1 +1,2 @@
-print('old')
+print('old')
+print('new')
""".lstrip()
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "DiffEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {"patch_text": diff_text},
                }
            ]
        )
        spec = GraderSpec(
            grader_type="diff_constraints",
            config={
                "allowed_paths": ["src/*.py"],
                "max_files": 1,
                "max_loc": 5,
                "secret_patterns": [],
            },
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["violations"] == []

    @pytest.mark.asyncio
    async def test_grade_fails_on_secrets_and_disallowed_paths(self):
        grader = DiffConstraintsGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        diff_text = """
diff --git a/secrets.txt b/secrets.txt
index 1111111..2222222 100644
--- a/secrets.txt
+++ b/secrets.txt
@@ -1,1 +1,2 @@
-old
+BEGIN RSA PRIVATE KEY
+AKIA1234567890ABCDEF
""".lstrip()
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "DiffEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {"patch_text": diff_text},
                }
            ]
        )
        spec = GraderSpec(
            grader_type="diff_constraints",
            config={
                "allowed_paths": ["src/*.py"],
                "max_files": 1,
                "max_loc": 10,
            },
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        errors = {violation["error"] for violation in result.details["violations"]}
        assert "disallowed_paths" in errors
        assert "secret_patterns_detected" in errors

    @pytest.mark.asyncio
    async def test_grade_reads_diff_from_tool_calls(self):
        grader = DiffConstraintsGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        diff_text = """
diff --git a/README.md b/README.md
index 1111111..2222222 100644
--- a/README.md
+++ b/README.md
@@ -1,1 +1,2 @@
-old
+old
+new
""".lstrip()
        transcript = EvalTranscript(
            tool_calls=[
                {
                    "tool": "bash",
                    "input": {"command": "git diff"},
                    "result": {"stdout": diff_text},
                }
            ]
        )
        spec = GraderSpec(
            grader_type="diff_constraints",
            config={
                "allowed_paths": ["README.md"],
                "max_files": 1,
                "max_loc": 5,
                "secret_patterns": [],
            },
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.details["diffs_checked"] == 1
