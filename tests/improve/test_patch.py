# type-hygiene: skip-file
from __future__ import annotations

from ash_hawk.improve.diagnose import Diagnosis
from ash_hawk.improve.patch import (
    ProposedPatch,
    _classify_cli_error,
    _extract_execution_metrics,
    _sanitize_path,
    write_patch,
)


def _make_patch(trial_id: str = "trial-1", file_path: str = "src/main.py") -> ProposedPatch:
    return ProposedPatch(
        diagnosis=Diagnosis(
            trial_id=trial_id,
            failure_summary="test failure",
            root_cause="bug",
            suggested_fix="fix it",
            target_files=[file_path],
        ),
        file_path=file_path,
        description="Fix the bug",
        diff="--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-old\n+new",
        rationale="The old code was wrong",
    )


class TestSanitizePath:
    def test_normal_name(self):
        assert _sanitize_path("trial-1") == "trial_1"

    def test_path_traversal(self):
        assert _sanitize_path("../../../etc/passwd") == "_________etc_passwd"

    def test_backslash(self):
        assert _sanitize_path("foo\\bar") == "foo_bar"


class TestWritePatch:
    def test_creates_file(self, tmp_path):
        patch = _make_patch()
        path = write_patch(patch, output_dir=tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "trial-1" in content
        assert "src/main.py" in content

    def test_creates_output_dir(self, tmp_path):
        output = tmp_path / "nested" / "dir"
        patch = _make_patch()
        path = write_patch(patch, output_dir=output)
        assert path.exists()
        assert output.exists()

    def test_sanitized_filename(self, tmp_path):
        patch = _make_patch(trial_id="../../etc/passwd")
        path = write_patch(patch, output_dir=tmp_path)
        assert ".." not in path.name
        assert path.parent == tmp_path


class TestExecutionMetrics:
    def test_extracts_tool_and_llm_metrics(self):
        stdout = "\n".join(
            [
                "Registered tool: read",
                "Registered tool: edit",
                "dawn_kestrel.provider.llm_client._collect_stream_events_once completed in 12.5s",
                "dawn_kestrel.provider.llm_client._collect_stream_events_once completed in 100.0s",
            ]
        )
        stderr = "Creating virtual environment at: .venv"

        metrics = _extract_execution_metrics(stdout, stderr)

        assert metrics.registered_tool_count == 2
        assert metrics.llm_completion_count == 2
        assert metrics.max_llm_completion_seconds == 100.0
        assert metrics.long_llm_completion_count == 1
        assert metrics.created_virtualenv is True


class TestMutationFailureClassification:
    def test_classifies_timeout(self):
        assert (
            _classify_cli_error("bolt-merlin CLI timed out after 360.0s") == "mutation_cli_timeout"
        )

    def test_classifies_zero_tools(self):
        assert (
            _classify_cli_error("mutation subprocess registered zero tools")
            == "mutation_zero_tools"
        )

    def test_classifies_generic_cli_error(self):
        assert _classify_cli_error("exit code 1") == "mutation_cli_error"
