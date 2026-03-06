from __future__ import annotations

import fnmatch
import re
from typing import Any

from ash_hawk.graders.base import Grader
from ash_hawk.types import EvalTranscript, EvalTrial, GraderResult, GraderSpec


class DiffConstraintsGrader(Grader):
    _default_secret_patterns = [
        r"BEGIN (?:RSA|OPENSSH) PRIVATE KEY",
        r"AKIA[0-9A-Z]{16}",
    ]

    @property
    def name(self) -> str:
        return "diff_constraints"

    def _is_diff_text(self, text: str) -> bool:
        stripped = text.lstrip()
        return (
            stripped.startswith("diff ")
            or "diff --git" in text
            or ("+++ " in text and "--- " in text)
        )

    def _normalize_path(self, path: str) -> str:
        cleaned = path.strip()
        if cleaned.startswith("a/") or cleaned.startswith("b/"):
            return cleaned[2:]
        return cleaned

    def _collect_tool_diff_texts(self, tool_call: dict[str, Any]) -> list[str]:
        candidates: list[str] = []

        for key in ("result", "output", "response"):
            value = tool_call.get(key)
            if isinstance(value, str):
                candidates.append(value)
            elif isinstance(value, dict):
                stdout = value.get("stdout")
                if isinstance(stdout, str):
                    candidates.append(stdout)

        stdout_top = tool_call.get("stdout")
        if isinstance(stdout_top, str):
            candidates.append(stdout_top)

        return [text for text in candidates if self._is_diff_text(text)]

    def _collect_diff_texts(self, transcript: EvalTranscript) -> list[str]:
        diff_texts: list[str] = []

        for event in transcript.trace_events or []:
            if not isinstance(event, dict):
                continue
            if event.get("event_type") != "DiffEvent":
                continue
            data = event.get("data", {})
            if not isinstance(data, dict):
                continue
            patch_text = data.get("patch_text")
            if isinstance(patch_text, str) and patch_text.strip():
                diff_texts.append(patch_text)

        for tool_call in transcript.tool_calls or []:
            if not isinstance(tool_call, dict):
                continue
            diff_texts.extend(self._collect_tool_diff_texts(tool_call))

        return diff_texts

    def _parse_diff_texts(self, diff_texts: list[str]) -> tuple[set[str], int]:
        changed_paths: set[str] = set()
        added_lines = 0

        for diff_text in diff_texts:
            for line in diff_text.splitlines():
                if line.startswith("diff --git "):
                    parts = line.split()
                    if len(parts) >= 4:
                        path = self._normalize_path(parts[3])
                        if path:
                            changed_paths.add(path)
                    continue
                if line.startswith("+++ "):
                    path = line[4:].strip()
                    if path and path != "/dev/null":
                        path = self._normalize_path(path)
                        if path:
                            changed_paths.add(path)
                    continue
                if line.startswith("+") and not line.startswith("+++"):
                    added_lines += 1

        return changed_paths, added_lines

    def _find_secret_matches(
        self, diff_texts: list[str], patterns: list[str]
    ) -> list[dict[str, str]]:
        matches: list[dict[str, str]] = []
        compiled_patterns: list[tuple[str, re.Pattern[str]]] = []
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            try:
                compiled_patterns.append((pattern, re.compile(pattern)))
            except re.error:
                matches.append({"pattern": pattern, "match": "invalid_regex"})

        for diff_text in diff_texts:
            for pattern, compiled in compiled_patterns:
                match = compiled.search(diff_text)
                if match:
                    matches.append({"pattern": pattern, "match": match.group(0)})

        return matches

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        effective_transcript = transcript
        if trial.result is not None:
            effective_transcript = trial.result.transcript

        config = spec.config
        allowed_paths_raw = config.get("allowed_paths", [])
        allowed_paths = [path for path in allowed_paths_raw if isinstance(path, str)]
        max_files = config.get("max_files")
        max_loc = config.get("max_loc")
        secret_patterns_raw = config.get("secret_patterns", self._default_secret_patterns)
        secret_patterns = [pattern for pattern in secret_patterns_raw if isinstance(pattern, str)]

        diff_texts = self._collect_diff_texts(effective_transcript)
        changed_paths, added_lines = self._parse_diff_texts(diff_texts)

        constraints_enabled = any(
            [
                allowed_paths,
                max_files is not None,
                max_loc is not None,
                secret_patterns,
            ]
        )

        violations: list[dict[str, Any]] = []
        if constraints_enabled and not diff_texts:
            violations.append({"error": "missing_diff"})

        if allowed_paths:
            disallowed = [
                path
                for path in sorted(changed_paths)
                if not any(fnmatch.fnmatch(path, pattern) for pattern in allowed_paths)
            ]
            if disallowed:
                violations.append(
                    {
                        "error": "disallowed_paths",
                        "paths": disallowed,
                    }
                )

        if max_files is not None and len(changed_paths) > max_files:
            violations.append(
                {
                    "error": "max_files_exceeded",
                    "max_files": max_files,
                    "changed_files": len(changed_paths),
                }
            )

        if max_loc is not None and added_lines > max_loc:
            violations.append(
                {
                    "error": "max_loc_exceeded",
                    "max_loc": max_loc,
                    "added_lines": added_lines,
                }
            )

        secret_matches = self._find_secret_matches(diff_texts, secret_patterns)
        if secret_matches:
            violations.append(
                {
                    "error": "secret_patterns_detected",
                    "matches": secret_matches,
                }
            )

        passed = not violations
        score = 1.0 if passed else 0.0

        return GraderResult(
            grader_type=self.name,
            score=score,
            passed=passed,
            details={
                "diffs_checked": len(diff_texts),
                "changed_files": len(changed_paths),
                "added_lines": added_lines,
                "violations": violations,
                "limits": {"max_files": max_files, "max_loc": max_loc},
                "allowed_paths": allowed_paths,
            },
        )


__all__ = ["DiffConstraintsGrader"]
