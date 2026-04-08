# type-hygiene: skip-file
"""Code-based graders for ash-hawk evaluation harness.

This module provides graders that evaluate code and agent outputs through:
- String matching (exact, regex, fuzzy)
- Test execution (pytest with sandboxing)
- Static analysis (ruff, mypy, bandit)
- Tool call verification
- Transcript analysis

All graders support partial credit scoring and enforce timeouts.
Secrets are redacted in stored artifacts.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ash_hawk.graders.base import Grader
from ash_hawk.types import (
    EvalTranscript,
    EvalTrial,
    GraderResult,
    GraderSpec,
)

if TYPE_CHECKING:
    pass

# Default timeout for command execution (in seconds)
DEFAULT_COMMAND_TIMEOUT = 60.0

# Patterns for detecting secrets in output
SECRET_PATTERNS = [
    # API keys
    re.compile(r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"]?[\w\-]{16,}['\"]?"),
    # AWS access keys
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # AWS secret keys
    re.compile(r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*['\"]?[\w/+=]{40}['\"]?"),
    # Generic secrets
    re.compile(r"(?i)(secret|password|passwd|token)\s*[=:]\s*['\"]?[^\s'\"]{8,}['\"]?"),
    # Bearer tokens
    re.compile(r"Bearer\s+[\w\-._~+/]+=*"),
    # Private keys
    re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----"),
    # Database URLs
    re.compile(r"(?i)(postgres|mysql|mongodb)://[^\s@]+:[^\s@]+@"),
]

# Default command allowlist for test/analysis runners
DEFAULT_COMMAND_ALLOWLIST = [
    "pytest",
    "python",
    "python3",
    "ruff",
    "mypy",
    "bandit",
    "pylint",
    "black",
    "isort",
]


@dataclass
class CommandResult:
    """Result of a sandboxed command execution."""

    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    execution_time_seconds: float = 0.0
    redacted_secrets: list[str] = field(default_factory=list)


def redact_secrets(text: str) -> tuple[str, list[str]]:
    """Redact secrets from text.

    Args:
        text: The text to process.

    Returns:
        Tuple of (redacted_text, list_of_redacted_values).
    """
    redacted = []
    result = text

    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(result):
            value = match.group(0)
            if value and len(value) > 4:
                redacted.append(value)
                masked = value[:2] + "*" * (len(value) - 4) + value[-2:]
                result = result.replace(value, masked, 1)

    return result, redacted


class SandboxViolationError(Exception):
    """Raised when a command violates sandbox rules."""

    pass


@dataclass
class SandboxConfig:
    """Configuration for command sandboxing."""

    command_allowlist: list[str] = field(default_factory=lambda: DEFAULT_COMMAND_ALLOWLIST)
    allowed_roots: list[str] = field(default_factory=list)
    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT
    env_vars_allowed: list[str] = field(default_factory=list)
    env_passthrough: list[str] = field(default_factory=lambda: ["PATH", "PYTHONPATH", "HOME"])
    max_output_bytes: int = 1024 * 1024  # 1MB max output

    def is_command_allowed(self, command: str) -> bool:
        """Check if a command is in the allowlist.

        Args:
            command: The command to check (first word of command line).

        Returns:
            True if command is allowed.
        """
        # Extract the base command (handle paths)
        base_cmd = os.path.basename(command.split()[0] if command else "")

        for pattern in self.command_allowlist:
            if fnmatch.fnmatch(base_cmd, pattern):
                return True

        return False

    def validate_path_access(self, path: str) -> bool:
        """Check if a path is within allowed roots.

        Args:
            path: The filesystem path to check.

        Returns:
            True if path access is allowed.
        """
        if not self.allowed_roots:
            # No roots configured = deny all by default
            return False

        try:
            abs_path = Path(path).resolve()
        except (OSError, ValueError):
            return False

        for allowed_root in self.allowed_roots:
            try:
                root = Path(allowed_root).resolve()
                if str(abs_path).startswith(str(root)):
                    return True
            except (OSError, ValueError):
                continue

        return False


async def run_sandboxed_command(
    command: str,
    cwd: str | None = None,
    sandbox: SandboxConfig | None = None,
    extra_env: dict[str, str] | None = None,
) -> CommandResult:
    """Run a command in a sandboxed environment.

    Args:
        command: The command to execute.
        cwd: Working directory for command execution.
        sandbox: Sandbox configuration. If None, uses defaults.
        extra_env: Additional environment variables to set.

    Returns:
        CommandResult with execution details.

    Raises:
        SandboxViolationError: If command violates sandbox rules.
    """
    sandbox = sandbox or SandboxConfig()
    start_time = time.time()

    # Parse command to check allowlist
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()

    if not parts:
        return CommandResult(
            return_code=1,
            stdout="",
            stderr="Empty command",
            execution_time_seconds=0.0,
        )

    base_cmd = os.path.basename(parts[0])

    if not sandbox.is_command_allowed(base_cmd):
        raise SandboxViolationError(
            f"Command '{base_cmd}' is not in the allowlist. "
            f"Allowed commands: {sandbox.command_allowlist}"
        )

    # Validate working directory
    if cwd and not sandbox.validate_path_access(cwd):
        raise SandboxViolationError(f"Working directory '{cwd}' is outside allowed roots")

    # Build environment
    env = {}
    for var in sandbox.env_passthrough:
        if var in os.environ:
            env[var] = os.environ[var]

    for var in sandbox.env_vars_allowed:
        if var in os.environ:
            env[var] = os.environ[var]

    if extra_env:
        env.update(extra_env)

    # Execute command with timeout
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env=env if env else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=sandbox.timeout_seconds,
            )
            timed_out = False
        except TimeoutError:
            process.kill()
            await process.wait()
            stdout_bytes, stderr_bytes = b"", b"Command timed out"
            timed_out = True

        stdout = stdout_bytes.decode("utf-8", errors="replace")[: sandbox.max_output_bytes]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[: sandbox.max_output_bytes]

        # Redact secrets from output
        stdout, stdout_secrets = redact_secrets(stdout)
        stderr, stderr_secrets = redact_secrets(stderr)

        execution_time = time.time() - start_time

        return CommandResult(
            return_code=process.returncode or 1,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            execution_time_seconds=execution_time,
            redacted_secrets=stdout_secrets + stderr_secrets,
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return CommandResult(
            return_code=1,
            stdout="",
            stderr=str(e),
            execution_time_seconds=execution_time,
        )


class StringMatchGrader(Grader):
    """Grader for string matching evaluation.

    Supports three matching modes:
    - exact: Character-by-character equality
    - regex: Regular expression pattern matching
    - fuzzy: Similarity-based matching using SequenceMatcher

    Config options:
        expected: The expected string value
        mode: Matching mode - "exact", "regex", or "fuzzy" (default: "exact")
        case_sensitive: Whether to match case (default: True)
        min_similarity: Minimum similarity ratio for fuzzy mode (default: 0.8)
        normalize_whitespace: Normalize whitespace before comparison (default: False)
        partial_credit: Enable partial credit for fuzzy matching (default: False)
    """

    @property
    def name(self) -> str:
        return "string_match"

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Grade by matching output against expected value."""
        config = spec.config
        expected = config.get("expected", "")
        mode = config.get("mode", "exact")
        case_sensitive = config.get("case_sensitive", True)
        min_similarity = config.get("min_similarity", 0.8)
        normalize_whitespace = config.get("normalize_whitespace", False)
        partial_credit = config.get("partial_credit", False)

        # Get actual output from transcript
        actual = self._get_actual_output(transcript, trial)

        # Normalize if requested
        if normalize_whitespace:
            actual = " ".join(actual.split())
            expected = " ".join(expected.split())

        if not case_sensitive:
            actual = actual.lower()
            expected = expected.lower()

        # Perform matching based on mode
        if mode == "exact":
            return self._exact_match(actual, expected, partial_credit)
        elif mode == "regex":
            return self._regex_match(actual, expected, partial_credit)
        elif mode == "fuzzy":
            return self._fuzzy_match(actual, expected, min_similarity, partial_credit)
        elif mode == "contains":
            # expected should be a list of substrings to find
            contains_list = config.get("contains", [])
            if isinstance(contains_list, str):
                contains_list = [contains_list]
            return self._contains_match(actual, contains_list, partial_credit)
        else:
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                error_message=f"Unknown matching mode: {mode}",
            )

    def _get_actual_output(self, transcript: EvalTranscript, trial: EvalTrial) -> str:
        """Extract actual output from transcript or trial."""
        # First check transcript agent_response
        if transcript.agent_response:
            if isinstance(transcript.agent_response, str):
                return transcript.agent_response
            elif isinstance(transcript.agent_response, dict):
                return transcript.agent_response.get("content", "")
        # Then check trial result
        if trial.result and trial.result.transcript.agent_response:
            resp = trial.result.transcript.agent_response
            if isinstance(resp, str):
                return resp
            elif isinstance(resp, dict):
                return resp.get("content", "")
        # Finally, check last message in transcript
        if transcript.messages:
            last_msg = transcript.messages[-1]
            content = last_msg.get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle structured content
                texts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                return " ".join(texts)
        return ""

    def _exact_match(
        self,
        actual: str,
        expected: str,
        partial_credit: bool,
    ) -> GraderResult:
        """Perform exact string matching."""
        matches = actual == expected
        score = 1.0 if matches else (0.5 if partial_credit and expected in actual else 0.0)

        return GraderResult(
            grader_type=self.name,
            score=score,
            passed=matches,
            details={
                "mode": "exact",
                "expected_length": len(expected),
                "actual_length": len(actual),
                "match_type": "full" if matches else ("partial" if score > 0 else "none"),
            },
        )

    def _regex_match(
        self,
        actual: str,
        expected: str,
        partial_credit: bool,
    ) -> GraderResult:
        """Perform regex pattern matching."""
        try:
            pattern = re.compile(expected)
            match = pattern.search(actual)
            full_match = pattern.fullmatch(actual)

            if full_match:
                score = 1.0
            elif match and partial_credit:
                # Partial credit based on match length vs actual length
                score = min(0.8, len(match.group()) / max(len(actual), 1))
            elif match:
                score = 1.0  # Any match counts
            else:
                score = 0.0

            return GraderResult(
                grader_type=self.name,
                score=score,
                passed=match is not None,
                details={
                    "mode": "regex",
                    "pattern": expected,
                    "matched": match is not None,
                    "full_match": full_match is not None,
                    "match_text": match.group() if match else None,
                },
            )
        except re.error as e:
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                error_message=f"Invalid regex pattern: {e}",
            )

    def _fuzzy_match(
        self,
        actual: str,
        expected: str,
        min_similarity: float,
        partial_credit: bool,
    ) -> GraderResult:
        """Perform fuzzy string matching using SequenceMatcher."""
        matcher = SequenceMatcher(None, actual, expected)
        similarity = matcher.ratio()
        passed = similarity >= min_similarity

        if partial_credit:
            # Scale score based on similarity
            if passed:
                score = similarity
            else:
                # Partial credit below threshold
                score = similarity * 0.5
        else:
            score = 1.0 if passed else 0.0

        return GraderResult(
            grader_type=self.name,
            score=score,
            passed=passed,
            details={
                "mode": "fuzzy",
                "similarity": round(similarity, 4),
                "min_similarity": min_similarity,
                "match_blocks": len(matcher.get_matching_blocks()),
            },
        )

    def _contains_match(
        self,
        actual: str,
        expected: list[str],
        partial_credit: bool,
    ) -> GraderResult:
        """Check if all expected substrings are present in actual."""
        if isinstance(expected, str):
            expected = [expected]

        actual_lower = actual.lower()
        found = [substr for substr in expected if substr.lower() in actual_lower]
        missing = [substr for substr in expected if substr.lower() not in actual_lower]
        score = len(found) / len(expected) if expected else 1.0

        if partial_credit:
            final_score = score
        else:
            final_score = 1.0 if score == 1.0 else 0.0

        return GraderResult(
            grader_type=self.name,
            score=final_score,
            passed=score == 1.0,
            details={
                "mode": "contains",
                "found": found,
                "missing": missing,
                "total_expected": len(expected),
                "total_found": len(found),
            },
        )


class TestRunnerGrader(Grader):
    """Grader that runs pytest and evaluates test results.

    Executes tests in a sandboxed environment with:
    - Command allowlist enforcement
    - Workspace path restrictions
    - Timeout enforcement
    - Secret redaction in output

    Config options:
        test_path: Path to test file or directory (required)
        pytest_args: Additional pytest arguments (default: [])
        pass_threshold: Minimum pass ratio to pass (default: 1.0)
        partial_credit: Enable scoring based on pass ratio (default: False)
        timeout_seconds: Command timeout (default: 60)
        command_allowlist: Allowed commands (default: pytest, python)
        workspace_root: Root directory for path validation (default: cwd)
        collect_coverage: Whether to collect coverage info (default: False)
    """

    @property
    def name(self) -> str:
        return "test_runner"

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Grade by running pytest tests."""
        config = spec.config
        test_path = config.get("test_path")
        if not test_path:
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                error_message="test_path is required in config",
            )

        pytest_args = config.get("pytest_args", [])
        pass_threshold = config.get("pass_threshold", 1.0)
        partial_credit = config.get("partial_credit", False)
        timeout_seconds = config.get("timeout_seconds", DEFAULT_COMMAND_TIMEOUT)
        command_allowlist = config.get(
            "command_allowlist",
            ["pytest", "python", "python3"],
        )
        workspace_root = config.get("workspace_root", os.getcwd())
        collect_coverage = config.get("collect_coverage", False)

        # Build sandbox config
        sandbox = SandboxConfig(
            command_allowlist=command_allowlist,
            allowed_roots=[workspace_root],
            timeout_seconds=timeout_seconds,
            env_vars_allowed=["PYTHONPATH", "PYTEST_*"],
        )

        # Build pytest command
        cmd_parts = ["pytest", test_path, "-v", "--tb=short"]
        cmd_parts.extend(pytest_args)

        if collect_coverage:
            cmd_parts.extend(["--cov=.", "--cov-report=term-missing"])

        command = " ".join(cmd_parts)

        try:
            result = await run_sandboxed_command(
                command=command,
                cwd=workspace_root,
                sandbox=sandbox,
            )

            # Parse test results
            test_results = self._parse_pytest_output(result.stdout, result.stderr)

            # Calculate score
            if test_results["total"] == 0:
                score = 0.0
                passed = False
            else:
                pass_ratio = test_results["passed"] / test_results["total"]
                if partial_credit:
                    score = pass_ratio
                else:
                    score = 1.0 if pass_ratio >= pass_threshold else 0.0
                passed = pass_ratio >= pass_threshold

            details = {
                "command": command,
                "return_code": result.return_code,
                "total_tests": test_results["total"],
                "passed_tests": test_results["passed"],
                "failed_tests": test_results["failed"],
                "skipped_tests": test_results["skipped"],
                "pass_ratio": round(test_results["passed"] / max(test_results["total"], 1), 4),
                "execution_time_seconds": result.execution_time_seconds,
                "timed_out": result.timed_out,
            }

            if collect_coverage:
                details["coverage"] = test_results.get("coverage")

            if result.redacted_secrets:
                details["secrets_redacted_count"] = len(result.redacted_secrets)

            return GraderResult(
                grader_type=self.name,
                score=score,
                passed=passed,
                details=details,
                execution_time_seconds=result.execution_time_seconds,
            )

        except SandboxViolationError as e:
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                error_message=f"Sandbox violation: {e}",
            )
        except Exception as e:
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                error_message=f"Test execution failed: {e}",
            )

    def _parse_pytest_output(self, stdout: str, stderr: str) -> dict[str, Any]:
        """Parse pytest output to extract test results."""
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
        }

        # Look for pytest summary line
        # Example: "5 passed, 2 failed, 1 skipped in 1.23s"
        output = stdout + stderr

        # Match summary pattern
        summary_patterns = [
            # Newer pytest format
            re.compile(
                r"(\d+)\s+passed\s*(?:,\s*(\d+)\s+failed\s*)?(?:,\s*(\d+)\s+skipped\s*)?",
                re.IGNORECASE,
            ),
            # Alternative format with errors
            re.compile(
                r"(\d+)\s+passed\s*(?:,\s*(\d+)\s+failed\s*)?"
                r"(?:,\s*(\d+)\s+errors?\s*)?(?:,\s*(\d+)\s+skipped\s*)?",
                re.IGNORECASE,
            ),
        ]

        for pattern in summary_patterns:
            match = pattern.search(output)
            if match:
                groups = match.groups()
                results["passed"] = int(groups[0] or 0)
                results["failed"] = int(groups[1] or 0)
                results["errors"] = int(groups[2] or 0) if len(groups) > 2 else 0
                results["skipped"] = int(groups[3] or 0) if len(groups) > 3 else 0
                break

        # Parse coverage if present
        coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+%)", output)
        if coverage_match:
            results["coverage"] = coverage_match.group(1)

        results["total"] = (
            results["passed"] + results["failed"] + results["skipped"] + results["errors"]
        )

        return results


class StaticAnalysisGrader(Grader):
    """Grader that runs static analysis tools (ruff, mypy, bandit).

    Executes analysis tools in a sandboxed environment and reports
    issues found.

    Config options:
        tools: List of tools to run - "ruff", "mypy", "bandit" (default: ["ruff"])
        target_path: Path to analyze (required)
        fail_on_issues: Fail if any issues found (default: True)
        max_issues: Maximum allowed issues (default: 0)
        partial_credit: Score based on issue severity (default: False)
        timeout_seconds: Per-tool timeout (default: 60)
        command_allowlist: Allowed commands (default: ruff, mypy, bandit)
        workspace_root: Root directory for path validation (default: cwd)
        ruff_args: Additional ruff arguments
        mypy_args: Additional mypy arguments
        bandit_args: Additional bandit arguments
    """

    @property
    def name(self) -> str:
        return "static_analysis"

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Grade by running static analysis tools."""
        config = spec.config
        target_path = config.get("target_path")
        if not target_path:
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                error_message="target_path is required in config",
            )

        tools = config.get("tools", ["ruff"])
        fail_on_issues = config.get("fail_on_issues", True)
        max_issues = config.get("max_issues", 0)
        partial_credit = config.get("partial_credit", False)
        timeout_seconds = config.get("timeout_seconds", DEFAULT_COMMAND_TIMEOUT)
        command_allowlist = config.get(
            "command_allowlist",
            ["ruff", "mypy", "bandit", "python", "python3"],
        )
        workspace_root = config.get("workspace_root", os.getcwd())

        # Build sandbox config
        sandbox = SandboxConfig(
            command_allowlist=command_allowlist,
            allowed_roots=[workspace_root],
            timeout_seconds=timeout_seconds,
        )

        # Run each tool
        tool_results: dict[str, dict[str, Any]] = {}
        total_issues = 0
        total_errors = 0

        for tool in tools:
            tool_result = await self._run_tool(
                tool=tool,
                target_path=target_path,
                workspace_root=workspace_root,
                sandbox=sandbox,
                config=config,
            )
            tool_results[tool] = tool_result
            total_issues += tool_result.get("issue_count", 0)
            total_errors += tool_result.get("error_count", 0)

        # Calculate score
        passed = total_issues <= max_issues
        if partial_credit:
            # Score decreases with more issues
            if total_issues == 0:
                score = 1.0
            elif total_issues <= 5:
                score = 0.8
            elif total_issues <= 10:
                score = 0.6
            elif total_issues <= 20:
                score = 0.4
            else:
                score = max(0.0, 0.2 - (total_issues * 0.01))
        else:
            score = 1.0 if passed else 0.0

        return GraderResult(
            grader_type=self.name,
            score=score,
            passed=passed if fail_on_issues else True,
            details={
                "tools_run": tools,
                "tool_results": tool_results,
                "total_issues": total_issues,
                "total_errors": total_errors,
                "max_issues": max_issues,
            },
        )

    async def _run_tool(
        self,
        tool: str,
        target_path: str,
        workspace_root: str,
        sandbox: SandboxConfig,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a single static analysis tool."""
        if tool == "ruff":
            args = config.get("ruff_args", [])
            command = f"ruff check {target_path} " + " ".join(args)
        elif tool == "mypy":
            args = config.get("mypy_args", [])
            command = f"mypy {target_path} " + " ".join(args)
        elif tool == "bandit":
            args = config.get("bandit_args", [])
            command = f"bandit -r {target_path} " + " ".join(args)
        else:
            return {"error": f"Unknown tool: {tool}", "issue_count": 0, "error_count": 0}

        try:
            result = await run_sandboxed_command(
                command=command,
                cwd=workspace_root,
                sandbox=sandbox,
            )

            issues = self._parse_tool_output(tool, result.stdout, result.stderr)
            return {
                "command": command,
                "return_code": result.return_code,
                "issue_count": issues["count"],
                "error_count": issues["errors"],
                "issues": issues["items"][:20],  # Limit stored issues
                "execution_time_seconds": result.execution_time_seconds,
            }

        except SandboxViolationError as e:
            return {
                "error": str(e),
                "issue_count": 0,
                "error_count": 1,
            }
        except Exception as e:
            return {
                "error": str(e),
                "issue_count": 0,
                "error_count": 1,
            }

    def _parse_tool_output(
        self,
        tool: str,
        stdout: str,
        stderr: str,
    ) -> dict[str, Any]:
        """Parse tool output to extract issues."""
        output = stdout + stderr
        issues = []

        if tool == "ruff":
            # Ruff output format: path:line:col: code message
            pattern = re.compile(r"(.+):(\d+):(\d+):\s+(\w+)\s+(.+)")
            for match in pattern.finditer(output):
                issues.append(
                    {
                        "file": match.group(1),
                        "line": int(match.group(2)),
                        "column": int(match.group(3)),
                        "code": match.group(4),
                        "message": match.group(5),
                    }
                )

        elif tool == "mypy":
            # Mypy output format: path:line: error: message [code]
            pattern = re.compile(r"(.+):(\d+):\s+(error|warning|note):\s+(.+?)(?:\s+\[(.+)\])?$")
            for match in pattern.finditer(output):
                issues.append(
                    {
                        "file": match.group(1),
                        "line": int(match.group(2)),
                        "severity": match.group(3),
                        "message": match.group(4),
                        "code": match.group(5),
                    }
                )

        elif tool == "bandit":
            # Bandit output format includes severity levels
            pattern = re.compile(
                r"(?:Issue|Test):\s+\[?\]?\s*(.+?)\s+Severity:\s+(\w+)\s+Confidence:\s+(\w+)"
            )
            for match in pattern.finditer(output):
                issues.append(
                    {
                        "message": match.group(1).strip(),
                        "severity": match.group(2),
                        "confidence": match.group(3),
                    }
                )

        # Count errors specifically (high severity issues)
        error_count = sum(
            1
            for i in issues
            if i.get("severity") in ("error", "Error", "HIGH") or i.get("code", "").startswith("E")
        )

        return {
            "count": len(issues),
            "errors": error_count,
            "items": issues,
        }


class ToolCallGrader(Grader):
    """Grader that verifies expected tool calls were made.

    Checks that specific tools were called during trial execution,
    optionally verifying call parameters and order.

    Config options:
        expected_calls: List of expected tool call specs
            - tool: Tool name (required)
            - input: Expected input parameters (optional)
            - input_match: "exact", "contains", "regex" (default: "contains")
            - min_count: Minimum times tool should be called (default: 1)
            - max_count: Maximum times tool should be called (optional)
        require_all: All expected calls must be found (default: True)
        check_order: Verify calls happened in order (default: False)
        partial_credit: Score based on match ratio (default: False)
    """

    @property
    def name(self) -> str:
        return "tool_call"

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Grade by verifying tool calls were made."""
        config = spec.config
        expected_calls = config.get("expected_calls", [])
        require_all = config.get("require_all", True)
        check_order = config.get("check_order", False)
        partial_credit = config.get("partial_credit", False)

        if not expected_calls:
            return GraderResult(
                grader_type=self.name,
                score=1.0,
                passed=True,
                details={"message": "No expected calls specified"},
            )

        # Get actual tool calls from transcript
        actual_calls = transcript.tool_calls
        if trial.result:
            actual_calls = trial.result.transcript.tool_calls

        # Check each expected call
        call_results = []
        matches = 0
        last_match_index = -1

        for expected in expected_calls:
            result = self._check_expected_call(
                expected=expected,
                actual_calls=actual_calls,
                check_order=check_order,
                last_match_index=last_match_index,
            )
            call_results.append(result)

            if result["matched"]:
                matches += 1
                if check_order and result.get("match_index") is not None:
                    last_match_index = result["match_index"]

        # Calculate score
        total_expected = len(expected_calls)
        match_ratio = matches / total_expected if total_expected > 0 else 0.0

        if partial_credit:
            score = match_ratio
        else:
            score = 1.0 if (matches == total_expected if require_all else matches > 0) else 0.0

        passed = (matches == total_expected) if require_all else (matches > 0)

        return GraderResult(
            grader_type=self.name,
            score=score,
            passed=passed,
            details={
                "expected_count": total_expected,
                "matched_count": matches,
                "match_ratio": round(match_ratio, 4),
                "call_results": call_results,
                "check_order": check_order,
            },
        )

    def _check_expected_call(
        self,
        expected: dict[str, Any],
        actual_calls: list[dict[str, Any]],
        check_order: bool,
        last_match_index: int,
    ) -> dict[str, Any]:
        """Check if an expected call is present in actual calls."""
        tool_name = expected.get("tool")
        expected_input = expected.get("input")
        input_match = expected.get("input_match", "contains")
        min_count = expected.get("min_count", 1)
        max_count = expected.get("max_count")

        result = {
            "tool": tool_name,
            "expected_input": expected_input,
            "matched": False,
            "match_count": 0,
            "matches": [],
        }

        if tool_name is None:
            return result

        # Find matching calls
        matching_indices = []
        for i, call in enumerate(actual_calls):
            call_tool = call.get("tool", call.get("name", ""))

            # Check tool name
            if not fnmatch.fnmatch(call_tool, tool_name):
                continue

            # Check order constraint
            if check_order and i <= last_match_index:
                continue

            # Check input if specified
            if expected_input is not None:
                call_input = call.get("input", call.get("arguments", {}))
                if not self._match_input(call_input, expected_input, input_match):
                    continue

            matching_indices.append(i)

        result["match_count"] = len(matching_indices)

        # Check count constraints
        count_ok = len(matching_indices) >= min_count
        if max_count is not None:
            count_ok = count_ok and len(matching_indices) <= max_count

        result["matched"] = count_ok
        result["matches"] = [{"index": i, "call": actual_calls[i]} for i in matching_indices[:5]]

        if matching_indices:
            result["match_index"] = matching_indices[0]

        return result

    def _match_input(
        self,
        actual: dict[str, Any],
        expected: dict[str, Any],
        mode: str,
    ) -> bool:
        """Check if actual input matches expected."""
        if mode == "exact":
            return actual == expected
        elif mode == "contains":
            # All expected keys/values must be present in actual
            for key, value in expected.items():
                if key not in actual:
                    return False
                if isinstance(value, str) and isinstance(actual[key], str):
                    if value.lower() not in actual[key].lower():
                        return False
                elif actual[key] != value:
                    return False
            return True
        elif mode == "regex":
            for key, pattern in expected.items():
                if key not in actual:
                    return False
                try:
                    if not re.search(pattern, str(actual[key])):
                        return False
                except re.error:
                    return False
            return True
        return False


class TranscriptGrader(Grader):
    """Grader that analyzes transcript metrics.

    Checks constraints on turn count, token usage, duration,
    and other transcript metrics.

    Config options:
        max_turns: Maximum allowed turns (optional)
        min_turns: Minimum required turns (optional)
        max_tokens: Maximum total tokens (optional)
        max_input_tokens: Maximum input tokens (optional)
        max_output_tokens: Maximum output tokens (optional)
        max_duration_seconds: Maximum duration (optional)
        min_duration_seconds: Minimum duration (optional)
        max_tool_calls: Maximum tool calls (optional)
        max_cost_usd: Maximum cost in USD (optional)
        require_no_errors: Transcript must have no errors (default: True)
        partial_credit: Enable partial scoring (default: False)
    """

    @property
    def name(self) -> str:
        return "transcript"

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Grade by analyzing transcript metrics."""
        config = spec.config
        partial_credit = config.get("partial_credit", False)

        # Get transcript from trial result if available
        effective_transcript = transcript
        if trial.result:
            effective_transcript = trial.result.transcript

        # Collect metrics
        metrics = self._collect_metrics(effective_transcript)

        # Check constraints
        violations = []
        warnings = []

        # Turn constraints
        turn_count = metrics["turn_count"]
        if (max_turns := config.get("max_turns")) is not None:
            if turn_count > max_turns:
                violations.append(f"Turn count ({turn_count}) exceeds max ({max_turns})")
        if (min_turns := config.get("min_turns")) is not None:
            if turn_count < min_turns:
                violations.append(f"Turn count ({turn_count}) below min ({min_turns})")

        # Token constraints
        tokens = effective_transcript.token_usage
        if (max_tokens := config.get("max_tokens")) is not None:
            if tokens.total > max_tokens:
                violations.append(f"Total tokens ({tokens.total}) exceeds max ({max_tokens})")
        if (max_input := config.get("max_input_tokens")) is not None:
            if tokens.input > max_input:
                violations.append(f"Input tokens ({tokens.input}) exceeds max ({max_input})")
        if (max_output := config.get("max_output_tokens")) is not None:
            if tokens.output > max_output:
                violations.append(f"Output tokens ({tokens.output}) exceeds max ({max_output})")

        # Duration constraints
        duration = effective_transcript.duration_seconds
        if (max_duration := config.get("max_duration_seconds")) is not None:
            if duration > max_duration:
                violations.append(f"Duration ({duration:.2f}s) exceeds max ({max_duration}s)")
        if (min_duration := config.get("min_duration_seconds")) is not None:
            if duration < min_duration:
                warnings.append(f"Duration ({duration:.2f}s) below min ({min_duration}s)")

        # Tool call constraints
        tool_calls = len(effective_transcript.tool_calls)
        if (max_tool_calls := config.get("max_tool_calls")) is not None:
            if tool_calls > max_tool_calls:
                violations.append(f"Tool calls ({tool_calls}) exceeds max ({max_tool_calls})")

        # Cost constraints
        cost = effective_transcript.cost_usd
        if (max_cost := config.get("max_cost_usd")) is not None:
            if cost > max_cost:
                violations.append(f"Cost (${cost:.4f}) exceeds max (${max_cost})")

        # Error check
        require_no_errors = config.get("require_no_errors", True)
        if require_no_errors and effective_transcript.error_trace:
            violations.append("Transcript contains error trace")

        # Calculate score
        passed = len(violations) == 0

        if partial_credit:
            # Reduce score based on number of violations
            score = max(0.0, 1.0 - (len(violations) * 0.2) - (len(warnings) * 0.05))
        else:
            score = 1.0 if passed else 0.0

        return GraderResult(
            grader_type=self.name,
            score=score,
            passed=passed,
            details={
                "metrics": metrics,
                "violations": violations,
                "warnings": warnings,
                "token_usage": {
                    "input": tokens.input,
                    "output": tokens.output,
                    "reasoning": tokens.reasoning,
                    "total": tokens.total,
                },
                "duration_seconds": duration,
                "tool_call_count": tool_calls,
                "cost_usd": cost,
            },
        )

    def _collect_metrics(self, transcript: EvalTranscript) -> dict[str, Any]:
        """Collect metrics from transcript."""
        # Count turns (pairs of user/assistant messages)
        messages = transcript.messages
        user_count = sum(1 for m in messages if m.get("role") == "user")
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
        turn_count = min(user_count, assistant_count)

        # Count by message type
        message_types: dict[str, int] = {}
        for msg in messages:
            role = msg.get("role", "unknown")
            message_types[role] = message_types.get(role, 0) + 1

        # Tool call summary
        tool_summary: dict[str, int] = {}
        for call in transcript.tool_calls:
            tool_name = call.get("tool", call.get("name", "unknown"))
            tool_summary[tool_name] = tool_summary.get(tool_name, 0) + 1

        return {
            "turn_count": turn_count,
            "message_count": len(messages),
            "user_message_count": user_count,
            "assistant_message_count": assistant_count,
            "message_types": message_types,
            "tool_call_count": len(transcript.tool_calls),
            "tool_summary": tool_summary,
            "has_error": transcript.error_trace is not None,
        }


__all__ = [
    "StringMatchGrader",
    "TestRunnerGrader",
    "StaticAnalysisGrader",
    "ToolCallGrader",
    "TranscriptGrader",
    "SandboxConfig",
    "SandboxViolationError",
    "CommandResult",
    "run_sandboxed_command",
    "redact_secrets",
]
