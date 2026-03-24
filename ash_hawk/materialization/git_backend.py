from __future__ import annotations

import asyncio
import re
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from ash_hawk.materialization.base import MaterializerBackend
from ash_hawk.materialization.types import (
    CommitMetadata,
    FileFormat,
    MaterializationConfig,
    MaterializationResult,
    PatchKind,
    PatchOperation,
    VerificationResult,
)


class GitRepoBackend(MaterializerBackend):
    """Materialize lessons into a git-tracked repository."""

    def __init__(self) -> None:
        self._pending_files: dict[str, str] = {}

    async def apply(
        self,
        patches: list[PatchOperation],
        config: MaterializationConfig,
    ) -> MaterializationResult:
        repo_root = Path(config.repo_root).resolve()
        materialization_id = f"mat-{uuid4().hex[:8]}"
        files_modified: list[str] = []

        try:
            for patch in patches:
                modified = self._apply_patch(patch, repo_root)
                if modified:
                    files_modified.append(modified)

            return MaterializationResult(
                materialization_id=materialization_id,
                lesson_id="",
                agent_id=config.agent_id,
                repo_path=str(repo_root),
                patches_applied=patches,
                files_modified=files_modified,
            )
        except Exception as exc:
            return MaterializationResult(
                materialization_id=materialization_id,
                lesson_id="",
                agent_id=config.agent_id,
                repo_path=str(repo_root),
                patches_applied=patches,
                files_modified=files_modified,
                error=str(exc),
            )

    async def verify(self, config: MaterializationConfig) -> VerificationResult:
        start = time.monotonic()
        errors: list[str] = []
        linter_errors: list[str] = []
        type_errors: list[str] = []
        test_failures: list[str] = []
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        repo_root = Path(config.repo_root)

        if config.run_lint:
            lint_cmd = config.lint_command or self._detect_lint_command(repo_root)
            if lint_cmd:
                result = await self._run_command(lint_cmd, repo_root)
                stdout_parts.append(result.stdout)
                stderr_parts.append(result.stderr)
                if result.returncode != 0:
                    linter_errors.extend(self._parse_lint_errors(result.stderr or result.stdout))
                    errors.extend(linter_errors)

        if config.run_types:
            type_cmd = config.type_command or self._detect_type_command(repo_root)
            if type_cmd:
                result = await self._run_command(type_cmd, repo_root)
                stdout_parts.append(result.stdout)
                stderr_parts.append(result.stderr)
                if result.returncode != 0:
                    type_errors.extend(self._parse_type_errors(result.stderr or result.stdout))
                    errors.extend(type_errors)

        if config.run_tests:
            test_cmd = config.test_command or self._detect_test_command(repo_root)
            if test_cmd:
                result = await self._run_command(test_cmd, repo_root)
                stdout_parts.append(result.stdout)
                stderr_parts.append(result.stderr)
                if result.returncode != 0:
                    test_failures.extend(self._parse_test_failures(result.stderr or result.stdout))
                    errors.extend(test_failures)

        duration_ms = int((time.monotonic() - start) * 1000)

        return VerificationResult(
            passed=len(errors) == 0,
            linter_errors=linter_errors,
            type_errors=type_errors,
            test_failures=test_failures,
            stdout="\n".join(filter(None, stdout_parts)) or None,
            stderr="\n".join(filter(None, stderr_parts)) or None,
            duration_ms=duration_ms,
        )

    async def commit(
        self,
        message: str,
        config: MaterializationConfig,
    ) -> CommitMetadata:
        repo_root = Path(config.repo_root)

        subprocess.run(
            ["git", "add", "-A"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        _commit_result = subprocess.run(
            ["git", "commit", "-m", message, "--author", "ash-hawk <ash-hawk@local>"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        sha = sha_result.stdout.strip() or "unknown"

        return CommitMetadata(
            sha=sha,
            message=message,
            author="ash-hawk",
        )

    async def rollback(self, config: MaterializationConfig) -> bool:
        repo_root = Path(config.repo_root)

        result = subprocess.run(
            ["git", "checkout", "--", "."],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        clean_result = subprocess.run(
            ["git", "clean", "-fd"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        return result.returncode == 0 and clean_result.returncode == 0

    def _apply_patch(self, patch: PatchOperation, repo_root: Path) -> str | None:
        target_path = repo_root / patch.path

        if patch.kind == PatchKind.CREATE_FILE:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            content = patch.content or ""
            if patch.marker:
                content = f"{patch.marker}\n\n{content}"
            target_path.write_text(content, encoding="utf-8")
            return patch.path

        if not target_path.exists():
            if patch.kind in {PatchKind.APPEND_SECTION, PatchKind.PREPEND_SECTION}:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text("", encoding="utf-8")
            else:
                return None

        existing = target_path.read_text(encoding="utf-8")

        if patch.kind == PatchKind.UPDATE_FILE:
            new_content = patch.content or ""
            if patch.marker and patch.marker in existing:
                return None
            if patch.marker:
                new_content = f"{patch.marker}\n{new_content}"
            target_path.write_text(new_content, encoding="utf-8")
            return patch.path

        if patch.kind == PatchKind.APPEND_SECTION:
            if patch.marker and patch.marker in existing:
                return None
            section = self._build_section(patch)
            separator = "\n\n" if existing.strip() else ""
            target_path.write_text(f"{existing}{separator}{section}", encoding="utf-8")
            return patch.path

        if patch.kind == PatchKind.PREPEND_SECTION:
            if patch.marker and patch.marker in existing:
                return None
            section = self._build_section(patch)
            separator = "\n\n" if existing.strip() else ""
            target_path.write_text(f"{section}{separator}{existing}", encoding="utf-8")
            return patch.path

        if patch.kind == PatchKind.DELETE_SECTION:
            if not patch.marker:
                return None
            new_content = self._remove_section(existing, patch.marker)
            if new_content != existing:
                target_path.write_text(new_content, encoding="utf-8")
                return patch.path
            return None

        if patch.kind == PatchKind.AST_TRANSFORM:
            if patch.format != FileFormat.PYTHON:
                return None
            new_content = self._apply_ast_transform(
                existing, patch.ast_pattern or "", patch.ast_replacement or ""
            )
            if new_content and new_content != existing:
                target_path.write_text(new_content, encoding="utf-8")
                return patch.path
            return None

        return None

    def _build_section(self, patch: PatchOperation) -> str:
        if patch.marker:
            return f"\n{patch.marker}\n{patch.content or ''}"
        return patch.content or ""

    def _remove_section(self, content: str, marker: str) -> str:
        pattern = re.escape(marker) + r".*?(?=\n<!-- ash-hawk-lesson:|$)"
        return re.sub(pattern, "", content, flags=re.DOTALL).strip()

    def _apply_ast_transform(self, content: str, pattern: str, replacement: str) -> str | None:
        try:
            import ast

            tree = ast.parse(content)
            del tree
            return content.replace(pattern, replacement)
        except SyntaxError:
            return None

    def _detect_lint_command(self, repo_root: Path) -> str | None:
        if (repo_root / "ruff.toml").exists() or (repo_root / "pyproject.toml").exists():
            return "ruff check ."
        if (repo_root / ".flake8").exists():
            return "flake8 ."
        return None

    def _detect_type_command(self, repo_root: Path) -> str | None:
        if (repo_root / "pyrightconfig.json").exists() or (repo_root / "pyproject.toml").exists():
            return "mypy ."
        return None

    def _detect_test_command(self, repo_root: Path) -> str | None:
        if (repo_root / "pytest.ini").exists() or (repo_root / "pyproject.toml").exists():
            return "pytest --no-cov -q"
        return None

    async def _run_command(self, command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        return subprocess.CompletedProcess(
            args=command,
            returncode=proc.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    def _parse_lint_errors(self, output: str) -> list[str]:
        lines = output.strip().splitlines()
        return [line for line in lines if re.match(r".*:\d+:\d+: [A-Z]\d+", line)]

    def _parse_type_errors(self, output: str) -> list[str]:
        lines = output.strip().splitlines()
        return [line for line in lines if "error:" in line.lower()]

    def _parse_test_failures(self, output: str) -> list[str]:
        lines = output.strip().splitlines()
        return [
            line
            for line in lines
            if "FAILED" in line or "ERROR" in line or "AssertionError" in line
        ]
