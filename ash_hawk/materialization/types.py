from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

import pydantic as pd


class PatchKind(str, Enum):
    UPDATE_FILE = "update_file"
    APPEND_SECTION = "append_section"
    PREPEND_SECTION = "prepend_section"
    CREATE_FILE = "create_file"
    AST_TRANSFORM = "ast_transform"
    DELETE_SECTION = "delete_section"


class FileFormat(str, Enum):
    PYTHON = "python"
    MARKDOWN = "markdown"
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    TEXT = "text"


class PatchOperation(pd.BaseModel):
    kind: PatchKind = pd.Field(description="Type of patch operation")
    path: str = pd.Field(description="Relative path within repo")
    format: FileFormat = pd.Field(default=FileFormat.TEXT, description="File format")
    content: str | None = pd.Field(default=None, description="Content for create/append")
    marker: str | None = pd.Field(default=None, description="Idempotency marker comment")
    section_title: str | None = pd.Field(
        default=None, description="Section title for append/prepend"
    )
    ast_pattern: str | None = pd.Field(default=None, description="AST pattern for transforms")
    ast_replacement: str | None = pd.Field(default=None, description="AST replacement pattern")
    line_start: int | None = pd.Field(default=None, description="Start line for range operations")
    line_end: int | None = pd.Field(default=None, description="End line for range operations")

    model_config = pd.ConfigDict(extra="forbid")


class VerificationResult(pd.BaseModel):
    passed: bool = pd.Field(description="Whether verification passed")
    linter_errors: list[str] = pd.Field(default_factory=list, description="Linter error messages")
    type_errors: list[str] = pd.Field(default_factory=list, description="Type checker errors")
    test_failures: list[str] = pd.Field(default_factory=list, description="Test failure summaries")
    stdout: str | None = pd.Field(default=None, description="Verification stdout")
    stderr: str | None = pd.Field(default=None, description="Verification stderr")
    duration_ms: int = pd.Field(default=0, description="Verification duration in ms")

    model_config = pd.ConfigDict(extra="forbid")


class CommitMetadata(pd.BaseModel):
    sha: str = pd.Field(description="Git commit SHA")
    message: str = pd.Field(description="Commit message")
    author: str = pd.Field(default="ash-hawk", description="Commit author")
    timestamp: datetime = pd.Field(default_factory=lambda: datetime.now(UTC))

    model_config = pd.ConfigDict(extra="forbid")


class MaterializationResult(pd.BaseModel):
    materialization_id: str = pd.Field(description="Unique materialization ID")
    lesson_id: str = pd.Field(description="Source lesson ID")
    agent_id: str = pd.Field(description="Target agent/project ID")
    repo_path: str = pd.Field(description="Path to repo root")
    patches_applied: list[PatchOperation] = pd.Field(default_factory=list)
    files_modified: list[str] = pd.Field(default_factory=list)
    verification: VerificationResult | None = pd.Field(default=None)
    commit: CommitMetadata | None = pd.Field(default=None)
    rolled_back: bool = pd.Field(default=False, description="Whether changes were rolled back")
    rollback_reason: str | None = pd.Field(default=None)
    error: str | None = pd.Field(default=None)

    model_config = pd.ConfigDict(extra="forbid")


class MaterializationConfig(pd.BaseModel):
    repo_root: str = pd.Field(description="Path to repository root")
    agent_id: str = pd.Field(description="Agent/project identifier")
    run_lint: bool = pd.Field(default=True, description="Run linter after patches")
    run_types: bool = pd.Field(default=True, description="Run type checker after patches")
    run_tests: bool = pd.Field(default=False, description="Run tests after patches")
    auto_commit: bool = pd.Field(default=False, description="Auto-commit successful patches")
    auto_rollback: bool = pd.Field(
        default=True, description="Auto-rollback on verification failure"
    )
    lint_command: str | None = pd.Field(default=None, description="Custom lint command")
    type_command: str | None = pd.Field(default=None, description="Custom type check command")
    test_command: str | None = pd.Field(default=None, description="Custom test command")

    model_config = pd.ConfigDict(extra="forbid")


class RepoTarget(pd.BaseModel):
    agent_id: str = pd.Field(description="Agent/project identifier")
    repo_url: str | None = pd.Field(default=None, description="Git remote URL (None for local)")
    repo_root: str = pd.Field(description="Local path to repo root")
    branch: str = pd.Field(default="main", description="Target branch")
    default_format: FileFormat = pd.Field(default=FileFormat.TEXT)

    model_config = pd.ConfigDict(extra="forbid")


class LessonPayload(pd.BaseModel):
    lesson_id: str
    lesson_type: str
    title: str
    description: str
    target_surface: str
    payload: dict[str, Any]
    sub_strategies: list[str] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="forbid")
