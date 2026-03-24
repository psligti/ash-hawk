from __future__ import annotations

from ash_hawk.materialization.base import MaterializerBackend, PayloadMapper
from ash_hawk.materialization.git_backend import GitRepoBackend
from ash_hawk.materialization.mappers import MarkdownPayloadMapper, PythonPayloadMapper
from ash_hawk.materialization.registry import (
    LessonMaterializer,
    MaterializationStore,
    ProjectRegistry,
)
from ash_hawk.materialization.types import (
    CommitMetadata,
    FileFormat,
    MaterializationConfig,
    MaterializationResult,
    PatchKind,
    PatchOperation,
    RepoTarget,
    VerificationResult,
)

__all__ = [
    "CommitMetadata",
    "FileFormat",
    "GitRepoBackend",
    "LessonMaterializer",
    "MarkdownPayloadMapper",
    "MaterializationConfig",
    "MaterializationResult",
    "MaterializationStore",
    "MaterializerBackend",
    "PatchKind",
    "PatchOperation",
    "PayloadMapper",
    "ProjectRegistry",
    "PythonPayloadMapper",
    "RepoTarget",
    "VerificationResult",
]
