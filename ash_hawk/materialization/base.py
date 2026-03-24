from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ash_hawk.contracts import CuratedLesson
from ash_hawk.materialization.types import (
    CommitMetadata,
    FileFormat,
    MaterializationConfig,
    MaterializationResult,
    PatchOperation,
    VerificationResult,
)


class PayloadMapper(ABC):
    """Map lesson payload to concrete patch operations."""

    @abstractmethod
    def can_map(self, lesson: CuratedLesson, target_format: FileFormat) -> bool:
        raise NotImplementedError

    @abstractmethod
    def map(
        self,
        lesson: CuratedLesson,
        repo_root: Path,
        target_format: FileFormat,
    ) -> list[PatchOperation]:
        raise NotImplementedError


class MaterializerBackend(ABC):
    """Backend for applying patch operations to a repository."""

    @abstractmethod
    async def apply(
        self,
        patches: list[PatchOperation],
        config: MaterializationConfig,
    ) -> MaterializationResult:
        raise NotImplementedError

    @abstractmethod
    async def verify(
        self,
        config: MaterializationConfig,
    ) -> VerificationResult:
        raise NotImplementedError

    @abstractmethod
    async def commit(
        self,
        message: str,
        config: MaterializationConfig,
    ) -> CommitMetadata:
        raise NotImplementedError

    @abstractmethod
    async def rollback(
        self,
        config: MaterializationConfig,
    ) -> bool:
        raise NotImplementedError
