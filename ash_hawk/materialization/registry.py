from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ash_hawk.contracts import CuratedLesson
from ash_hawk.materialization.base import MaterializerBackend, PayloadMapper
from ash_hawk.materialization.types import (
    FileFormat,
    MaterializationConfig,
    MaterializationResult,
    RepoTarget,
)


class ProjectRegistry:
    """Registry of target projects/repositories for materialization."""

    def __init__(self) -> None:
        self._targets: dict[str, RepoTarget] = {}

    def register(
        self,
        agent_id: str,
        repo_root: str | Path,
        *,
        repo_url: str | None = None,
        branch: str = "main",
        default_format: FileFormat = FileFormat.TEXT,
    ) -> None:
        self._targets[agent_id] = RepoTarget(
            agent_id=agent_id,
            repo_url=repo_url,
            repo_root=str(Path(repo_root).resolve()),
            branch=branch,
            default_format=default_format,
        )

    def get(self, agent_id: str) -> RepoTarget | None:
        return self._targets.get(agent_id)

    def get_config(
        self,
        agent_id: str,
        *,
        run_lint: bool = True,
        run_types: bool = True,
        run_tests: bool = False,
        auto_commit: bool = False,
        auto_rollback: bool = True,
        lint_command: str | None = None,
        type_command: str | None = None,
        test_command: str | None = None,
    ) -> MaterializationConfig | None:
        target = self.get(agent_id)
        if target is None:
            return None
        return MaterializationConfig(
            repo_root=target.repo_root,
            agent_id=target.agent_id,
            run_lint=run_lint,
            run_types=run_types,
            run_tests=run_tests,
            auto_commit=auto_commit,
            auto_rollback=auto_rollback,
            lint_command=lint_command,
            type_command=type_command,
            test_command=test_command,
        )

    def list_agents(self) -> list[str]:
        return list(self._targets.keys())


class LessonMaterializer:
    """Orchestrates lesson materialization across projects."""

    def __init__(
        self,
        backend: MaterializerBackend,
        mappers: list[PayloadMapper],
        registry: ProjectRegistry,
    ) -> None:
        self._backend = backend
        self._mappers = mappers
        self._registry = registry

    async def materialize(
        self,
        lesson: CuratedLesson,
        config: MaterializationConfig,
    ) -> MaterializationResult:
        materialization_id = f"mat-{uuid4().hex[:8]}"
        repo_root = Path(config.repo_root)
        target = self._registry.get(config.agent_id)
        target_format = target.default_format if target else FileFormat.TEXT

        patches = self._map_lesson(lesson, repo_root, target_format)

        if not patches:
            return MaterializationResult(
                materialization_id=materialization_id,
                lesson_id=lesson.lesson_id,
                agent_id=config.agent_id,
                repo_path=str(repo_root),
                error="No patches generated for lesson",
            )

        apply_result = await self._backend.apply(patches, config)
        apply_result.materialization_id = materialization_id
        apply_result.lesson_id = lesson.lesson_id

        if apply_result.error:
            return apply_result

        if config.run_lint or config.run_types or config.run_tests:
            verification = await self._backend.verify(config)
            apply_result.verification = verification

            if not verification.passed and config.auto_rollback:
                rolled_back = await self._backend.rollback(config)
                apply_result.rolled_back = rolled_back
                apply_result.rollback_reason = "Verification failed"
                return apply_result

        if config.auto_commit and (
            apply_result.verification is None or apply_result.verification.passed
        ):
            message = self._build_commit_message(lesson)
            commit = await self._backend.commit(message, config)
            apply_result.commit = commit

        return apply_result

    async def materialize_batch(
        self,
        lessons: list[CuratedLesson],
        config: MaterializationConfig,
    ) -> list[MaterializationResult]:
        results: list[MaterializationResult] = []
        for lesson in lessons:
            result = await self.materialize(lesson, config)
            results.append(result)
        return results

    def _map_lesson(
        self,
        lesson: CuratedLesson,
        repo_root: Path,
        target_format: FileFormat,
    ) -> list[Any]:
        patches: list[Any] = []
        for mapper in self._mappers:
            if mapper.can_map(lesson, target_format):
                patches.extend(mapper.map(lesson, repo_root, target_format))
        return patches

    def _build_commit_message(self, lesson: CuratedLesson) -> str:
        lines = [
            f"feat(lesson): {lesson.title}",
            "",
            f"Lesson ID: {lesson.lesson_id}",
            f"Type: {lesson.lesson_type}",
            f"Source: {lesson.source_proposal_id}",
        ]
        if lesson.strategy:
            lines.append(f"Strategy: {lesson.strategy.value}")
        lines.append("")
        lines.append(lesson.description)
        return "\n".join(lines)


class MaterializationStore:
    """Persistent store for materialization results."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)

    def save(self, result: MaterializationResult) -> Path:
        import json

        file_path = self._base_path / f"{result.materialization_id}.json"
        data = result.model_dump(mode="json")
        data["timestamp"] = datetime.now(UTC).isoformat()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return file_path

    def load(self, materialization_id: str) -> MaterializationResult | None:
        import json

        file_path = self._base_path / f"{materialization_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        return MaterializationResult.model_validate(data)

    def list_for_agent(self, agent_id: str) -> list[MaterializationResult]:
        results: list[MaterializationResult] = []
        for file_path in self._base_path.glob("mat-*.json"):
            result = self.load(file_path.stem)
            if result and result.agent_id == agent_id:
                results.append(result)
        return results

    def list_successful(self) -> list[MaterializationResult]:
        results: list[MaterializationResult] = []
        for file_path in self._base_path.glob("mat-*.json"):
            result = self.load(file_path.stem)
            if result and not result.rolled_back and not result.error:
                results.append(result)
        return results
