from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel

from ash_hawk.improve_cycle.models import (
    AdversarialScenario,
    ChangeSet,
    CuratedLesson,
    ExperimentPlan,
    ImprovementProposal,
    KnowledgeEntry,
    PromotionDecision,
    ReviewFinding,
    RunArtifactBundle,
    VerificationReport,
)

T = TypeVar("T", bound=BaseModel)


@dataclass
class JsonEntityStore(Generic[T]):
    file_path: Path
    key_field: str
    _cache: dict[str, T] = field(default_factory=lambda: cast(dict[str, T], {}))
    _loaded: bool = False

    def _ensure_loaded(self, model_type: type[T]) -> None:
        if self._loaded:
            return
        if self.file_path.exists():
            try:
                data = json.loads(self.file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {self.file_path}") from exc
            if isinstance(data, list):
                typed_data = cast(list[dict[str, Any]], data)
                for typed_item in typed_data:
                    model = model_type.model_validate(typed_item)
                    key = getattr(model, self.key_field)
                    self._cache[str(key)] = model
        self._loaded = True

    def _persist(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [item.model_dump(mode="json") for item in self._cache.values()]
        temp_path = self.file_path.with_suffix(f"{self.file_path.suffix}.tmp")
        try:
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temp_path.replace(self.file_path)
        except OSError as exc:
            raise RuntimeError(f"Failed to persist improve-cycle store {self.file_path}") from exc

    def upsert(self, item: T, model_type: type[T]) -> None:
        self._ensure_loaded(model_type)
        key = str(getattr(item, self.key_field))
        self._cache[key] = item
        self._persist()

    def get(self, key: str, model_type: type[T]) -> T | None:
        self._ensure_loaded(model_type)
        return self._cache.get(key)

    def list_all(self, model_type: type[T]) -> list[T]:
        self._ensure_loaded(model_type)
        return list(self._cache.values())


class ImproveCycleStorage:
    runs: JsonEntityStore[RunArtifactBundle]
    proposals: JsonEntityStore[ImprovementProposal]
    findings: JsonEntityStore[ReviewFinding]
    lessons: JsonEntityStore[CuratedLesson]
    experiment_plans: JsonEntityStore[ExperimentPlan]
    change_sets: JsonEntityStore[ChangeSet]
    verifications: JsonEntityStore[VerificationReport]
    promotions: JsonEntityStore[PromotionDecision]
    knowledge: JsonEntityStore[KnowledgeEntry]
    adversarial_scenarios: JsonEntityStore[AdversarialScenario]

    def __init__(self, root: str | Path = ".ash-hawk/improve-cycle") -> None:
        base = Path(root)
        self.runs = JsonEntityStore[RunArtifactBundle](base / "runs.json", "run_id")
        self.proposals = JsonEntityStore[ImprovementProposal](
            base / "proposals.json", "proposal_id"
        )
        self.findings = JsonEntityStore[ReviewFinding](base / "findings.json", "finding_id")
        self.lessons = JsonEntityStore[CuratedLesson](base / "lessons.json", "lesson_id")
        self.experiment_plans = JsonEntityStore[ExperimentPlan](
            base / "experiment_plans.json", "experiment_plan_id"
        )
        self.change_sets = JsonEntityStore[ChangeSet](base / "change_sets.json", "change_set_id")
        self.verifications = JsonEntityStore[VerificationReport](
            base / "verifications.json", "verification_id"
        )
        self.promotions = JsonEntityStore[PromotionDecision](
            base / "promotions.json", "decision_id"
        )
        self.knowledge = JsonEntityStore[KnowledgeEntry](base / "knowledge.json", "knowledge_id")
        self.adversarial_scenarios = JsonEntityStore[AdversarialScenario](
            base / "adversarial_scenarios.json", "scenario_id"
        )
