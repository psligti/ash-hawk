from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pydantic as pd
import yaml


class ImproveCycleCompetitorConfig(pd.BaseModel):
    enabled: bool = pd.Field(default=True)

    model_config = pd.ConfigDict(extra="forbid")


class ImproveCycleTriageConfig(pd.BaseModel):
    enabled: bool = pd.Field(default=True)
    allow_multi_causal: bool = pd.Field(default=True)

    model_config = pd.ConfigDict(extra="forbid")


class ImproveCycleCuratorConfig(pd.BaseModel):
    min_confidence: float = pd.Field(default=0.7, ge=0.0, le=1.0)
    require_evidence: bool = pd.Field(default=True)

    model_config = pd.ConfigDict(extra="forbid")


class ImproveCycleVerifierConfig(pd.BaseModel):
    enabled: bool = pd.Field(default=True)
    min_repeats: int = pd.Field(default=3, ge=1)
    max_variance: float = pd.Field(default=0.01, ge=0.0)
    max_latency_delta_pct: float = pd.Field(default=15.0, ge=0.0)
    max_token_delta_pct: float = pd.Field(default=10.0, ge=0.0)
    require_regression_pass: bool = pd.Field(default=True)

    model_config = pd.ConfigDict(extra="forbid")


class ImproveCyclePromotionConfig(pd.BaseModel):
    default_scope: str = pd.Field(default="agent-specific")
    low_risk_success_threshold: int = pd.Field(default=3, ge=1)
    medium_risk_success_threshold: int = pd.Field(default=5, ge=1)

    model_config = pd.ConfigDict(extra="forbid")


class ImproveCycleAdversaryConfig(pd.BaseModel):
    enabled: bool = pd.Field(default=True)
    auto_expand_eval_packs: bool = pd.Field(default=False)

    model_config = pd.ConfigDict(extra="forbid")


class ImproveCycleHistorianConfig(pd.BaseModel):
    enabled: bool = pd.Field(default=True)

    model_config = pd.ConfigDict(extra="forbid")


class ImproveCycleLibrarianConfig(pd.BaseModel):
    enabled: bool = pd.Field(default=True)

    model_config = pd.ConfigDict(extra="forbid")


class ImproveCycleConfig(pd.BaseModel):
    competitor: ImproveCycleCompetitorConfig = pd.Field(
        default_factory=ImproveCycleCompetitorConfig
    )
    triage: ImproveCycleTriageConfig = pd.Field(default_factory=ImproveCycleTriageConfig)
    curator: ImproveCycleCuratorConfig = pd.Field(default_factory=ImproveCycleCuratorConfig)
    verifier: ImproveCycleVerifierConfig = pd.Field(default_factory=ImproveCycleVerifierConfig)
    promotion: ImproveCyclePromotionConfig = pd.Field(default_factory=ImproveCyclePromotionConfig)
    adversary: ImproveCycleAdversaryConfig = pd.Field(default_factory=ImproveCycleAdversaryConfig)
    historian: ImproveCycleHistorianConfig = pd.Field(default_factory=ImproveCycleHistorianConfig)
    librarian: ImproveCycleLibrarianConfig = pd.Field(default_factory=ImproveCycleLibrarianConfig)

    model_config = pd.ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> ImproveCycleConfig:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Improve-cycle config not found: {path}")
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("Improve-cycle config root must be a mapping")
        if "improve_cycle" in raw:
            inner = cast(Any, raw["improve_cycle"])
            if not isinstance(inner, dict):
                raise ValueError("improve_cycle must be a mapping")
            raw = cast(dict[str, Any], inner)
        return cls.model_validate(cast(dict[str, Any], raw))


def load_improve_cycle_config(config_path: str | None = None) -> ImproveCycleConfig:
    candidate_paths: list[Path] = []
    if config_path:
        candidate_paths.append(Path(config_path))
    else:
        candidate_paths.extend(
            [
                Path("improve_cycle.yaml"),
                Path("config/improve_cycle.yaml"),
            ]
        )

    for candidate in candidate_paths:
        if candidate.exists():
            return ImproveCycleConfig.from_yaml(candidate)

    default_path = Path(__file__).resolve().parent / "config" / "improve_cycle.yaml"
    return ImproveCycleConfig.from_yaml(default_path)
