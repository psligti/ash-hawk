from __future__ import annotations

import pydantic as pd


class EmotionDimension(pd.BaseModel):
    name: str = pd.Field(description="Dimension name used for scoring outputs")
    positive_pole: str = pd.Field(description="Positive pole label (e.g., confident)")
    negative_pole: str = pd.Field(description="Negative pole label (e.g., anxious)")
    description: str | None = pd.Field(
        default=None, description="Optional description of this emotional dimension"
    )
    anchors: dict[str, float] | None = pd.Field(
        default=None,
        description="Optional anchor examples mapped to target scores (-1.0 to 1.0)",
    )

    model_config = pd.ConfigDict(extra="forbid")


class EmotionModelConfig(pd.BaseModel):
    model: str = pd.Field(default="gemini-2.5-flash", description="LLM model for emotion scoring")
    provider: str | None = pd.Field(
        default=None, description="LLM provider (defaults to dawn-kestrel settings)"
    )
    temperature: float = pd.Field(
        default=0.0, ge=0.0, le=2.0, description="Sampling temperature for scoring"
    )
    max_concurrent: int = pd.Field(
        default=4, ge=1, description="Max concurrent LLM scoring requests"
    )

    model_config = pd.ConfigDict(extra="forbid")


DEFAULT_EMOTION_DIMENSIONS: list[EmotionDimension] = [
    EmotionDimension(
        name="confidence",
        positive_pole="confident",
        negative_pole="anxious",
        description="Perceived confidence vs uncertainty in the agent's behavior.",
        anchors={
            "assertive": 0.8,
            "steady": 0.4,
            "uncertain": -0.4,
            "anxious": -0.8,
        },
    ),
    EmotionDimension(
        name="engagement",
        positive_pole="engaged",
        negative_pole="withdrawn",
        description="Level of engagement and momentum in the interaction.",
    ),
    EmotionDimension(
        name="effectiveness",
        positive_pole="effective",
        negative_pole="stuck",
        description="Sense of progress and effectiveness in reaching the goal.",
    ),
]


class EmotionGraderConfig(pd.BaseModel):
    dimensions: list[EmotionDimension] = pd.Field(
        default_factory=lambda: list(DEFAULT_EMOTION_DIMENSIONS),
        description="Emotional dimensions to score",
    )
    model_config_ref: EmotionModelConfig = pd.Field(
        default_factory=EmotionModelConfig,
        description="Model configuration reference for scoring",
    )
    context_window: int = pd.Field(
        default=10, ge=0, description="Number of prior steps to include as context"
    )
    score_confidence: bool = pd.Field(
        default=True, description="Whether to request confidence scores per dimension"
    )
    detect_inflections: bool = pd.Field(
        default=True, description="Whether to detect inflection points in trajectories"
    )
    generate_visualization: bool = pd.Field(
        default=False, description="Whether to generate visualization data"
    )
    model_config = pd.ConfigDict(extra="forbid")


__all__ = [
    "EmotionDimension",
    "EmotionModelConfig",
    "EmotionGraderConfig",
    "DEFAULT_EMOTION_DIMENSIONS",
]
