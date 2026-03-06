from __future__ import annotations

from typing import Any, Literal

import pydantic as pd


class SUTConfig(pd.BaseModel):
    type: Literal["coding_agent", "agentic_sdk"] = pd.Field(
        description="Type of SUT integration",
    )
    adapter: str = pd.Field(
        description="Adapter identifier for the SUT",
    )
    config: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Adapter-specific configuration",
    )

    model_config = pd.ConfigDict(extra="forbid")


class ToolingConfig(pd.BaseModel):
    allowed_tools: list[str] = pd.Field(
        default_factory=list,
        description="Allowlist of tools available to the SUT",
    )
    mocks: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Mock tool responses keyed by tool name",
    )
    fault_injection: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Fault injection configuration for tools",
    )

    model_config = pd.ConfigDict(extra="forbid")


class BudgetConfig(pd.BaseModel):
    max_steps: int | None = pd.Field(
        default=None,
        description="Maximum number of steps allowed",
    )
    max_tool_calls: int | None = pd.Field(
        default=None,
        description="Maximum number of tool calls allowed",
    )
    max_tokens: int | None = pd.Field(
        default=None,
        description="Maximum number of tokens allowed",
    )
    max_time_seconds: float | None = pd.Field(
        default=None,
        description="Maximum execution time in seconds",
    )

    model_config = pd.ConfigDict(extra="forbid")


class ExpectationConfig(pd.BaseModel):
    must_events: list[str] = pd.Field(
        default_factory=list,
        description="Events that must occur",
    )
    must_not_events: list[str] = pd.Field(
        default_factory=list,
        description="Events that must not occur",
    )
    ordering_rules: list[dict[str, Any]] = pd.Field(
        default_factory=list,
        description="Ordering rules for event sequences",
    )
    diff_assertions: list[dict[str, Any]] = pd.Field(
        default_factory=list,
        description="Diff-based assertions for outputs",
    )
    output_assertions: list[dict[str, Any]] = pd.Field(
        default_factory=list,
        description="Output assertions for responses",
    )

    model_config = pd.ConfigDict(extra="forbid")


class ScenarioGraderSpec(pd.BaseModel):
    grader_type: str = pd.Field(
        description="Type of grader (e.g., 'string_match', 'test_runner')",
    )
    config: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Grader-specific configuration",
    )
    weight: float = pd.Field(
        default=1.0,
        ge=0.0,
        description="Weight for this grader in aggregate scoring",
    )
    required: bool = pd.Field(
        default=False,
        description="Whether this grader must pass for overall success",
    )
    timeout_seconds: float | None = pd.Field(
        default=None,
        description="Timeout for grader execution",
    )

    model_config = pd.ConfigDict(extra="forbid")


class ScenarioV1(pd.BaseModel):
    schema_version: Literal["v1"] = pd.Field(
        description="Scenario schema version",
    )
    id: str = pd.Field(
        description="Unique scenario identifier",
    )
    description: str = pd.Field(
        default="",
        description="Human-readable scenario description",
    )
    sut: SUTConfig = pd.Field(
        description="System under test configuration",
    )
    inputs: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Scenario input payloads",
    )
    tools: ToolingConfig = pd.Field(
        default_factory=ToolingConfig,
        description="Tooling configuration",
    )
    budgets: BudgetConfig = pd.Field(
        default_factory=BudgetConfig,
        description="Resource budgets",
    )
    expectations: ExpectationConfig = pd.Field(
        default_factory=ExpectationConfig,
        description="Expected outcomes and assertions",
    )
    graders: list[ScenarioGraderSpec] = pd.Field(
        default_factory=list,
        description="Graders applied to this scenario",
    )

    model_config = pd.ConfigDict(extra="forbid")


__all__ = [
    "BudgetConfig",
    "ExpectationConfig",
    "ScenarioGraderSpec",
    "ScenarioV1",
    "SUTConfig",
    "ToolingConfig",
]
