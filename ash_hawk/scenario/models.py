# type-hygiene: skip-file
from __future__ import annotations

from typing import Any, Literal, TypeAlias, cast

import pydantic as pd

from ash_hawk.types import EvalOutcome

JSONValue: TypeAlias = object


class SUTConfig(pd.BaseModel):
    type: Literal["coding_agent", "agentic_sdk"] = pd.Field(
        description="Type of SUT integration",
    )
    adapter: str = pd.Field(
        description="Adapter identifier for the SUT",
    )
    config: dict[str, JSONValue] = pd.Field(
        default_factory=dict,
        description="Adapter-specific configuration",
    )

    model_config = pd.ConfigDict(extra="forbid")


class ToolingConfig(pd.BaseModel):
    allowed_tools: list[str] = pd.Field(
        default_factory=list,
        description="Allowlist of tools available to the SUT",
    )
    mocks: dict[str, JSONValue] = pd.Field(
        default_factory=dict,
        description="Mock tool responses keyed by tool name",
    )
    fault_injection: dict[str, JSONValue] = pd.Field(
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
    ordering_rules: list[dict[str, JSONValue]] = pd.Field(
        default_factory=list,
        description="Ordering rules for event sequences",
    )
    diff_assertions: list[dict[str, JSONValue]] = pd.Field(
        default_factory=list,
        description="Diff-based assertions for outputs",
    )
    output_assertions: list[dict[str, JSONValue]] = pd.Field(
        default_factory=list,
        description="Output assertions for responses",
    )

    model_config = pd.ConfigDict(extra="forbid")


class ScenarioGraderSpec(pd.BaseModel):
    grader_type: str = pd.Field(
        description="Type of grader (e.g., 'string_match', 'test_runner')",
    )
    config: dict[str, JSONValue] = pd.Field(
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
    inputs: dict[str, JSONValue] = pd.Field(
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
    workspace: dict[str, str] = pd.Field(
        default_factory=dict,
        description="Baseline workspace file contents to reset before each run",
    )

    model_config = pd.ConfigDict(extra="forbid")


class ScenarioTraceEvent(pd.BaseModel):
    schema_version: int = pd.Field(default=1, description="Trace event schema version")
    event_type: str = pd.Field(default="UnknownEvent", description="Trace event type")
    ts: str = pd.Field(default="1970-01-01T00:00:00Z", description="Timestamp for trace event")
    data: dict[str, JSONValue] = pd.Field(default_factory=dict, description="Trace event payload")

    model_config = pd.ConfigDict(extra="forbid")


class ScenarioMessage(pd.BaseModel):
    role: str = pd.Field(description="Message role")
    content: str = pd.Field(description="Message content")

    model_config = pd.ConfigDict(extra="forbid")


class ScenarioToolCall(pd.BaseModel):
    name: str = pd.Field(description="Tool name")
    arguments: dict[str, JSONValue] = pd.Field(
        default_factory=dict,
        description="Tool arguments",
    )

    model_config = pd.ConfigDict(extra="forbid")


def parse_scenario_tool_call(raw: object) -> ScenarioToolCall | None:
    if isinstance(raw, ScenarioToolCall):
        return raw
    if not isinstance(raw, dict):
        return None

    tool_name_raw = raw.get("name") or raw.get("tool") or raw.get("tool_name")
    if not isinstance(tool_name_raw, str) or not tool_name_raw.strip():
        return None

    arguments_raw = raw.get("arguments")
    if arguments_raw is None:
        arguments_raw = raw.get("input")
    if arguments_raw is None:
        arguments_raw = raw.get("args")
    if arguments_raw is None:
        arguments_raw = {}

    if not isinstance(arguments_raw, dict):
        arguments: dict[str, JSONValue] = {"value": cast(JSONValue, arguments_raw)}
    else:
        arguments = {str(key): cast(JSONValue, value) for key, value in arguments_raw.items()}

    try:
        return ScenarioToolCall.model_validate(
            {
                "name": tool_name_raw.strip(),
                "arguments": arguments,
            }
        )
    except pd.ValidationError:
        return None


class ScenarioAdapterResult(pd.BaseModel):
    final_output: str | dict[str, object] | None = pd.Field(
        default=None,
        description="Primary adapter output",
    )
    trace_events: list[ScenarioTraceEvent] = pd.Field(
        default_factory=list,
        description="Structured trace events (single source of truth)",
    )
    artifacts: dict[str, JSONValue] = pd.Field(
        default_factory=dict,
        description="Adapter artifacts",
    )
    outcome: EvalOutcome = pd.Field(
        default_factory=EvalOutcome.success,
        description="Outcome from adapter execution",
    )

    model_config = pd.ConfigDict(extra="forbid")

    def extract_messages(self) -> list[ScenarioMessage]:
        return [
            ScenarioMessage(role=cast(str, e.data["role"]), content=cast(str, e.data["content"]))
            for e in self.trace_events
            if e.event_type == "ModelMessageEvent"
            and isinstance(e.data.get("role"), str)
            and isinstance(e.data.get("content"), str)
        ]

    def extract_tool_calls(self) -> list[ScenarioToolCall]:
        return [
            ScenarioToolCall(
                name=cast(str, e.data.get("name") or e.data.get("tool", "")),
                arguments=cast(dict[str, Any], e.data.get("arguments") or e.data.get("input", {})),
            )
            for e in self.trace_events
            if e.event_type == "ToolCallEvent"
            and isinstance(e.data.get("name") or e.data.get("tool"), str)
            and (e.data.get("name") or e.data.get("tool"))
        ]


__all__ = [
    "BudgetConfig",
    "ExpectationConfig",
    "JSONValue",
    "ScenarioAdapterResult",
    "ScenarioGraderSpec",
    "ScenarioMessage",
    "ScenarioToolCall",
    "ScenarioTraceEvent",
    "ScenarioV1",
    "SUTConfig",
    "ToolingConfig",
    "parse_scenario_tool_call",
]
