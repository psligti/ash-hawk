# type-hygiene: skip-file
from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable, cast

import pydantic as pd

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_types import (
    SchemaFieldType,
    ToolCallContext,
    ToolExecutionObservability,
    ToolExecutionPayload,
    ToolFieldSpec,
    ToolSchemaSpec,
)

ToolExecutor = Callable[[ToolCall], tuple[bool, ToolExecutionPayload, str, list[str]]]


def basic_input_schema() -> ToolSchemaSpec:
    return ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="goal_id",
                type=SchemaFieldType.STRING,
                description="Goal id",
                required=True,
            ),
            ToolFieldSpec(
                name="remaining_tools",
                type=SchemaFieldType.ARRAY,
                item_type=SchemaFieldType.STRING,
                description="Remaining candidate tools",
            ),
            ToolFieldSpec(
                name="retry_count",
                type=SchemaFieldType.INTEGER,
                description="Current retry count",
            ),
        ],
        required=["goal_id"],
    )


def context_input_schema() -> ToolSchemaSpec:
    return ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="goal_id",
                type=SchemaFieldType.STRING,
                description="Goal id",
                required=True,
            ),
            ToolFieldSpec(
                name="context",
                type=SchemaFieldType.OBJECT,
                description="Typed call context",
                required=True,
            ),
        ],
        required=["goal_id", "context"],
    )


def delegation_input_schema() -> ToolSchemaSpec:
    return ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="goal_id",
                type=SchemaFieldType.STRING,
                description="Goal id",
                required=True,
            ),
            ToolFieldSpec(
                name="available_contexts",
                type=SchemaFieldType.ARRAY,
                item_type=SchemaFieldType.STRING,
                description="Currently unlocked context names",
                required=True,
            ),
            ToolFieldSpec(
                name="context",
                type=SchemaFieldType.OBJECT,
                description="Typed call context",
                required=True,
            ),
        ],
        required=["goal_id", "available_contexts", "context"],
    )


def standard_output_schema() -> ToolSchemaSpec:
    return ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="success",
                type=SchemaFieldType.BOOLEAN,
                description="Whether tool succeeded",
            ),
            ToolFieldSpec(
                name="result",
                type=SchemaFieldType.OBJECT,
                description="Main structured payload returned by the tool",
            ),
            ToolFieldSpec(
                name="message",
                type=SchemaFieldType.STRING,
                description="Human-readable execution summary",
            ),
            ToolFieldSpec(
                name="errors",
                type=SchemaFieldType.ARRAY,
                item_type=SchemaFieldType.STRING,
                description="Errors",
            ),
        ]
    )


@dataclass(frozen=True)
class ToolCommand:
    name: str
    summary: str
    when_to_use: list[str]
    when_not_to_use: list[str]
    input_schema: ToolSchemaSpec
    output_schema: ToolSchemaSpec
    side_effects: list[str]
    risk_level: str
    timeout_seconds: int
    completion_criteria: list[str]
    escalation_rules: list[str]
    executor: ToolExecutor
    model_input_schema: ToolSchemaSpec | None = None

    @property
    def dk_inputs(self) -> ToolSchemaSpec:
        return self.model_input_schema or self.input_schema

    @property
    def outputs(self) -> ToolSchemaSpec:
        return self.output_schema

    def run(self, call: ToolCall) -> ToolResult:
        input_errors = self._validate_call(call)
        if input_errors:
            return ToolResult(
                tool_name=self.name,
                success=False,
                payload=ToolExecutionPayload(
                    message="Input schema validation failed", errors=input_errors
                ),
                error="; ".join(input_errors),
            )
        started = monotonic()
        success, result, message, errors = self.executor(call)
        latency_ms = int((monotonic() - started) * 1000)
        payload = result.model_copy(
            update={
                "message": message,
                "errors": errors,
                "observability": ToolExecutionObservability(
                    tool_name=self.name,
                    latency_ms=latency_ms,
                    success=success,
                ),
            }
        )
        output_errors = self._validate_result(
            success=success, payload=payload, message=message, errors=errors
        )
        if output_errors:
            return ToolResult(
                tool_name=self.name,
                success=False,
                payload=ToolExecutionPayload(
                    message="Output schema validation failed", errors=output_errors
                ),
                error="; ".join(output_errors),
            )
        return ToolResult(
            tool_name=self.name,
            success=success,
            payload=payload,
            error="; ".join(errors) if errors else None,
        )

    def _validate_call(self, call: ToolCall) -> list[str]:
        runtime_model = self._schema_model(self.input_schema, output=False)
        runtime_payload = {
            "goal_id": call.goal_id,
            "remaining_tools": call.remaining_tools,
            "available_contexts": call.available_contexts,
            "agent_text": call.agent_text,
            "iterations": call.iterations,
            "max_iterations": call.max_iterations,
            "retry_count": call.retry_count,
            "context": call.context,
        }
        errors = list(self.input_schema.validate_payload(runtime_payload))
        if not errors:
            runtime_model.model_validate(runtime_payload)
        if self.model_input_schema is not None:
            arg_model = self._schema_model(self.model_input_schema, output=False)
            arg_payload = dict(call.tool_args)
            errors.extend(self.model_input_schema.validate_payload(arg_payload))
            if not errors:
                arg_model.model_validate(arg_payload)
        return errors

    def _validate_result(
        self,
        *,
        success: bool,
        payload: ToolExecutionPayload,
        message: str,
        errors: list[str],
    ) -> list[str]:
        model = self._schema_model(self.output_schema, output=True)
        result_payload = {
            "success": success,
            "result": payload,
            "message": message,
            "errors": errors,
        }
        schema_errors = self.output_schema.validate_payload(result_payload)
        if schema_errors:
            return schema_errors
        model.model_validate(result_payload)
        return []

    def _schema_model(self, schema: ToolSchemaSpec, *, output: bool) -> type[pd.BaseModel]:
        fields: dict[str, tuple[Any, Any]] = {}
        for field in schema.properties:
            annotation: Any = _python_type(field, output=output)
            if field.required:
                default: object = ...
            elif field.type is SchemaFieldType.ARRAY:
                default = pd.Field(default_factory=lambda: [])
            else:
                default = None
                annotation = _optional_python_type(field, output=output)
            fields[field.name] = (annotation, default)
        model: type[pd.BaseModel] = pd.create_model(
            f"{self.name.title().replace('_', '')}Schema",
            **cast(dict[str, Any], fields),
        )
        return model


def _python_type(field: ToolFieldSpec, *, output: bool) -> object:
    if field.type is SchemaFieldType.STRING:
        return str
    if field.type is SchemaFieldType.INTEGER:
        return int
    if field.type is SchemaFieldType.NUMBER:
        return float
    if field.type is SchemaFieldType.BOOLEAN:
        return bool
    if field.type is SchemaFieldType.ARRAY:
        item_type = field.item_type or SchemaFieldType.STRING
        if item_type is SchemaFieldType.STRING:
            return list[str]
        if item_type is SchemaFieldType.INTEGER:
            return list[int]
        if item_type is SchemaFieldType.NUMBER:
            return list[float]
        if item_type is SchemaFieldType.BOOLEAN:
            return list[bool]
        if item_type is SchemaFieldType.OBJECT:
            return list[dict[str, Any]]
        return list[str]
    if field.name == "context":
        return ToolCallContext
    if output and field.name == "result":
        return ToolExecutionPayload
    return dict[str, Any]


def _optional_python_type(field: ToolFieldSpec, *, output: bool) -> object:
    if field.type is SchemaFieldType.STRING:
        return str | None
    if field.type is SchemaFieldType.INTEGER:
        return int | None
    if field.type is SchemaFieldType.NUMBER:
        return float | None
    if field.type is SchemaFieldType.BOOLEAN:
        return bool | None
    if field.name == "context":
        return ToolCallContext | None
    if output and field.name == "result":
        return ToolExecutionPayload | None
    return dict[str, str] | None
