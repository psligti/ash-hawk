"""Structured output graders for validating response formats.

This module provides graders for validating structured outputs:
- SchemaGrader: Validates output against Pydantic models
- FormatGrader: Validates JSON, YAML, CSV format compliance
- ToolUsageGrader: Validates tool call patterns

These graders support custom validation functions for flexible evaluation.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Callable

import pydantic as pd

from ash_hawk.graders.base import Grader
from ash_hawk.types import EvalTranscript, EvalTrial, GraderResult, GraderSpec


class SchemaGrader(Grader):
    """Grader that validates output against a Pydantic model schema.

    Validates that the agent's response conforms to a specified Pydantic model,
    useful for structured output evaluation where specific fields and types
    are required.

    Config options:
        schema: Pydantic model class or dict with field specifications
        strict: If True, all fields must match exactly (default: True)
        required_fields: List of field names that must be present
        allow_extra_fields: If True, additional fields are allowed (default: False)
        custom_validator: Optional callable for additional validation
    """

    def __init__(
        self,
        schema: type[pd.BaseModel] | dict[str, Any] | None = None,
        strict: bool = True,
        required_fields: list[str] | None = None,
        allow_extra_fields: bool = False,
        custom_validator: Callable[[dict[str, Any]], bool] | None = None,
    ):
        """Initialize the SchemaGrader.

        Args:
            schema: Pydantic model class or field specification dict
            strict: Whether to enforce strict validation
            required_fields: Fields that must be present in the output
            allow_extra_fields: Whether to allow fields not in schema
            custom_validator: Optional additional validation function
        """
        self._schema = schema
        self._strict = strict
        self._required_fields = required_fields or []
        self._allow_extra_fields = allow_extra_fields
        self._custom_validator = custom_validator

    @property
    def name(self) -> str:
        """Return the grader name."""
        return "schema"

    def _parse_response(self, response: str | dict[str, Any]) -> dict[str, Any] | None:
        """Parse response string to dict if needed.

        Args:
            response: String or dict response from agent

        Returns:
            Parsed dict or None if parsing fails
        """
        if isinstance(response, dict):
            return response
        if isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return None
        return None

    def _create_dynamic_model(self, field_specs: dict[str, Any]) -> type[pd.BaseModel] | None:
        """Create a Pydantic model from field specifications.

        Args:
            field_specs: Dict mapping field names to type specifications

        Returns:
            Dynamically created Pydantic model class
        """
        try:
            fields = {}
            for field_name, field_type in field_specs.items():
                if isinstance(field_type, dict):
                    field_type = field_type.get("type", str)
                if field_type == "str":
                    fields[field_name] = (str, ...)
                elif field_type == "int":
                    fields[field_name] = (int, ...)
                elif field_type == "float":
                    fields[field_name] = (float, ...)
                elif field_type == "bool":
                    fields[field_name] = (bool, ...)
                elif field_type == "list":
                    fields[field_name] = (list, ...)
                elif field_type == "dict":
                    fields[field_name] = (dict, ...)
                else:
                    fields[field_name] = (Any, ...)

            return pd.create_model("DynamicSchema", **fields)
        except Exception:
            return None

    def _validate_schema(
        self, data: dict[str, Any], model: type[pd.BaseModel]
    ) -> tuple[bool, list[str]]:
        """Validate data against a Pydantic model.

        Args:
            data: Data to validate
            model: Pydantic model to validate against

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        try:
            model.model_validate(data)
            return True, errors
        except pd.ValidationError as e:
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                errors.append(f"{loc}: {error['msg']}")
            return False, errors

    def _check_required_fields(self, data: dict[str, Any], required: list[str]) -> list[str]:
        """Check that all required fields are present.

        Args:
            data: Data to check
            required: List of required field names

        Returns:
            List of missing field names
        """
        missing = []
        for field in required:
            if field not in data:
                missing.append(field)
        return missing

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Grade trial output against schema.

        Args:
            trial: The trial being evaluated
            transcript: The execution transcript
            spec: Grader specification with config

        Returns:
            GraderResult with validation status and details
        """
        config = spec.config

        # Get schema from config or instance
        schema = config.get("schema", self._schema)
        strict = config.get("strict", self._strict)
        required_fields = config.get("required_fields", self._required_fields)
        allow_extra = config.get("allow_extra_fields", self._allow_extra_fields)
        custom_validator = config.get("custom_validator", self._custom_validator)

        # Get response from transcript
        response = transcript.agent_response
        if response is None:
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                details={"error": "No agent response in transcript"},
            )

        # Parse response
        parsed = self._parse_response(response)
        if parsed is None:
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                details={"error": "Failed to parse response as JSON or dict"},
            )

        errors = []

        # Check required fields
        missing = self._check_required_fields(parsed, required_fields)
        if missing:
            errors.append(f"Missing required fields: {', '.join(missing)}")

        # Validate against schema if provided
        if schema is not None:
            if isinstance(schema, dict):
                model = self._create_dynamic_model(schema)
            else:
                model = schema

            if model is not None:
                is_valid, schema_errors = self._validate_schema(parsed, model)
                errors.extend(schema_errors)
            else:
                errors.append("Failed to create validation model from schema")

        # Run custom validator if provided
        if custom_validator is not None:
            try:
                if not custom_validator(parsed):
                    errors.append("Custom validation failed")
            except Exception as e:
                errors.append(f"Custom validator error: {str(e)}")

        # Calculate score and result
        passed = len(errors) == 0
        score = 1.0 if passed else 0.0

        # Partial credit for having some required fields
        if not passed and required_fields:
            present_count = len([f for f in required_fields if f in parsed])
            score = present_count / len(required_fields)

        return GraderResult(
            grader_type=self.name,
            score=score,
            passed=passed,
            details={
                "errors": errors,
                "field_count": len(parsed),
                "required_fields_present": len([f for f in required_fields if f in parsed]),
            },
        )


class FormatGrader(Grader):
    """Grader that validates output format (JSON, YAML, CSV).

    Validates that the agent's response is properly formatted according
    to the specified format type.

    Config options:
        format: One of 'json', 'yaml', 'csv' (required)
        strict: If True, parsing must succeed without errors (default: True)
        schema: Optional schema dict for additional validation
        custom_validator: Optional callable for format-specific validation
    """

    VALID_FORMATS = ("json", "yaml", "csv")

    def __init__(
        self,
        format_type: str = "json",
        strict: bool = True,
        custom_validator: Callable[[str], bool] | None = None,
    ):
        """Initialize the FormatGrader.

        Args:
            format_type: Expected format ('json', 'yaml', 'csv')
            strict: Whether to enforce strict parsing
            custom_validator: Optional additional validation function
        """
        if format_type not in self.VALID_FORMATS:
            raise ValueError(f"Invalid format '{format_type}'. Must be one of {self.VALID_FORMATS}")
        self._format_type = format_type
        self._strict = strict
        self._custom_validator = custom_validator

    @property
    def name(self) -> str:
        """Return the grader name."""
        return "format"

    def _validate_json(self, content: str) -> tuple[bool, list[str], Any]:
        """Validate JSON format.

        Args:
            content: String content to validate

        Returns:
            Tuple of (is_valid, errors, parsed_data)
        """
        errors = []
        try:
            parsed = json.loads(content)
            return True, errors, parsed
        except json.JSONDecodeError as e:
            errors.append(f"JSON parse error: {str(e)}")
            return False, errors, None

    def _validate_yaml(self, content: str) -> tuple[bool, list[str], Any]:
        """Validate YAML format.

        Args:
            content: String content to validate

        Returns:
            Tuple of (is_valid, errors, parsed_data)
        """
        errors = []
        try:
            import yaml

            parsed = yaml.safe_load(content)
            return True, errors, parsed
        except ImportError:
            errors.append("PyYAML not installed, cannot validate YAML")
            return False, errors, None
        except yaml.YAMLError as e:
            errors.append(f"YAML parse error: {str(e)}")
            return False, errors, None

    def _validate_csv(self, content: str) -> tuple[bool, list[str], Any]:
        """Validate CSV format.

        Args:
            content: String content to validate

        Returns:
            Tuple of (is_valid, errors, parsed_data)
        """
        errors = []
        try:
            reader = csv.reader(io.StringIO(content))
            rows = list(reader)
            if not rows:
                errors.append("CSV is empty")
                return False, errors, None
            return True, errors, rows
        except csv.Error as e:
            errors.append(f"CSV parse error: {str(e)}")
            return False, errors, None

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Grade trial output format.

        Args:
            trial: The trial being evaluated
            transcript: The execution transcript
            spec: Grader specification with config

        Returns:
            GraderResult with format validation status
        """
        config = spec.config

        # Get config values
        format_type = config.get("format", self._format_type)
        if format_type not in self.VALID_FORMATS:
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                details={"error": f"Invalid format type: {format_type}"},
            )

        custom_validator = config.get("custom_validator", self._custom_validator)

        # Get response from transcript
        response = transcript.agent_response
        if response is None:
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                details={"error": "No agent response in transcript"},
            )

        # Convert response to string if needed
        content = json.dumps(response) if isinstance(response, dict) else str(response)

        # Validate format
        validators = {
            "json": self._validate_json,
            "yaml": self._validate_yaml,
            "csv": self._validate_csv,
        }

        is_valid, errors, parsed = validators[format_type](content)

        # Run custom validator if provided
        if custom_validator is not None and is_valid:
            try:
                if not custom_validator(content):
                    errors.append("Custom validation failed")
                    is_valid = False
            except Exception as e:
                errors.append(f"Custom validator error: {str(e)}")
                is_valid = False

        # Calculate score
        score = 1.0 if is_valid else 0.0

        details = {
            "format": format_type,
            "is_valid": is_valid,
            "errors": errors,
        }

        if parsed is not None and format_type == "json":
            if isinstance(parsed, dict):
                details["field_count"] = len(parsed)
            elif isinstance(parsed, list):
                details["item_count"] = len(parsed)

        return GraderResult(
            grader_type=self.name,
            score=score,
            passed=is_valid,
            details=details,
        )


class ToolUsageGrader(Grader):
    """Grader that validates tool call patterns.

    Validates that the agent made appropriate tool calls during execution,
    including checking for required tools, forbidden tools, and call patterns.

    Config options:
        required_tools: List of tools that must be called
        forbidden_tools: List of tools that must not be called
        min_calls: Minimum number of tool calls required
        max_calls: Maximum number of tool calls allowed
        tool_sequence: List of tool names in expected order (partial match)
        custom_validator: Optional callable for pattern validation
    """

    def __init__(
        self,
        required_tools: list[str] | None = None,
        forbidden_tools: list[str] | None = None,
        min_calls: int = 0,
        max_calls: int | None = None,
        tool_sequence: list[str] | None = None,
        custom_validator: Callable[[list[dict[str, Any]]], bool] | None = None,
    ):
        """Initialize the ToolUsageGrader.

        Args:
            required_tools: Tools that must be called
            forbidden_tools: Tools that must not be called
            min_calls: Minimum tool calls required
            max_calls: Maximum tool calls allowed
            tool_sequence: Expected sequence of tool calls
            custom_validator: Optional custom validation function
        """
        self._required_tools = required_tools or []
        self._forbidden_tools = forbidden_tools or []
        self._min_calls = min_calls
        self._max_calls = max_calls
        self._tool_sequence = tool_sequence or []
        self._custom_validator = custom_validator

    @property
    def name(self) -> str:
        """Return the grader name."""
        return "tool_usage"

    def _get_called_tools(self, tool_calls: list[dict[str, Any]]) -> list[str]:
        """Extract list of called tool names.

        Args:
            tool_calls: List of tool call dicts

        Returns:
            List of tool names in order called
        """
        tools = []
        for call in tool_calls:
            tool_name = call.get("tool") or call.get("name") or call.get("tool_name")
            if tool_name:
                tools.append(tool_name)
        return tools

    def _check_required_tools(self, called: list[str], required: list[str]) -> list[str]:
        """Check which required tools are missing.

        Args:
            called: List of tools that were called
            required: List of required tool names

        Returns:
            List of missing required tools
        """
        return [t for t in required if t not in called]

    def _check_forbidden_tools(self, called: list[str], forbidden: list[str]) -> list[str]:
        """Check which forbidden tools were called.

        Args:
            called: List of tools that were called
            forbidden: List of forbidden tool names

        Returns:
            List of forbidden tools that were called
        """
        return [t for t in called if t in forbidden]

    def _check_sequence(self, called: list[str], expected: list[str]) -> tuple[bool, str]:
        """Check if expected sequence appears in called tools.

        Args:
            called: List of tools that were called
            expected: Expected sequence of tool names

        Returns:
            Tuple of (matches, description)
        """
        if not expected:
            return True, "No sequence requirement"

        # Find if expected sequence is a subsequence
        expected_idx = 0
        for tool in called:
            if tool == expected[expected_idx]:
                expected_idx += 1
                if expected_idx == len(expected):
                    return True, "Expected sequence found"

        return False, f"Sequence incomplete: {expected_idx}/{len(expected)} matched"

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Grade trial tool usage.

        Args:
            trial: The trial being evaluated
            transcript: The execution transcript
            spec: Grader specification with config

        Returns:
            GraderResult with tool usage validation status
        """
        config = spec.config

        # Get config values
        required_tools = config.get("required_tools", self._required_tools)
        forbidden_tools = config.get("forbidden_tools", self._forbidden_tools)
        min_calls = config.get("min_calls", self._min_calls)
        max_calls = config.get("max_calls", self._max_calls)
        tool_sequence = config.get("tool_sequence", self._tool_sequence)
        custom_validator = config.get("custom_validator", self._custom_validator)

        # Get tool calls from transcript
        tool_calls = transcript.tool_calls or []
        called_tools = self._get_called_tools(tool_calls)

        errors = []
        partial_scores = []

        # Check required tools
        missing = self._check_required_tools(called_tools, required_tools)
        if missing:
            errors.append(f"Missing required tools: {', '.join(missing)}")
            if required_tools:
                found_ratio = (len(required_tools) - len(missing)) / len(required_tools)
                partial_scores.append(found_ratio)
        elif required_tools:
            partial_scores.append(1.0)

        # Check forbidden tools
        forbidden = self._check_forbidden_tools(called_tools, forbidden_tools)
        if forbidden:
            errors.append(f"Used forbidden tools: {', '.join(forbidden)}")
            partial_scores.append(0.0)
        else:
            partial_scores.append(1.0)

        # Check call count
        call_count = len(tool_calls)
        if call_count < min_calls:
            errors.append(f"Too few calls: {call_count} < {min_calls}")
            partial_scores.append(call_count / min_calls if min_calls > 0 else 0.0)
        elif max_calls is not None and call_count > max_calls:
            errors.append(f"Too many calls: {call_count} > {max_calls}")
            partial_scores.append(max_calls / call_count)
        else:
            partial_scores.append(1.0)

        # Check sequence
        if tool_sequence:
            seq_matches, seq_desc = self._check_sequence(called_tools, tool_sequence)
            if not seq_matches:
                errors.append(f"Tool sequence mismatch: {seq_desc}")
            partial_scores.append(1.0 if seq_matches else 0.5)

        # Run custom validator if provided
        if custom_validator is not None:
            try:
                if not custom_validator(tool_calls):
                    errors.append("Custom tool validation failed")
                    partial_scores.append(0.0)
                else:
                    partial_scores.append(1.0)
            except Exception as e:
                errors.append(f"Custom validator error: {str(e)}")
                partial_scores.append(0.0)

        # Calculate score
        passed = len(errors) == 0

        if partial_scores:
            score = sum(partial_scores) / len(partial_scores)
        else:
            score = 1.0 if passed else 0.0

        return GraderResult(
            grader_type=self.name,
            score=score,
            passed=passed,
            details={
                "errors": errors,
                "total_calls": call_count,
                "called_tools": called_tools,
                "unique_tools": list(set(called_tools)),
            },
        )


__all__ = ["SchemaGrader", "FormatGrader", "ToolUsageGrader"]
