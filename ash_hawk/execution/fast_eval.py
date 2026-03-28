"""Lightweight runner for fast evaluation suites.

This module provides a fast evaluation runner optimized for:
- Direct LLM calls (skip full agent runtime)
- Minimal overhead
- Focused grading (string_match, regex, json_schema, llm_rubric)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from ash_hawk.types import (
    FastEval,
    FastEvalGraderConfig,
    FastEvalGraderType,
    FastEvalResult,
    FastEvalSuite,
    FastEvalSuiteResult,
)

logger = logging.getLogger(__name__)


def _extract_json_candidate(raw_output: str) -> str:
    candidate = raw_output.strip()
    if "```json" in candidate:
        start = candidate.find("```json") + 7
        end = candidate.find("```", start)
        if end != -1:
            return candidate[start:end].strip()
    if "```" in candidate:
        start = candidate.find("```") + 3
        end = candidate.find("```", start)
        if end != -1:
            return candidate[start:end].strip()
    return candidate


def _extract_first_json_object(candidate: str) -> str | None:
    start = candidate.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for idx, char in enumerate(candidate[start:]):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : start + idx + 1]
    return None


class FastEvalRunner:
    """Lightweight runner for fast evaluation suites.

    Executes fast evals with direct LLM calls, no agent runtime,
    focused grading, and minimal overhead.

    Attributes:
        _suite: The fast eval suite to run.
        _parallelism: Number of parallel evaluations.
        _client: Optional pre-configured LLM client.
    """

    def __init__(
        self,
        suite: FastEvalSuite,
        parallelism: int | None = None,
        client: Any | None = None,
    ) -> None:
        """Initialize the FastEvalRunner.

        Args:
            suite: The fast eval suite to run.
            parallelism: Number of parallel evaluations.
            client: Optional pre-configured LLM client.
        """
        self._suite = suite
        self._parallelism = parallelism or suite.parallelism
        self._client = client
        self._resolved_provider: str | None = None
        self._resolved_model: str | None = None

    def _get_client(self) -> Any:
        """Get or create the LLM client for fast evals."""
        if self._client is None:
            from dawn_kestrel.core.settings import get_settings
            from dawn_kestrel.llm.client import LLMClient

            settings = get_settings()
            provider = self._suite.provider or settings.get_default_provider().value
            model = self._suite.model or settings.get_default_model(provider)

            api_key_secret = settings.get_api_key_for_provider(provider)
            api_key = api_key_secret.get_secret_value() if api_key_secret else None

            self._resolved_provider = provider
            self._resolved_model = model
            self._client = LLMClient(
                provider_id=provider,
                model=model,
                api_key=api_key,
            )
        return self._client

    async def _call_llm(self, prompt: str, timeout: float) -> str:
        """Make a direct LLM call.

        Args:
            prompt: The input prompt.
            timeout: Timeout in seconds.

        Returns:
            The LLM response text.
        """
        client = self._get_client()

        from dawn_kestrel.llm.client import LLMRequestOptions

        options = LLMRequestOptions(
            temperature=0.0,
            max_tokens=1024,
        )
        messages = [{"role": "user", "content": prompt}]

        response = await asyncio.wait_for(
            client.complete(messages=messages, options=options),
            timeout=timeout,
        )

        return response.text or ""

    def _grade_string_match(
        self,
        response: str,
        expected: str | list[str] | None,
        config: FastEvalGraderConfig,
    ) -> tuple[float, bool, dict[str, Any]]:
        """Grade using string matching."""
        if expected is None:
            return 0.0, False, {"error": "No expected value provided"}

        response_to_check = response.lower() if config.case_insensitive else response
        expected_list = [expected] if isinstance(expected, str) else expected
        mode = config.mode

        if mode == "exact":
            passed = any(response_to_check == exp.lower() for exp in expected_list)
            score = 1.0 if passed else 0.0
        elif mode == "contains":
            passed = any(exp.lower() in response_to_check for exp in expected_list)
            score = 1.0 if passed else 0.0
        elif mode == "starts_with":
            passed = any(response_to_check.startswith(exp.lower()) for exp in expected_list)
            score = 1.0 if passed else 0.0
        elif mode == "ends_with":
            passed = any(response_to_check.endswith(exp.lower()) for exp in expected_list)
            score = 1.0 if passed else 0.0
        else:
            passed = False
            score = 0.0

        return score, passed, {"expected": expected, "mode": mode, "matched": passed}

    def _grade_regex(
        self,
        response: str,
        pattern: str | None,
        config: FastEvalGraderConfig,
    ) -> tuple[float, bool, dict[str, Any]]:
        """Grade using regex matching."""
        if pattern is None:
            return 0.0, False, {"error": "No pattern provided"}

        flags = re.IGNORECASE if config.case_insensitive else 0
        try:
            match = re.search(pattern, response, flags)
            passed = match is not None
            score = 1.0 if passed else 0.0
            return score, passed, {"pattern": pattern, "matched": passed}
        except re.error as e:
            return 0.0, False, {"error": f"Invalid regex: {e}"}

    def _grade_json_schema(
        self,
        response: str,
        schema: dict[str, Any] | None,
        config: FastEvalGraderConfig,
    ) -> tuple[float, bool, dict[str, Any]]:
        """Grade using JSON schema validation."""
        if schema is None:
            return 0.0, False, {"error": "No schema provided"}

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            return 0.0, False, {"error": f"Invalid JSON: {e}"}

        required = schema.get("required", [])
        properties = schema.get("properties", {})

        missing = [f for f in required if f not in data]
        if missing:
            return 0.0, False, {"error": f"Missing required fields: {missing}"}

        type_errors = []
        for field, field_schema in properties.items():
            if field in data:
                expected_type = field_schema.get("type")
                actual_value = data[field]
                if expected_type == "string" and not isinstance(actual_value, str):
                    type_errors.append(f"{field}: expected string")
                elif expected_type == "integer" and not isinstance(actual_value, int):
                    type_errors.append(f"{field}: expected integer")
                elif expected_type == "number" and not isinstance(actual_value, (int, float)):
                    type_errors.append(f"{field}: expected number")
                elif expected_type == "boolean" and not isinstance(actual_value, bool):
                    type_errors.append(f"{field}: expected boolean")
                elif expected_type == "array" and not isinstance(actual_value, list):
                    type_errors.append(f"{field}: expected array")
                elif expected_type == "object" and not isinstance(actual_value, dict):
                    type_errors.append(f"{field}: expected object")

        if type_errors:
            return 0.0, False, {"error": "Type mismatches", "details": type_errors}

        return 1.0, True, {"schema": schema, "validated": True}

    async def _grade_llm_rubric(
        self,
        response: str,
        rubric: str | None,
        task_input: str,
        config: FastEvalGraderConfig,
    ) -> tuple[float, bool, dict[str, Any]]:
        """Grade using LLM rubric evaluation."""
        if rubric is None:
            return 0.0, False, {"error": "No rubric provided"}

        judge_prompt = f"""You are an evaluator grading an AI response.

## Task Input
{task_input}

## Agent's Response
{response}

## Rubric
{rubric}

## Output Format
You MUST respond with ONLY a valid JSON object:
```json
{{
  "score": <float between 0.0 and 1.0>,
  "passed": <boolean>,
  "reasoning": "<brief explanation>"
}}
```
Respond with ONLY the JSON object, no additional text."""

        try:
            client = self._get_client()
            from dawn_kestrel.llm.client import LLMRequestOptions

            options = LLMRequestOptions(
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            messages = [{"role": "user", "content": judge_prompt}]
            judge_response = await client.complete(messages=messages, options=options)
            raw_output = judge_response.text or ""

            json_candidate = _extract_json_candidate(raw_output)
            if not json_candidate:
                raise ValueError("Judge returned empty JSON after extraction")
            try:
                data = json.loads(json_candidate)
            except json.JSONDecodeError as parse_error:
                extracted = _extract_first_json_object(json_candidate)
                if extracted is None:
                    raise ValueError(f"Failed to parse judge output as JSON: {parse_error}")
                data = json.loads(extracted)
            score = float(data.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            passed = data.get("passed")
            if not isinstance(passed, bool):
                passed = score >= config.pass_threshold
            reasoning = data.get("reasoning", "No reasoning provided")
            return score, passed, {"reasoning": reasoning, "rubric_preview": rubric[:100]}

        except Exception as e:
            return 0.0, False, {"error": f"LLM rubric grading failed: {e}"}

    async def run_single_eval(self, eval_item: FastEval) -> FastEvalResult:
        """Run a single fast eval.

        Args:
            eval_item: The fast eval to run.

        Returns:
            FastEvalResult with score and status.
        """
        start_time = time.time()

        # Get grader config - computed_field is accessed as property
        config = eval_item.grader_config
        if not isinstance(config, FastEvalGraderConfig):
            config = FastEvalGraderConfig(grader_type=config)

        # Merge with suite defaults if present
        if self._suite.defaults:
            merged = self._suite.defaults.model_dump()
            merged.update(config.model_dump())
            config = FastEvalGraderConfig(**merged)

        grader_type = config.grader_type

        try:
            # Make direct LLM call
            response = await self._call_llm(eval_item.input, eval_item.timeout_seconds)

            # Grade based on grader type
            if grader_type == FastEvalGraderType.STRING_MATCH:
                score, passed, details = self._grade_string_match(
                    response, eval_item.expected, config
                )
            elif grader_type == FastEvalGraderType.REGEX:
                score, passed, details = self._grade_regex(response, eval_item.pattern, config)
            elif grader_type == FastEvalGraderType.JSON_SCHEMA:
                json_schema = getattr(eval_item, "json_schema", None)
                score, passed, details = self._grade_json_schema(
                    response,
                    json_schema,
                    config,
                )
            elif grader_type == FastEvalGraderType.LLM_RUBRIC:
                score, passed, details = await self._grade_llm_rubric(
                    response, eval_item.rubric, eval_item.input, config
                )
            else:
                score, passed, details = (
                    0.0,
                    False,
                    {"error": f"Unknown grader type: {grader_type}"},
                )

            duration = time.time() - start_time

            return FastEvalResult(
                eval_id=eval_item.id,
                passed=passed,
                score=score,
                grader_type=grader_type.value,
                response=response[:500] if response else None,
                details=details,
                duration_seconds=duration,
            )

        except TimeoutError:
            duration = time.time() - start_time
            return FastEvalResult(
                eval_id=eval_item.id,
                passed=False,
                score=0.0,
                grader_type=grader_type.value,
                error_message="Evaluation timed out",
                details={"error": "timeout"},
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Fast eval execution failed: {e}", exc_info=True)
            return FastEvalResult(
                eval_id=eval_item.id,
                passed=False,
                score=0.0,
                grader_type=grader_type.value,
                error_message=str(e),
                details={"error": str(e)},
                duration_seconds=duration,
            )

    async def run_suite(
        self,
        filter_tags: list[str] | None = None,
        eval_ids: list[str] | None = None,
    ) -> FastEvalSuiteResult:
        """Run an entire fast eval suite.

        Args:
            filter_tags: Optional list of tags to filter evals.
            eval_ids: Optional list of eval IDs to run.

        Returns:
            FastEvalSuiteResult with all results.
        """
        # Filter evals
        evals_to_run = list(self._suite.evals)

        if filter_tags:
            evals_to_run = [e for e in evals_to_run if any(t in e.tags for t in filter_tags)]

        if eval_ids:
            evals_to_run = [e for e in evals_to_run if e.id in eval_ids]

        if not evals_to_run:
            return FastEvalSuiteResult.compute(
                suite_id=self._suite.id,
                results=[],
            )

        # Run evals in parallel with semaphore
        semaphore = asyncio.Semaphore(self._parallelism)

        async def run_eval(eval_item: FastEval) -> FastEvalResult:
            async with semaphore:
                return await self.run_single_eval(eval_item)

        results = await asyncio.gather(*[run_eval(e) for e in evals_to_run])

        return FastEvalSuiteResult.compute(
            suite_id=self._suite.id,
            results=list(results),
        )


__all__ = ["FastEvalRunner"]
