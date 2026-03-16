"""Simplified LLM rubric grader for fast evals.

This module provides a lightweight LLM grader optimized for fast evaluations:
- Single judge (no consensus overhead)
- Minimal context (just input/output, no transcript)
- Lower max_tokens (256 vs 1024)
- Temperature 0.0 for determinism
- Pre-defined rubric template
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

import pydantic as pd

from ash_hawk.graders.base import Grader
from ash_hawk.types import (
    EvalTranscript,
    EvalTrial,
    FailureMode,
    GraderResult,
    GraderSpec,
    TokenUsage,
)

if TYPE_CHECKING:
    from dawn_kestrel.llm.client import LLMClient

logger = logging.getLogger(__name__)


_RUBRIC_PROMPT_TEMPLATE: str = """You are an evaluator grading an AI response.

## Task Input
{task_input}

## Expected Output (if provided)
{expected_output}

## Agent's Response
{agent_response}

## Rubric
{rubric}

## Output Format
You MUST respond with ONLY a valid JSON object matching this schema:
```json
{{
  "score": <float between 0.0 and 1.0>,
  "passed": <boolean>,
  "reasoning": "<brief explanation>"
}}
```

Respond with ONLY the JSON object, no additional text."""


class LLMRubricConfig(pd.BaseModel):
    """Configuration for LLM rubric grader."""

    rubric: str = pd.Field(
        description="Rubric text describing evaluation criteria",
    )
    pass_threshold: float = pd.Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for passing",
    )
    judge_model: str | None = pd.Field(
        default=None,
        description="Model to use for judging (defaults to settings default)",
    )
    judge_provider: str | None = pd.Field(
        default=None,
        description="Provider for judge model (defaults to settings default)",
    )
    temperature: float = pd.Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for judge calls (0.0 for determinism)",
    )
    max_tokens: int = pd.Field(
        default=256,
        ge=1,
        description="Max tokens for judge response (keep low for speed)",
    )
    expected_output: str | None = pd.Field(
        default=None,
        description="Expected output for comparison",
    )

    model_config = pd.ConfigDict(extra="forbid")


class LLMRubricGrader(Grader):
    """Simplified LLM rubric grader for fast evals.

    This grader uses a single LLM call with a structured rubric to evaluate
    agent responses. Optimized for speed with:
    - Single judge (no consensus)
    - Temperature 0.0 (deterministic)
    - Low max_tokens (256)
    - No transcript context

    Attributes:
        _config: Rubric configuration.
        _client: Dawn-kestrel LLM client.
    """

    def __init__(
        self,
        config: LLMRubricConfig | dict[str, Any] | None = None,
        client: LLMClient | None = None,
    ) -> None:
        """Initialize the LLM rubric grader.

        Args:
            config: Rubric configuration (dict or LLMRubricConfig).
            client: Optional pre-configured LLM client.
        """
        if config is None:
            self._config = LLMRubricConfig(rubric="Grade on correctness")
        elif isinstance(config, dict):
            self._config = LLMRubricConfig(**config)
        else:
            self._config = config

        self._client = client
        self._resolved_provider: str | None = None
        self._resolved_model: str | None = None

    @property
    def name(self) -> str:
        """Return the grader name."""
        return "llm_rubric"

    def _get_client(self) -> LLMClient:
        """Get or create the LLM client."""
        if self._client is None:
            from dawn_kestrel.core.settings import get_settings
            from dawn_kestrel.llm.client import LLMClient

            settings = get_settings()

            # Use config values or fall back to settings defaults
            provider = self._config.judge_provider or settings.get_default_provider().value
            model = self._config.judge_model or settings.get_default_model(provider)

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

    def _build_rubric_prompt(
        self,
        task_input: str,
        agent_response: str,
        expected_output: str | None = None,
    ) -> str:
        """Build the rubric prompt with task and response.

        Args:
            task_input: The input prompt.
            agent_response: The agent's response.
            expected_output: Optional expected output.

        Returns:
            Formatted rubric prompt.
        """
        return _RUBRIC_PROMPT_TEMPLATE.format(
            task_input=task_input,
            expected_output=expected_output or "Not provided",
            agent_response=agent_response,
            rubric=self._config.rubric,
        )

    def _parse_output(self, raw_output: str) -> tuple[float, bool, str]:
        """Parse the judge's JSON output.

        Args:
            raw_output: Raw output from the judge.

        Returns:
            Tuple of (score, passed, reasoning).

        Raises:
            ValueError: If output cannot be parsed.
        """
        if not raw_output or not raw_output.strip():
            raise ValueError("Judge returned empty output")

        # Extract JSON from response
        json_str = raw_output.strip()

        # Handle potential markdown code blocks
        if "```json" in json_str:
            start = json_str.find("```json") + 7
            end = json_str.find("```", start)
            if end != -1:
                json_str = json_str[start:end].strip()
        elif "```" in json_str:
            start = json_str.find("```") + 3
            end = json_str.find("```", start)
            if end != -1:
                json_str = json_str[start:end].strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse judge output as JSON: {e}") from e

        # Extract score
        score = data.get("score", 0.0)
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError:
                score = 0.0
        score = max(0.0, min(1.0, float(score)))

        # Extract passed
        passed = data.get("passed")
        if not isinstance(passed, bool):
            passed = score >= self._config.pass_threshold

        # Extract reasoning
        reasoning = data.get("reasoning", data.get("explanation", "No reasoning provided"))
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)

        return score, passed, reasoning

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Grade a trial using the LLM rubric.

        Args:
            trial: The trial being evaluated.
            transcript: The complete execution transcript.
            spec: The grader specification with configuration.

        Returns:
            A GraderResult with score, pass/fail status, and details.
        """
        start_time = time.time()

        # Merge spec config with instance config
        if spec.config:
            self._config = LLMRubricConfig(**{**self._config.model_dump(), **spec.config})

        try:
            # Get input and response
            task_input = trial.input_snapshot
            if task_input is None:
                task_input = "No task input available."
            elif isinstance(task_input, dict):
                task_input = json.dumps(task_input, indent=2)

            # Get agent response
            agent_response = transcript.agent_response
            if agent_response is None:
                agent_response = "No agent response available."
            elif isinstance(agent_response, dict):
                agent_response = json.dumps(agent_response, indent=2)

            # Get expected output from spec or config
            expected_output = spec.config.get("expected_output") or self._config.expected_output

            # Build prompt
            prompt = self._build_rubric_prompt(task_input, agent_response, expected_output)

            # Run judge
            client = self._get_client()

            from dawn_kestrel.llm.client import LLMRequestOptions

            options = LLMRequestOptions(
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                response_format={"type": "json_object"},
            )

            messages = [{"role": "user", "content": prompt}]

            response = await client.complete(messages=messages, options=options)
            raw_output = response.text

            # Parse output
            score, passed, reasoning = self._parse_output(raw_output)

            execution_time = time.time() - start_time

            # Build token usage
            token_usage = TokenUsage()
            if hasattr(response, "usage") and response.usage:
                token_usage = TokenUsage(
                    input=getattr(response.usage, "prompt_tokens", 0) or 0,
                    output=getattr(response.usage, "completion_tokens", 0) or 0,
                )

            return GraderResult(
                grader_type=self.name,
                score=score,
                passed=passed,
                details={
                    "reasoning": reasoning,
                    "rubric_preview": self._config.rubric[:200] + "..."
                    if len(self._config.rubric) > 200
                    else self._config.rubric,
                    "judge_model": self._resolved_model or self._config.judge_model or "unknown",
                    "judge_provider": self._resolved_provider
                    or self._config.judge_provider
                    or "unknown",
                },
                execution_time_seconds=execution_time,
            )

        except ImportError as e:
            execution_time = time.time() - start_time
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                error_message=f"dawn-kestrel not installed: {e}",
                details={"failure_mode": FailureMode.JUDGE_ERROR.value},
                execution_time_seconds=execution_time,
            )

        except ValueError as e:
            execution_time = time.time() - start_time
            logger.warning(f"Rubric judge output parsing failed: {e}")
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                error_message=f"Judge output parsing failed: {e}",
                details={"failure_mode": FailureMode.JUDGE_ERROR.value},
                execution_time_seconds=execution_time,
                needs_review=True,
                review_reason="Judge returned unparseable output",
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"LLM rubric judge error: {e}", exc_info=True)
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                error_message=str(e),
                details={"failure_mode": FailureMode.JUDGE_ERROR.value},
                execution_time_seconds=execution_time,
            )


__all__ = ["LLMRubricGrader", "LLMRubricConfig"]
