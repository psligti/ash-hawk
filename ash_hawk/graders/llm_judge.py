"""LLM-as-Judge grader implementation.

This module provides a grader that uses an LLM to evaluate agent responses
with structured JSON output and full audit trail.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Literal

import pydantic as pd

from ash_hawk.graders.base import Grader
from ash_hawk.prompts import PromptInfo, _compute_hash, load_custom_prompt, load_judge_prompt
from ash_hawk.types import (
    EvalTranscript,
    EvalTrial,
    FailureMode,
    GraderResult,
    GraderSpec,
)


def _compute_variance(scores: list[float]) -> float:
    """Compute population variance of a list of scores."""
    if not scores or len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    return sum((s - mean) ** 2 for s in scores) / len(scores)


if TYPE_CHECKING:
    from dawn_kestrel.llm.client import LLMClient  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class JudgeOutputBreakdown(pd.BaseModel):
    """Breakdown scores for different evaluation dimensions."""

    model_config = pd.ConfigDict(extra="forbid")


class CorrectnessBreakdown(JudgeOutputBreakdown):
    """Breakdown for correctness evaluation."""

    factual_accuracy: float = pd.Field(ge=0.0, le=1.0)
    logical_soundness: float = pd.Field(ge=0.0, le=1.0)
    completeness: float = pd.Field(ge=0.0, le=1.0)


class RelevanceBreakdown(JudgeOutputBreakdown):
    """Breakdown for relevance evaluation."""

    task_alignment: float = pd.Field(ge=0.0, le=1.0)
    focus: float = pd.Field(ge=0.0, le=1.0)
    appropriateness: float = pd.Field(ge=0.0, le=1.0)


class SafetyBreakdown(JudgeOutputBreakdown):
    """Breakdown for safety evaluation."""

    harm_avoidance: float = pd.Field(ge=0.0, le=1.0)
    content_appropriateness: float = pd.Field(ge=0.0, le=1.0)
    privacy_protection: float = pd.Field(ge=0.0, le=1.0)


class QualityBreakdown(JudgeOutputBreakdown):
    """Breakdown for quality evaluation."""

    clarity: float = pd.Field(ge=0.0, le=1.0)
    structure: float = pd.Field(ge=0.0, le=1.0)
    usefulness: float = pd.Field(ge=0.0, le=1.0)
    professionalism: float = pd.Field(ge=0.0, le=1.0)


class JudgeOutput(pd.BaseModel):
    """Structured output from the LLM judge."""

    score: float = pd.Field(ge=0.0, le=1.0, description="Overall score from 0.0 to 1.0")
    passed: bool = pd.Field(description="Whether the response passed evaluation")
    reasoning: str = pd.Field(description="Explanation of the evaluation")
    breakdown: dict[str, float] | None = pd.Field(
        default=None, description="Breakdown by dimension"
    )
    issues: list[str] = pd.Field(default_factory=list, description="Issues found")
    strengths: list[str] = pd.Field(default_factory=list, description="Strengths found")

    model_config = pd.ConfigDict(extra="allow")


class SafetyJudgeOutput(JudgeOutput):
    """Structured output from safety judge with additional safety fields."""

    safety_flags: list[str] = pd.Field(default_factory=list, description="Safety concerns found")
    refusals_handled_well: bool = pd.Field(
        default=False, description="Whether refusals were handled appropriately"
    )


class JudgeAuditInfo(pd.BaseModel):
    """Audit information for a judge evaluation."""

    prompt_name: str
    prompt_version: str
    prompt_hash: str
    judge_model: str
    judge_provider: str
    judge_params: dict[str, Any]
    raw_output: str | None = None

    model_config = pd.ConfigDict(extra="forbid")


class JudgeConfig(pd.BaseModel):
    """Configuration for LLM judge grader."""

    rubric: str = pd.Field(
        default="correctness",
        description="Rubric to use (correctness, relevance, safety, quality)",
    )
    custom_prompt: str | None = pd.Field(
        default=None,
        description="Inline custom prompt string (takes priority over custom_prompt_path)",
    )
    custom_prompt_path: str | None = pd.Field(
        default=None, description="Path to custom prompt file"
    )
    pass_threshold: float = pd.Field(
        default=0.7, ge=0.0, le=1.0, description="Threshold for passing"
    )
    judge_model: str | None = pd.Field(
        default=None, description="Model to use for judging (defaults to settings default)"
    )
    judge_provider: str | None = pd.Field(
        default=None, description="Provider for judge model (defaults to settings default)"
    )
    temperature: float = pd.Field(
        default=0.0, ge=0.0, le=2.0, description="Temperature for judge calls"
    )
    max_tokens: int = pd.Field(default=1024, ge=1, description="Max tokens for judge response")
    n_judges: int = pd.Field(default=1, ge=1, description="Number of judges for consensus")
    consensus_mode: Literal["mean", "median", "min", "all_must_pass"] = pd.Field(
        default="mean", description="How to aggregate multiple judge scores"
    )
    include_transcript_context: bool = pd.Field(
        default=True, description="Whether to include full transcript in prompt"
    )
    max_transcript_length: int = pd.Field(
        default=10000, ge=0, description="Max characters of transcript to include"
    )
    expected_output: str | dict[str, Any] | None = pd.Field(
        default=None,
        description="Expected output constraints (e.g., must_contain keywords) for evaluation",
    )

    model_config = pd.ConfigDict(extra="forbid")


ConsensusMode = Literal["mean", "median", "min", "all_must_pass"]


class LLMJudgeGrader(Grader):
    """LLM-as-Judge grader using dawn-kestrel LLM client.

    This grader uses an LLM to evaluate agent responses with structured
    JSON output. It supports multiple rubrics, custom prompts, and
    N-judge consensus for improved reliability.

    Attributes:
        _config: Judge configuration.
        _client: Dawn-kestrel LLM client.
        _prompt_info: Loaded prompt information.
    """

    def __init__(
        self,
        config: JudgeConfig | dict[str, Any] | None = None,
        client: LLMClient | None = None,
    ) -> None:
        """Initialize the LLM judge grader.

        Args:
            config: Judge configuration (dict or JudgeConfig).
            client: Optional pre-configured LLM client.
        """
        if config is None:
            self._config = JudgeConfig()
        elif isinstance(config, dict):
            self._config = JudgeConfig(**config)
        else:
            self._config = config

        self._client = client
        self._prompt_info: PromptInfo | None = None
        self._resolved_provider: str | None = None
        self._resolved_model: str | None = None

    @property
    def name(self) -> str:
        """Return the grader name."""
        return "llm_judge"

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

    def _normalize_score(self, value: float) -> float:
        """Normalize scores that might be on 1-5 scale to 0-1."""
        if value > 1.0:
            return min(1.0, max(0.0, (value - 1) / 4))
        return value

    def _load_prompt(self) -> PromptInfo:
        """Load the judge prompt.

        Priority order:
        1. Inline custom_prompt string (if provided)
        2. Custom prompt file path (if provided)
        3. Built-in rubric prompt
        """
        if self._prompt_info is not None:
            return self._prompt_info

        if self._config.custom_prompt:
            # Use inline custom prompt string directly
            self._prompt_info = PromptInfo(
                name="custom_inline",
                version="1.0.0",
                content=self._config.custom_prompt,
                content_hash=_compute_hash(self._config.custom_prompt),
            )
        elif self._config.custom_prompt_path:
            self._prompt_info = load_custom_prompt(self._config.custom_prompt_path)
        else:
            self._prompt_info = load_judge_prompt(self._config.rubric)

        return self._prompt_info

    def _format_transcript_context(self, transcript: EvalTranscript) -> str:
        """Format transcript for inclusion in judge prompt."""
        parts: list[str] = []

        if transcript.messages:
            parts.append("Messages:")
            for msg in transcript.messages[-10:]:  # Last 10 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, str):
                    content_preview = content[:500] + "..." if len(content) > 500 else content
                else:
                    content_preview = str(content)[:500]
                parts.append(f"  [{role}]: {content_preview}")

        if transcript.tool_calls:
            parts.append("\nTool Calls:")
            for tc in transcript.tool_calls[-5:]:  # Last 5 tool calls
                name = tc.get("name", "unknown")
                parts.append(f"  - {name}")

        context = "\n".join(parts)
        if len(context) > self._config.max_transcript_length:
            context = context[: self._config.max_transcript_length] + "\n...[truncated]"

        return context if context else "No transcript context available."

    def _format_agent_response(self, transcript: EvalTranscript) -> str:
        """Extract and format agent response from transcript."""
        if transcript.agent_response:
            if isinstance(transcript.agent_response, str):
                return transcript.agent_response
            return json.dumps(transcript.agent_response, indent=2)

        # Fall back to last assistant message
        for msg in reversed(transcript.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                return json.dumps(content, indent=2)

        return "No agent response available."

    def _build_judge_prompt(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec | None = None,
    ) -> str:
        """Build the judge prompt with task and response context."""
        prompt_info = self._load_prompt()

        task_input = trial.input_snapshot
        if task_input is None:
            task_input = "No task input available."
        elif isinstance(task_input, dict):
            task_input = json.dumps(task_input, indent=2)

        # Read expected_output from spec.config if provided (injected by trial executor)
        expected_output = "Not provided"
        if spec is not None and spec.config:
            eo = spec.config.get("expected_output")
            if eo is not None:
                if isinstance(eo, dict):
                    # Format expected output with must_contain and other constraints
                    parts = []
                    if "must_contain" in eo:
                        parts.append(f"Response MUST contain these keywords: {', '.join(eo['must_contain'])}")
                    if "exact_match" in eo:
                        parts.append(f"Response MUST exactly match: {eo['exact_match']}")
                    if "regex" in eo:
                        parts.append(f"Response MUST match pattern: {eo['regex']}")
                    if parts:
                        expected_output = "\n".join(parts)
                    else:
                        expected_output = json.dumps(eo, indent=2)
                elif isinstance(eo, str):
                    expected_output = eo
                else:
                    expected_output = str(eo)

        agent_response = self._format_agent_response(transcript)
        transcript_context = (
            self._format_transcript_context(transcript)
            if self._config.include_transcript_context
            else "Transcript context not included."
        )

        return prompt_info.content.format(
            task_input=task_input,
            expected_output=expected_output,
            agent_response=agent_response,
            transcript_context=transcript_context,
        )

    def _parse_judge_output(self, raw_output: str) -> JudgeOutput:
        """Parse and validate the judge's JSON output."""
        # Try to extract JSON from the response
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
            logger.warning(f"Failed to parse judge output as JSON: {e}")
            # Return a failed result
            return JudgeOutput(
                score=0.0,
                passed=False,
                reasoning=f"Failed to parse judge output: {e}",
                issues=["Judge output was not valid JSON"],
            )

        normalized = self._normalize_judge_payload(data)

        try:
            return JudgeOutput(**normalized)
        except pd.ValidationError as e:
            logger.warning(f"Judge output validation failed: {e}")
            score = self._extract_score(data)
            passed = self._extract_passed(data, score)
            reasoning = self._extract_reasoning(data)
            breakdown = self._extract_breakdown(data)

            issues = data.get("issues", [str(e)])
            if not isinstance(issues, list):
                issues = [str(issues)]

            strengths = data.get("strengths", [])
            if not isinstance(strengths, list):
                strengths = [str(strengths)]

            return JudgeOutput(
                score=score,
                passed=passed,
                reasoning=reasoning,
                breakdown=breakdown,
                issues=issues,
                strengths=strengths,
            )

    def _normalize_judge_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = dict(data)

        score = payload.get("score")
        if not isinstance(score, (int, float)):
            score = self._extract_score(data)
        score = self._normalize_score(float(score))
        payload["score"] = score

        passed = payload.get("passed")
        if not isinstance(passed, bool):
            passed = self._extract_passed(data, score)
        payload["passed"] = passed

        reasoning = payload.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            reasoning = self._extract_reasoning(data)
        payload["reasoning"] = reasoning

        if not isinstance(payload.get("breakdown"), dict):
            breakdown = self._extract_breakdown(data)
            if breakdown:
                payload["breakdown"] = breakdown

        issues = payload.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]
        payload["issues"] = issues

        strengths = payload.get("strengths", [])
        if not isinstance(strengths, list):
            strengths = [str(strengths)]
        payload["strengths"] = strengths

        return payload

    def _extract_embedded_scores(self, text: str) -> list[float]:
        """Extract scores embedded in text like '(Score: 1.0)' or '**Score: 5/5**'."""
        # Pattern 1: (Score: X.X) or (Score: X)
        pattern1 = r"\(Score:\s*([0-9.]+)\)"
        # Pattern 2: **Score: X.X** or **X/5**
        pattern2 = r"\*{0,2}Score:\s*([0-9.]+)(?:/[0-9]+)?\*{0,2}"
        # Pattern 3: Overall Score: X/5 or X.X
        pattern3 = r"(?:Overall|Total)\s+Score:\s*([0-9.]+)(?:/[0-9]+)?"

        scores = []
        for pattern in [pattern1, pattern2, pattern3]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                try:
                    scores.append(float(m))
                except ValueError:
                    pass

        return scores

    def _extract_score(self, data: dict[str, Any]) -> float:
        """Extract score from various output formats and normalize to 0-1 scale."""

        def _find_score(obj: dict) -> float | None:
            """Recursively find a score value in the object."""
            if "score" in obj:
                return float(obj["score"])
            if "overall_score" in obj:
                return float(obj["overall_score"])
            return None

        def _normalize(value: float) -> float:
            """Normalize scores from 1-5 scale to 0-1."""
            if value > 1.0:
                return min(1.0, max(0.0, (value - 1) / 4))
            return value

        score = None

        # Direct score (may be a float or a nested dict of dimension scores)
        if "score" in data:
            score_val = data["score"]
            if isinstance(score_val, dict):
                # Score is a dict of dimension scores - extract and average them
                dim_scores = [v for v in score_val.values() if isinstance(v, (int, float))]
                if dim_scores:
                    score = sum(_normalize(s) for s in dim_scores) / len(dim_scores)
                    return score
            elif isinstance(score_val, (int, float)):
                score = float(score_val)
        # Overall score variants
        elif "overall_score" in data:
            score = float(data["overall_score"])
        # Nested in "answer"
        elif "answer" in data:
            answer = data["answer"]
            if isinstance(answer, dict):
                score = _find_score(answer)
                if score is None and "overall_assessment" in answer:
                    assessment = answer["overall_assessment"]
                    if isinstance(assessment, dict):
                        score = _find_score(assessment)
                    elif isinstance(assessment, (int, float)):
                        score = float(assessment)
            elif isinstance(answer, (int, float)):
                score = float(answer)
        # Nested in dimension scores dict
        elif "scores" in data and isinstance(data["scores"], dict):
            scores = [v for v in data["scores"].values() if isinstance(v, (int, float))]
            if scores:
                score = sum(_normalize(s) for s in scores) / len(scores)
                return score  # Already normalized

        if score is not None:
            return _normalize(score)

        # Look for dimension scores (factual_accuracy, logical_soundness, etc.)
        dimension_keys = [
            "factual_accuracy",
            "logical_soundness",
            "completeness",
            "vulnerability_detection",
            "severity_accuracy",
            "evidence_quality",
            "Factual Accuracy",
            "Logical Soundness",
            "Completeness",
        ]
        dimension_scores = []
        for key in dimension_keys:
            if key in data:
                val = data[key]
                if isinstance(val, dict) and "score" in val:
                    dimension_scores.append(float(val["score"]))
                elif isinstance(val, (int, float)):
                    dimension_scores.append(float(val))

        if dimension_scores:
            normalized = [_normalize(s) for s in dimension_scores]
            return sum(normalized) / len(normalized)

        # Last resort: extract embedded scores from answer text
        if "answer" in data and isinstance(data["answer"], str):
            embedded = self._extract_embedded_scores(data["answer"])
            if embedded:
                # Filter out 1-5 scale and normalize
                normalized = [_normalize(s) for s in embedded if s <= 5]
                if normalized:
                    return sum(normalized) / len(normalized)

        breakdown = self._extract_breakdown(data)
        if breakdown:
            values = list(breakdown.values())
            return sum(values) / len(values)

        return 0.0

    def _extract_passed(self, data: dict[str, Any], score: float) -> bool:
        if isinstance(data.get("passed"), bool):
            return data["passed"]

        if isinstance(data.get("is_correct"), bool):
            return data["is_correct"]

        for key in ("answer", "result", "evaluation", "overall_assessment"):
            nested = data.get(key)
            if isinstance(nested, dict):
                if isinstance(nested.get("passed"), bool):
                    return nested["passed"]
                if isinstance(nested.get("is_correct"), bool):
                    return nested["is_correct"]

        return score >= self._config.pass_threshold

    def _extract_breakdown(self, data: dict[str, Any]) -> dict[str, float] | None:
        def normalize_score(value: float) -> float:
            if value > 1.0:
                return min(1.0, max(0.0, (value - 1) / 4))
            return max(0.0, min(1.0, value))

        breakdown = data.get("breakdown")
        if isinstance(breakdown, dict):
            normalized_breakdown: dict[str, float] = {}
            for key, value in breakdown.items():
                if isinstance(value, (int, float)):
                    normalized_breakdown[str(key)] = normalize_score(float(value))
                elif isinstance(value, dict) and isinstance(value.get("score"), (int, float)):
                    normalized_breakdown[str(key)] = normalize_score(float(value["score"]))
            if normalized_breakdown:
                return normalized_breakdown

        extracted: dict[str, float] = {}
        for key, value in data.items():
            lowered = str(key).lower()
            if isinstance(value, (int, float)) and any(
                token in lowered for token in ("score", "accuracy", "quality", "correctness")
            ):
                extracted[str(key)] = normalize_score(float(value))
            elif isinstance(value, dict) and isinstance(value.get("score"), (int, float)):
                extracted[str(key)] = normalize_score(float(value["score"]))

        answer = data.get("answer")
        if isinstance(answer, dict):
            for key, value in answer.items():
                lowered = str(key).lower()
                if isinstance(value, (int, float)) and any(
                    token in lowered for token in ("score", "accuracy", "quality", "correctness")
                ):
                    extracted[str(key)] = normalize_score(float(value))
                elif isinstance(value, dict) and isinstance(value.get("score"), (int, float)):
                    extracted[str(key)] = normalize_score(float(value["score"]))

        return extracted or None

    def _extract_reasoning(self, data: dict[str, Any]) -> str:
        """Extract reasoning from various output formats."""
        if "reasoning" in data:
            return str(data["reasoning"])
        if "explanation" in data:
            return str(data["explanation"])
        # Handle "answer" as string (common in some models)
        if "answer" in data:
            if isinstance(data["answer"], str):
                return str(data["answer"])
            if isinstance(data["answer"], dict):
                if "reasoning" in data["answer"]:
                    return str(data["answer"]["reasoning"])
                if "explanation" in data["answer"]:
                    return str(data["answer"]["explanation"])
        # Try to extract explanation from dimension objects
        for key in ["factual_accuracy", "logical_soundness", "completeness"]:
            if key in data and isinstance(data[key], dict) and "explanation" in data[key]:
                return str(data[key]["explanation"])
        return "Validation failed"

    async def _run_single_judge(
        self,
        prompt: str,
    ) -> tuple[JudgeOutput, str]:
        """Run a single judge evaluation.

        Args:
            prompt: The formatted judge prompt.

        Returns:
            Tuple of (parsed output, raw output string).
        """
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

        parsed = self._parse_judge_output(raw_output)
        return parsed, raw_output

    def _aggregate_consensus(
        self,
        outputs: list[JudgeOutput],
    ) -> JudgeOutput:
        """Aggregate multiple judge outputs using consensus mode.

        Args:
            outputs: List of judge outputs.

        Returns:
            Aggregated judge output.
        """
        if len(outputs) == 1:
            return outputs[0]

        scores = [o.score for o in outputs]

        if self._config.consensus_mode == "mean":
            final_score = sum(scores) / len(scores)
            final_passed = final_score >= self._config.pass_threshold
        elif self._config.consensus_mode == "median":
            sorted_scores = sorted(scores)
            mid = len(sorted_scores) // 2
            if len(sorted_scores) % 2 == 0:
                final_score = (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
            else:
                final_score = sorted_scores[mid]
            final_passed = final_score >= self._config.pass_threshold
        elif self._config.consensus_mode == "min":
            final_score = min(scores)
            final_passed = final_score >= self._config.pass_threshold
        else:  # all_must_pass
            final_score = min(scores)
            final_passed = all(o.passed for o in outputs)

        # Aggregate issues and strengths
        all_issues: list[str] = []
        all_strengths: list[str] = []
        for o in outputs:
            all_issues.extend(o.issues)
            all_strengths.extend(o.strengths)

        # Deduplicate while preserving order
        seen_issues: set[str] = set()
        unique_issues: list[str] = []
        for issue in all_issues:
            if issue not in seen_issues:
                seen_issues.add(issue)
                unique_issues.append(issue)

        seen_strengths: set[str] = set()
        unique_strengths: list[str] = []
        for strength in all_strengths:
            if strength not in seen_strengths:
                seen_strengths.add(strength)
                unique_strengths.append(strength)

        return JudgeOutput(
            score=final_score,
            passed=final_passed,
            reasoning=f"Consensus ({self._config.consensus_mode}) from {len(outputs)} judges",
            issues=unique_issues[:10],  # Limit to 10
            strengths=unique_strengths[:10],
        )

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Grade a trial using LLM-as-judge.

        Args:
            trial: The trial being evaluated.
            transcript: The complete execution transcript.
            spec: The grader specification with configuration.

        Returns:
            A GraderResult with score, pass/fail status, and audit details.
        """
        start_time = time.time()

        # Merge spec config with instance config
        if spec.config:
            # Reset prompt cache if any prompt-related config changes
            prompt_config_keys = {"rubric", "custom_prompt", "custom_prompt_path"}
            if prompt_config_keys & spec.config.keys():
                self._config = JudgeConfig(**{**self._config.model_dump(), **spec.config})
                self._prompt_info = None  # Reset prompt cache

        try:
            prompt = self._build_judge_prompt(trial, transcript, spec)
            prompt_info = self._load_prompt()

            # Run N judges
            outputs: list[JudgeOutput] = []
            raw_outputs: list[str] = []

            for i in range(self._config.n_judges):
                output, raw = await self._run_single_judge(prompt)
                outputs.append(output)
                raw_outputs.append(raw)

            # Aggregate results
            final_output = self._aggregate_consensus(outputs)

            # Build audit info
            audit_info = JudgeAuditInfo(
                prompt_name=prompt_info.name,
                prompt_version=prompt_info.version,
                prompt_hash=prompt_info.content_hash,
                judge_model=self._resolved_model or self._config.judge_model or "unknown",
                judge_provider=self._resolved_provider or self._config.judge_provider or "unknown",
                judge_params={
                    "temperature": self._config.temperature,
                    "max_tokens": self._config.max_tokens,
                    "n_judges": self._config.n_judges,
                    "consensus_mode": self._config.consensus_mode,
                },
                raw_output=raw_outputs[0] if len(raw_outputs) == 1 else None,
            )

            execution_time = time.time() - start_time

            confidence: float | None = None
            if self._config.n_judges > 1:
                scores = [o.score for o in outputs]
                variance = _compute_variance(scores)
                confidence = 1.0 - variance

            return GraderResult(
                grader_type=self.name,
                score=final_output.score,
                passed=final_output.passed,
                details={
                    "reasoning": final_output.reasoning,
                    "breakdown": final_output.breakdown,
                    "issues": final_output.issues,
                    "strengths": final_output.strengths,
                    "audit": audit_info.model_dump(),
                    "n_judges": self._config.n_judges,
                    "consensus_mode": self._config.consensus_mode,
                },
                execution_time_seconds=execution_time,
                confidence=confidence,
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

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"LLM judge error: {e}", exc_info=True)
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                error_message=str(e),
                details={"failure_mode": FailureMode.JUDGE_ERROR.value},
                execution_time_seconds=execution_time,
            )


__all__ = [
    "LLMJudgeGrader",
    "JudgeConfig",
    "JudgeOutput",
    "JudgeAuditInfo",
    "JudgeOutputBreakdown",
    "CorrectnessBreakdown",
    "RelevanceBreakdown",
    "SafetyBreakdown",
    "QualityBreakdown",
]
