"""LLM Boolean Judge grader for fast fitness detection.

This module provides a grader that uses an LLM to answer true/false questions
about transcript content. Designed for fast fitness detection scenarios like:
- AI slop content detection
- Soul alignment checking
- Voice/principles verification
- LLM-generated content detection
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pydantic as pd

from ash_hawk.graders.base import Grader
from ash_hawk.types import EvalTranscript, EvalTrial, GraderResult, GraderSpec

if TYPE_CHECKING:
    from dawn_kestrel.llm.client import LLMClient

logger = logging.getLogger(__name__)


class BooleanJudgeConfig(pd.BaseModel):
    """Configuration for LLM boolean judge grader."""

    questions: list[str] = pd.Field(
        default_factory=list, description="List of yes/no questions to ask about the transcript"
    )
    require_all: bool = pd.Field(
        default=True,
        description="If True, all questions must pass. If False, any question passing is sufficient.",
    )
    context_max_chars: int = pd.Field(
        default=8000, description="Maximum characters of transcript context to include"
    )
    judge_model: str | None = pd.Field(
        default=None, description="Model to use for judging (defaults to settings default)"
    )
    judge_provider: str | None = pd.Field(
        default=None, description="Provider for judge model (defaults to settings default)"
    )
    temperature: float = pd.Field(
        default=0.0, description="Temperature for judge calls (0.0 for deterministic)"
    )
    max_tokens: int = pd.Field(default=256, description="Max tokens for judge response")

    model_config = pd.ConfigDict(extra="forbid")


_BOOLEAN_SYSTEM_PROMPT = """You are a precise binary classifier. You answer questions about content with only TRUE or FALSE.

Rules:
- Answer ONLY "true" or "false" (lowercase, no punctuation, no explanation)
- Be strict: if uncertain, answer false
- Evaluate each question independently
- Base answers ONLY on the provided content"""

_BOOLEAN_USER_TEMPLATE = """Content to evaluate:

{content}

Questions (answer true or false for each):
{questions}"""


class LLMBooleanJudgeGrader(Grader):
    """Grader that uses LLM to answer true/false questions about transcripts.

    Fast fitness detection grader optimized for:
    - AI slop content detection
    - Soul alignment verification
    - Voice/principles checking
    - LLM-generated content detection

    Config options:
        questions: List of yes/no questions to evaluate
        require_all: If True, all must pass; if False, any passing is sufficient
        context_max_chars: Max transcript context to include
        judge_model: Specific model for judging
        judge_provider: Provider for judge model
        temperature: Temperature for LLM calls (default 0.0)
        max_tokens: Max response tokens (default 256)
    """

    def __init__(
        self,
        config: BooleanJudgeConfig | dict[str, Any] | None = None,
        client: LLMClient | None = None,
    ) -> None:
        """Initialize the boolean judge grader.

        Args:
            config: Judge configuration
            client: Optional LLM client (will be created if not provided)
        """
        if config is None:
            self._config = BooleanJudgeConfig()
        elif isinstance(config, dict):
            self._config = BooleanJudgeConfig(**config)
        else:
            self._config = config
        self._client = client

    @property
    def name(self) -> str:
        """Return the grader name."""
        return "llm_boolean"

    def _get_client(self) -> LLMClient:
        """Get or create the LLM client."""
        if self._client is None:
            from dawn_kestrel.core.settings import get_settings
            from dawn_kestrel.llm.client import LLMClient

            settings = get_settings()
            self._client = LLMClient(settings=settings)
        return self._client

    def _extract_content(self, transcript: EvalTranscript, trial: EvalTrial) -> str:
        """Extract content from transcript for evaluation."""
        parts = []

        if transcript.agent_response:
            if isinstance(transcript.agent_response, str):
                parts.append(transcript.agent_response)
            elif isinstance(transcript.agent_response, dict):
                if "content" in transcript.agent_response:
                    parts.append(str(transcript.agent_response["content"]))

        for msg in transcript.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(f"[{role}]: {content}")

        content = "\n\n".join(parts)

        if len(content) > self._config.context_max_chars:
            content = content[: self._config.context_max_chars] + "\n...[truncated]"

        return content

    def _parse_boolean_response(self, response: str, num_questions: int) -> list[bool]:
        """Parse LLM response into boolean list."""
        response = response.strip().lower()

        results = []

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("true") or line in ("yes", "1", "y"):
                results.append(True)
            elif line.startswith("false") or line in ("no", "0", "n"):
                results.append(False)

        while len(results) < num_questions:
            results.append(False)

        return results[:num_questions]

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Grade by evaluating boolean questions against transcript content."""
        config = spec.config

        questions = config.get("questions", self._config.questions)
        require_all = config.get("require_all", self._config.require_all)

        if not questions:
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                error_message="No questions provided for boolean evaluation",
            )

        content = self._extract_content(transcript, trial)

        if not content.strip():
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                error_message="No content found in transcript to evaluate",
            )

        questions_text = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))

        user_prompt = _BOOLEAN_USER_TEMPLATE.format(
            content=content,
            questions=questions_text,
        )

        try:
            client = self._get_client()

            response = await client.chat(
                messages=[
                    {"role": "system", "content": _BOOLEAN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=self._config.judge_model,
                provider=self._config.judge_provider,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )

            if not response or not response.content:
                return GraderResult(
                    grader_type=self.name,
                    score=0.0,
                    passed=False,
                    error_message="Empty response from LLM",
                )

            raw_response = response.content
            results = self._parse_boolean_response(raw_response, len(questions))
            true_count = sum(results)
            total_count = len(results)

            if require_all:
                passed = all(results)
                score = true_count / total_count if total_count > 0 else 0.0
            else:
                passed = any(results)
                score = 1.0 if passed else 0.0

            details = {
                "questions": questions,
                "answers": ["true" if r else "false" for r in results],
                "true_count": true_count,
                "total_count": total_count,
                "require_all": require_all,
                "raw_response": raw_response,
            }

            return GraderResult(
                grader_type=self.name,
                score=score,
                passed=passed,
                details=details,
            )

        except ImportError as e:
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                error_message=f"dawn-kestrel not installed: {e}",
            )
        except Exception as e:
            logger.exception("LLM boolean judge failed")
            return GraderResult(
                grader_type=self.name,
                score=0.0,
                passed=False,
                error_message=str(e),
            )
