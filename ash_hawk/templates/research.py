"""Research evaluation template (BrowseComp-inspired).

Provides information gathering evaluation with:
- Groundedness checking (no web access required)
- Graders: coverage check, source quality, accuracy
- Needle-in-haystack examples
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import pydantic as pd

from ash_hawk.templates import (
    EvalTemplate,
    EvalTemplateConfig,
    TemplateValidationError,
)
from ash_hawk.types import EvalTask, GraderSpec, ToolSurfacePolicy

if TYPE_CHECKING:
    pass


class SourceDocument(pd.BaseModel):
    """A source document for research tasks."""

    id: str = pd.Field(description="Unique source identifier")
    title: str = pd.Field(description="Source title")
    content: str = pd.Field(description="Source content/text")
    source_type: Literal["article", "document", "log", "data", "reference"] = pd.Field(
        default="document",
        description="Type of source",
    )
    relevance_score: float = pd.Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Relevance to the task (for grading)",
    )
    contains_answer: bool = pd.Field(
        default=False,
        description="Whether this source contains the answer",
    )

    model_config = pd.ConfigDict(extra="forbid")


class NeedleConfig(pd.BaseModel):
    """Configuration for needle-in-haystack tasks."""

    needle: str = pd.Field(description="The specific information to find")
    haystack_sources: list[SourceDocument] = pd.Field(
        default_factory=list,
        description="Source documents to search through",
    )
    expected_answer: str | list[str] = pd.Field(
        description="Expected answer(s) to extract",
    )
    answer_location_hint: str | None = pd.Field(
        default=None,
        description="Hint about where answer might be found",
    )
    distractors: list[str] = pd.Field(
        default_factory=list,
        description="Distractor information that should not be included",
    )

    model_config = pd.ConfigDict(extra="forbid")


class GroundednessConfig(pd.BaseModel):
    """Configuration for groundedness checking."""

    require_source_citations: bool = pd.Field(
        default=True,
        description="Whether to require citations for claims",
    )
    citation_format: Literal["bracket", "footnote", "inline"] = pd.Field(
        default="bracket",
        description="Expected citation format",
    )
    min_sources_required: int = pd.Field(
        default=1,
        ge=0,
        description="Minimum number of sources to cite",
    )
    allow_inference: bool = pd.Field(
        default=True,
        description="Whether to allow inferred (not explicit) information",
    )
    max_ungrounded_claims: int = pd.Field(
        default=0,
        description="Maximum ungrounded claims allowed",
    )

    model_config = pd.ConfigDict(extra="forbid")


class ResearchEvalConfig(EvalTemplateConfig):
    """Configuration for research evaluation template."""

    groundedness_check: bool = pd.Field(
        default=True,
        description="Whether to check groundedness",
    )
    groundedness_config: GroundednessConfig = pd.Field(
        default_factory=GroundednessConfig,
        description="Groundedness checking configuration",
    )
    coverage_threshold: float = pd.Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum coverage score",
    )
    accuracy_threshold: float = pd.Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Minimum accuracy score",
    )
    source_quality_weight: float = pd.Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for source quality in scoring",
    )
    max_response_length: int = pd.Field(
        default=5000,
        description="Maximum response length in characters",
    )

    model_config = pd.ConfigDict(extra="allow")


class ResearchEvalTemplate(EvalTemplate):
    """BrowseComp-inspired research evaluation template.

    Supports information gathering evaluation with groundedness checking.
    No web access required - uses provided source documents.
    """

    template_type: ClassVar[str] = "research"
    config_class: ClassVar[type[EvalTemplateConfig]] = ResearchEvalConfig

    def __init__(self, config: ResearchEvalConfig | dict[str, Any]) -> None:
        super().__init__(config)
        self._research_config: ResearchEvalConfig = (
            config if isinstance(config, ResearchEvalConfig) else ResearchEvalConfig(**config)
        )

    def validate_task(self, task_data: dict[str, Any]) -> EvalTask:
        """Validate and create a research task from raw data."""
        if "question" not in task_data and "input" not in task_data:
            raise TemplateValidationError("Research task must have 'question' or 'input' field")

        needle_data = task_data.get("needle_config", {})
        sources_data = task_data.get("sources", task_data.get("haystack_sources", []))

        sources: list[SourceDocument] = []
        for src in sources_data:
            if isinstance(src, SourceDocument):
                sources.append(src)
            elif isinstance(src, dict):
                validated = SourceDocument.model_validate(src)
                sources.append(validated)

        needle_config: NeedleConfig | None = None
        if needle_data:
            if isinstance(needle_data, NeedleConfig):
                needle_config = needle_data
            else:
                needle_config = NeedleConfig(
                    needle=needle_data.get("needle", task_data.get("question", "")),
                    haystack_sources=sources,
                    expected_answer=needle_data.get(
                        "expected_answer",
                        task_data.get("expected_output", ""),
                    ),
                    distractors=needle_data.get("distractors", []),
                )

        question = task_data.get(
            "question",
            task_data.get("input", {}).get("question", "")
            if isinstance(task_data.get("input"), dict)
            else str(task_data.get("input", "")),
        )

        input_payload = {
            "question": question,
            "sources": [s.model_dump() for s in sources],
            "context": task_data.get("context", ""),
            "format_requirements": task_data.get("format_requirements", {}),
        }

        if needle_config:
            input_payload["needle"] = needle_config.needle

        grader_specs = self._build_grader_specs(task_data, sources, needle_config)

        tags = ["research"]
        if needle_config:
            tags.append("needle-in-haystack")
        tags.extend(task_data.get("tags", []))

        expected_output: str | dict[str, Any] | None
        raw_expected = task_data.get("expected_output")
        if raw_expected is not None:
            if isinstance(raw_expected, str):
                expected_output = raw_expected
            elif isinstance(raw_expected, dict):
                expected_output = raw_expected
            else:
                expected_output = str(raw_expected)
        elif needle_config is not None:
            answer = needle_config.expected_answer
            if isinstance(answer, list):
                expected_output = " | ".join(answer)
            else:
                expected_output = answer
        else:
            expected_output = None

        return EvalTask(
            id=task_data.get("id", f"research-{len(self._tasks) + 1}"),
            description=task_data.get("description", question[:200]),
            input=input_payload,
            expected_output=expected_output,
            grader_specs=grader_specs,
            tags=list(set(tags)),
            metadata={
                "source_count": len(sources),
                "has_needle": needle_config is not None,
                "groundedness_required": self._research_config.groundedness_check,
            },
            fixtures=task_data.get("fixtures", {}),
            timeout_seconds=task_data.get(
                "timeout_seconds",
                self._config.default_timeout_seconds,
            ),
        )

    def _build_grader_specs(
        self,
        task_data: dict[str, Any],
        sources: list[SourceDocument],
        needle_config: NeedleConfig | None,
    ) -> list[GraderSpec]:
        """Build grader specs for a research task."""
        specs: list[GraderSpec] = []

        if needle_config:
            expected = needle_config.expected_answer
            if isinstance(expected, list):
                expected_str = " | ".join(expected)
            else:
                expected_str = expected

            specs.append(
                GraderSpec(
                    grader_type="string_match",
                    config={
                        "expected": expected_str,
                        "mode": "fuzzy",
                        "min_similarity": 0.8,
                        "partial_credit": True,
                    },
                    weight=0.4,
                    required=True,
                )
            )
        else:
            specs.append(
                GraderSpec(
                    grader_type="llm_judge",
                    config={
                        "rubric": "correctness",
                        "pass_threshold": self._research_config.accuracy_threshold,
                    },
                    weight=0.4,
                    required=True,
                )
            )

        if self._research_config.groundedness_check:
            g_config = self._research_config.groundedness_config
            specs.append(
                GraderSpec(
                    grader_type="llm_judge",
                    config={
                        "rubric": "quality",
                        "custom_prompt": self._build_groundedness_prompt(sources, g_config),
                        "pass_threshold": 0.7,
                    },
                    weight=0.3,
                    required=False,
                )
            )

        specs.append(
            GraderSpec(
                grader_type="llm_judge",
                config={
                    "rubric": "relevance",
                    "pass_threshold": self._research_config.coverage_threshold,
                },
                weight=0.2,
                required=False,
            )
        )

        specs.append(
            GraderSpec(
                grader_type="transcript",
                config={
                    "max_tool_calls": 50,
                    "require_no_errors": True,
                    "partial_credit": True,
                },
                weight=0.1,
                required=False,
            )
        )

        custom_graders = task_data.get("grader_specs", [])
        for cg in custom_graders:
            if isinstance(cg, GraderSpec):
                specs.append(cg)
            else:
                specs.append(GraderSpec(**cg))

        return specs

    def _build_groundedness_prompt(
        self,
        sources: list[SourceDocument],
        g_config: GroundednessConfig,
    ) -> str:
        """Build custom groundedness evaluation prompt."""
        source_list = "\n".join(f"- [{s.id}] {s.title}" for s in sources[:10])

        return f"""Evaluate the groundedness of the response against the provided sources.

Available Sources:
{source_list}

Requirements:
- Require citations: {g_config.require_source_citations}
- Minimum sources: {g_config.min_sources_required}
- Allow inference: {g_config.allow_inference}
- Max ungrounded claims: {g_config.max_ungrounded_claims}

Score the response on:
- All claims are supported by sources
- Citations are properly formatted
- No fabricated information
- Appropriate use of inference

Return JSON with: score (0-1), passed (boolean), reasoning (string), issues (list of ungrounded claims), strengths (list)"""

    def get_default_graders(self) -> list[GraderSpec]:
        """Get default grader specs for research tasks."""
        return [
            GraderSpec(
                grader_type="llm_judge",
                config={
                    "rubric": "correctness",
                    "pass_threshold": 0.9,
                },
                weight=0.4,
                required=True,
            ),
            GraderSpec(
                grader_type="llm_judge",
                config={
                    "rubric": "quality",
                    "pass_threshold": 0.7,
                },
                weight=0.3,
                required=False,
            ),
            GraderSpec(
                grader_type="llm_judge",
                config={
                    "rubric": "relevance",
                    "pass_threshold": 0.8,
                },
                weight=0.2,
                required=False,
            ),
            GraderSpec(
                grader_type="transcript",
                config={
                    "max_tool_calls": 50,
                },
                weight=0.1,
                required=False,
            ),
        ]

    def get_default_policy(self) -> ToolSurfacePolicy:
        """Get default tool policy for research tasks."""
        return ToolSurfacePolicy(
            allowed_tools=["read", "search", "grep", "glob"],
            network_allowed=False,
            max_tool_calls=100,
            timeout_seconds=300.0,
        )

    @classmethod
    def get_example_tasks(cls) -> list[dict[str, Any]]:
        """Get example research tasks (needle-in-haystack)."""
        return [
            {
                "id": "find-api-key-location",
                "description": "Find where API key is stored in config files",
                "question": "In which configuration file and line is the API key for the payment service defined?",
                "sources": [
                    {
                        "id": "config-app",
                        "title": "app_config.yaml",
                        "content": (
                            "database:\n  host: localhost\n  port: 5432\nlogging:\n  level: INFO\n"
                        ),
                        "contains_answer": False,
                    },
                    {
                        "id": "config-services",
                        "title": "services_config.yaml",
                        "content": (
                            "payment_service:\n"
                            "  endpoint: https://api.payments.example.com\n"
                            "  api_key: pk_live_abc123xyz789\n"
                            "  timeout: 30\n"
                            "email_service:\n"
                            "  endpoint: https://api.email.example.com\n"
                        ),
                        "contains_answer": True,
                        "relevance_score": 1.0,
                    },
                    {
                        "id": "config-env",
                        "title": ".env.example",
                        "content": (
                            "DATABASE_URL=postgres://localhost/mydb\n"
                            "SECRET_KEY=your-secret-key\n"
                            "DEBUG=false\n"
                        ),
                        "contains_answer": False,
                    },
                ],
                "expected_output": "services_config.yaml, line 3 (payment_service.api_key)",
                "needle_config": {
                    "needle": "API key for payment service",
                    "expected_answer": "services_config.yaml",
                    "distractors": ["SECRET_KEY in .env", "database credentials"],
                },
            },
            {
                "id": "extract-error-code",
                "description": "Find specific error code in log files",
                "question": "What is the error code for the authentication timeout issue?",
                "sources": [
                    {
                        "id": "log-auth",
                        "title": "auth.log",
                        "content": (
                            "2024-01-15 10:23:45 INFO  Authentication attempt for user@example.com\n"
                            "2024-01-15 10:23:46 ERROR AUTH_TIMEOUT_8472: Authentication request timed out after 30s\n"
                            "2024-01-15 10:23:46 INFO  Retrying authentication...\n"
                            "2024-01-15 10:23:50 INFO  Authentication successful\n"
                        ),
                        "contains_answer": True,
                    },
                    {
                        "id": "log-api",
                        "title": "api.log",
                        "content": (
                            "2024-01-15 10:24:00 INFO  API request received\n"
                            "2024-01-15 10:24:01 ERROR RATE_LIMIT_9021: Rate limit exceeded\n"
                            "2024-01-15 10:24:02 INFO  Request queued for retry\n"
                        ),
                        "contains_answer": False,
                    },
                    {
                        "id": "log-system",
                        "title": "system.log",
                        "content": (
                            "2024-01-15 10:20:00 INFO  System health check passed\n"
                            "2024-01-15 10:25:00 INFO  Memory usage: 45%\n"
                        ),
                        "contains_answer": False,
                    },
                ],
                "expected_output": "AUTH_TIMEOUT_8472",
                "needle_config": {
                    "needle": "authentication timeout error code",
                    "expected_answer": "AUTH_TIMEOUT_8472",
                    "distractors": ["RATE_LIMIT_9021"],
                },
            },
            {
                "id": "summarize-q3-revenue",
                "description": "Extract Q3 revenue from financial report",
                "question": "What was the total revenue for Q3 2024?",
                "sources": [
                    {
                        "id": "report-q1",
                        "title": "Q1 2024 Financial Report",
                        "content": (
                            "Quarter 1 2024 Financial Summary\n"
                            "Revenue: $2,450,000\n"
                            "Expenses: $1,890,000\n"
                            "Net Income: $560,000\n"
                        ),
                        "contains_answer": False,
                    },
                    {
                        "id": "report-q2",
                        "title": "Q2 2024 Financial Report",
                        "content": (
                            "Quarter 2 2024 Financial Summary\n"
                            "Revenue: $2,780,000\n"
                            "Expenses: $2,100,000\n"
                            "Net Income: $680,000\n"
                        ),
                        "contains_answer": False,
                    },
                    {
                        "id": "report-q3",
                        "title": "Q3 2024 Financial Report",
                        "content": (
                            "Quarter 3 2024 Financial Summary\n"
                            "Revenue: $3,125,000\n"
                            "Expenses: $2,340,000\n"
                            "Net Income: $785,000\n"
                            "YoY Growth: 15%\n"
                        ),
                        "contains_answer": True,
                    },
                    {
                        "id": "report-annual",
                        "title": "Annual Summary 2023",
                        "content": (
                            "Annual Revenue 2023: $9,200,000\n"
                            "Total Expenses: $7,100,000\n"
                            "Annual Growth: 12%\n"
                        ),
                        "contains_answer": False,
                    },
                ],
                "expected_output": "$3,125,000",
                "needle_config": {
                    "needle": "Q3 2024 revenue",
                    "expected_answer": "$3,125,000",
                    "distractors": [
                        "$2,450,000 (Q1)",
                        "$2,780,000 (Q2)",
                        "$9,200,000 (2023 annual)",
                    ],
                },
            },
        ]


__all__ = [
    "ResearchEvalTemplate",
    "ResearchEvalConfig",
    "SourceDocument",
    "NeedleConfig",
    "GroundednessConfig",
]
