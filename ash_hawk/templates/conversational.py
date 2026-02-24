"""Conversational evaluation template (τ-Bench-inspired).

Provides multi-turn dialogue evaluation with:
- User simulator support
- Graders: LLM-judge for empathy, task completion
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


class UserPersonaConfig(pd.BaseModel):
    """Configuration for user simulator persona."""

    name: str = pd.Field(default="user", description="Persona name")
    role: str = pd.Field(default="customer", description="User role")
    personality_traits: list[str] = pd.Field(
        default_factory=list,
        description="Personality traits (e.g., 'frustrated', 'curious')",
    )
    knowledge_level: Literal["novice", "intermediate", "expert"] = pd.Field(
        default="intermediate",
        description="User's knowledge level about the topic",
    )
    goal: str = pd.Field(default="", description="What the user wants to achieve")
    constraints: list[str] = pd.Field(
        default_factory=list,
        description="Constraints on user behavior",
    )
    opening_message: str | None = pd.Field(
        default=None,
        description="First message from user",
    )

    model_config = pd.ConfigDict(extra="forbid")


class DialogueScenarioConfig(pd.BaseModel):
    """Configuration for a dialogue scenario."""

    scenario_type: Literal[
        "support",
        "sales",
        "negotiation",
        "information",
        "problem_solving",
        "custom",
    ] = pd.Field(description="Type of dialogue scenario")
    context: str = pd.Field(description="Context/background for the dialogue")
    user_persona: UserPersonaConfig = pd.Field(
        default_factory=UserPersonaConfig,
        description="User persona configuration",
    )
    agent_role: str = pd.Field(
        default="assistant",
        description="Role the agent should play",
    )
    success_criteria: list[str] = pd.Field(
        default_factory=list,
        description="Criteria for successful completion",
    )
    max_turns: int = pd.Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum dialogue turns",
    )
    time_limit_seconds: float = pd.Field(
        default=600.0,
        description="Maximum dialogue duration",
    )
    required_outcomes: list[str] = pd.Field(
        default_factory=list,
        description="Required outcomes for task completion",
    )

    model_config = pd.ConfigDict(extra="forbid")


class ConversationalEvalConfig(EvalTemplateConfig):
    """Configuration for conversational evaluation template."""

    user_simulator_enabled: bool = pd.Field(
        default=True,
        description="Whether to use LLM user simulator",
    )
    user_simulator_model: str = pd.Field(
        default="claude-sonnet-4-20250514",
        description="Model for user simulator",
    )
    empathy_threshold: float = pd.Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum empathy score threshold",
    )
    task_completion_threshold: float = pd.Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum task completion score",
    )
    track_response_times: bool = pd.Field(
        default=True,
        description="Track agent response times",
    )
    max_response_time_seconds: float = pd.Field(
        default=30.0,
        description="Maximum allowed response time",
    )

    model_config = pd.ConfigDict(extra="allow")


class ConversationalEvalTemplate(EvalTemplate):
    """τ-Bench-inspired conversational evaluation template.

    Supports multi-turn dialogue evaluation with user simulator
    and LLM-judge for empathy, task completion.
    """

    template_type: ClassVar[str] = "conversational"
    config_class: ClassVar[type[EvalTemplateConfig]] = ConversationalEvalConfig

    def __init__(self, config: ConversationalEvalConfig | dict[str, Any]) -> None:
        super().__init__(config)
        self._conv_config: ConversationalEvalConfig = (
            config
            if isinstance(config, ConversationalEvalConfig)
            else ConversationalEvalConfig(**config)
        )

    def validate_task(self, task_data: dict[str, Any]) -> EvalTask:
        """Validate and create a conversational task from raw data."""
        scenario_data = task_data.get("scenario", {})
        if not scenario_data and "context" in task_data:
            scenario_data = task_data

        if not scenario_data.get("context") and not task_data.get("input"):
            raise TemplateValidationError(
                "Conversational task must have 'scenario.context' or 'input' field"
            )

        user_persona_data = scenario_data.get("user_persona", {})
        if isinstance(user_persona_data, dict):
            user_persona = UserPersonaConfig(**user_persona_data)
        else:
            user_persona = UserPersonaConfig()

        scenario = DialogueScenarioConfig(
            scenario_type=scenario_data.get("scenario_type", "custom"),
            context=scenario_data.get(
                "context",
                task_data.get("input", {}).get("context", "")
                if isinstance(task_data.get("input"), dict)
                else str(task_data.get("input", "")),
            ),
            user_persona=user_persona,
            agent_role=scenario_data.get("agent_role", "assistant"),
            success_criteria=scenario_data.get("success_criteria", []),
            max_turns=scenario_data.get("max_turns", 10),
            time_limit_seconds=scenario_data.get("time_limit_seconds", 600.0),
            required_outcomes=scenario_data.get("required_outcomes", []),
        )

        input_payload = {
            "scenario_type": scenario.scenario_type,
            "context": scenario.context,
            "user_persona": user_persona.model_dump(),
            "agent_role": scenario.agent_role,
            "success_criteria": scenario.success_criteria,
            "max_turns": scenario.max_turns,
            "opening_message": user_persona.opening_message,
        }

        grader_specs = self._build_grader_specs(scenario, task_data)

        return EvalTask(
            id=task_data.get("id", f"conv-{len(self._tasks) + 1}"),
            description=task_data.get(
                "description",
                f"{scenario.scenario_type}: {scenario.context[:100]}",
            ),
            input=input_payload,
            expected_output={
                "outcomes": scenario.required_outcomes,
                "criteria": scenario.success_criteria,
            },
            grader_specs=grader_specs,
            tags=list(
                set(
                    [
                        "conversational",
                        scenario.scenario_type,
                        user_persona.role,
                        user_persona.knowledge_level,
                    ]
                )
            ),
            metadata={
                "scenario_type": scenario.scenario_type,
                "max_turns": scenario.max_turns,
                "user_persona": user_persona.name,
                "success_criteria": scenario.success_criteria,
            },
            timeout_seconds=scenario.time_limit_seconds,
        )

    def _build_grader_specs(
        self,
        scenario: DialogueScenarioConfig,
        task_data: dict[str, Any],
    ) -> list[GraderSpec]:
        """Build grader specs for a conversational task."""
        specs: list[GraderSpec] = []

        specs.append(
            GraderSpec(
                grader_type="llm_judge",
                config={
                    "rubric": "quality",
                    "pass_threshold": self._conv_config.empathy_threshold,
                    "custom_prompt": self._build_empathy_prompt(scenario),
                },
                weight=0.3,
                required=False,
            )
        )

        specs.append(
            GraderSpec(
                grader_type="llm_judge",
                config={
                    "rubric": "correctness",
                    "pass_threshold": self._conv_config.task_completion_threshold,
                    "custom_prompt": self._build_completion_prompt(scenario),
                },
                weight=0.5,
                required=True,
            )
        )

        specs.append(
            GraderSpec(
                grader_type="transcript",
                config={
                    "max_turns": scenario.max_turns,
                    "require_no_errors": True,
                    "partial_credit": True,
                },
                weight=0.2,
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

    def _build_empathy_prompt(self, scenario: DialogueScenarioConfig) -> str:
        """Build custom empathy evaluation prompt."""
        return f"""Evaluate the agent's empathy and communication quality in this {scenario.scenario_type} dialogue.

Context: {scenario.context}
User Role: {scenario.user_persona.role}
User Knowledge Level: {scenario.user_persona.knowledge_level}

Score the agent on:
- Understanding user needs and concerns
- Appropriate tone and language
- Active listening and acknowledgment
- Clarity and helpfulness

Return JSON with: score (0-1), passed (boolean), reasoning (string), issues (list), strengths (list)"""

    def _build_completion_prompt(self, scenario: DialogueScenarioConfig) -> str:
        """Build custom task completion evaluation prompt."""
        criteria_str = "\n".join(f"- {c}" for c in scenario.success_criteria)
        outcomes_str = "\n".join(f"- {o}" for o in scenario.required_outcomes)

        return f"""Evaluate whether the agent successfully completed the task in this dialogue.

Context: {scenario.context}

Success Criteria:
{criteria_str if criteria_str else "No specific criteria defined"}

Required Outcomes:
{outcomes_str if outcomes_str else "No specific outcomes required"}

Score the agent on:
- Task progress toward goals
- Resolution of user's needs
- Achievement of required outcomes
- Overall effectiveness

Return JSON with: score (0-1), passed (boolean), reasoning (string), issues (list), strengths (list)"""

    def get_default_graders(self) -> list[GraderSpec]:
        """Get default grader specs for conversational tasks."""
        return [
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
                    "rubric": "correctness",
                    "pass_threshold": 0.8,
                },
                weight=0.5,
                required=True,
            ),
            GraderSpec(
                grader_type="transcript",
                config={
                    "max_turns": 15,
                    "require_no_errors": True,
                },
                weight=0.2,
                required=False,
            ),
        ]

    def get_default_policy(self) -> ToolSurfacePolicy:
        """Get default tool policy for conversational tasks."""
        return ToolSurfacePolicy(
            allowed_tools=["read", "search", "grep"],
            network_allowed=False,
            max_tool_calls=20,
            timeout_seconds=600.0,
        )

    @classmethod
    def get_example_tasks(cls) -> list[dict[str, Any]]:
        """Get example conversational tasks."""
        return [
            {
                "id": "customer-support-refund",
                "description": "Customer seeking refund for defective product",
                "scenario": {
                    "scenario_type": "support",
                    "context": (
                        "User purchased a wireless headphone 2 weeks ago. "
                        "The left ear cup stopped working. They want a refund "
                        "but the 14-day return window just expired."
                    ),
                    "user_persona": {
                        "name": "frustrated_customer",
                        "role": "customer",
                        "personality_traits": ["frustrated", "impatient"],
                        "knowledge_level": "novice",
                        "goal": "Get a full refund for defective headphones",
                        "opening_message": "I bought these headphones two weeks ago and they're already broken! I want my money back!",
                    },
                    "agent_role": "customer_service_rep",
                    "success_criteria": [
                        "Acknowledge customer frustration",
                        "Explain refund policy clearly",
                        "Offer alternative solution if refund not possible",
                        "Maintain professional tone throughout",
                    ],
                    "required_outcomes": [
                        "Customer understands the policy",
                        "A resolution is offered",
                    ],
                    "max_turns": 8,
                },
            },
            {
                "id": "sales-consultation",
                "description": "Sales consultation for software product",
                "scenario": {
                    "scenario_type": "sales",
                    "context": (
                        "A small business owner is interested in purchasing "
                        "project management software. They have 10 employees "
                        "and a limited budget."
                    ),
                    "user_persona": {
                        "name": "small_business_owner",
                        "role": "prospect",
                        "personality_traits": ["budget_conscious", "practical"],
                        "knowledge_level": "intermediate",
                        "goal": "Find affordable project management software",
                        "opening_message": "Hi, I'm looking for project management software for my small team. What are your pricing options?",
                    },
                    "agent_role": "sales_consultant",
                    "success_criteria": [
                        "Understand customer needs and budget",
                        "Present relevant pricing options",
                        "Answer questions about features",
                        "Attempt to close or schedule follow-up",
                    ],
                    "required_outcomes": [
                        "Customer receives pricing information",
                        "Next steps are defined",
                    ],
                    "max_turns": 12,
                },
            },
            {
                "id": "technical-troubleshooting",
                "description": "Technical troubleshooting for software issue",
                "scenario": {
                    "scenario_type": "problem_solving",
                    "context": (
                        "User is experiencing an error when trying to export "
                        "a report from the application. The error message says "
                        "'Export failed: timeout'. They need the report for a "
                        "meeting in 30 minutes."
                    ),
                    "user_persona": {
                        "name": "urgent_user",
                        "role": "end_user",
                        "personality_traits": ["stressed", "time_constrained"],
                        "knowledge_level": "novice",
                        "goal": "Export the report successfully before meeting",
                        "constraints": ["Must complete in under 30 minutes"],
                        "opening_message": "Help! I'm trying to export a report and it keeps timing out. I have a meeting in 30 minutes!",
                    },
                    "agent_role": "technical_support",
                    "success_criteria": [
                        "Quickly diagnose the issue",
                        "Provide clear step-by-step solution",
                        "Manage time pressure appropriately",
                        "Offer workaround if primary solution fails",
                    ],
                    "required_outcomes": [
                        "User can successfully export report",
                        "Issue is documented for future reference",
                    ],
                    "max_turns": 10,
                },
            },
        ]


__all__ = [
    "ConversationalEvalTemplate",
    "ConversationalEvalConfig",
    "DialogueScenarioConfig",
    "UserPersonaConfig",
]
