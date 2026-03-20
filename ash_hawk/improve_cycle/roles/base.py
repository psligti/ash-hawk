from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ash_hawk.improve_cycle.models import RoleContract, RoleRuntimeConfig
from ash_hawk.improve_cycle.prompt_packs import default_prompt_pack

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


ROLE_ALLOWED_ACTIONS: dict[str, list[str]] = {
    "competitor": ["rerun baseline with candidate lessons", "produce comparison evidence"],
    "translator": ["normalize artifacts", "map findings to canonical schema"],
    "analyst": ["identify patterns", "assign severity with evidence"],
    "triage": ["classify failure category", "assign primary owner"],
    "coach": ["propose behavior changes", "scope policy and skills updates"],
    "architect": ["propose infra changes", "scope tool and harness updates"],
    "curator": ["deduplicate proposals", "approve experiment-worthy lessons"],
    "experiment_designer": ["select experiment mode", "define acceptance and rejection criteria"],
    "applier": ["create reversible change sets", "record touched surfaces"],
    "verifier": ["run checks", "recommend reject hold or promote"],
    "promotion_manager": ["issue lifecycle decisions", "set scope and rollback triggers"],
    "librarian": ["derive reusable patterns", "capture anti-patterns"],
    "historian": ["record lineage", "summarize trends"],
    "adversary": ["generate stress scenarios", "propose eval expansion"],
}


ROLE_FORBIDDEN_ACTIONS: dict[str, list[str]] = {
    "competitor": ["author final policy", "promote lessons"],
    "translator": ["root-cause ownership decisions", "proposal authoring"],
    "analyst": ["promotion decisions", "direct system rewrites"],
    "triage": ["draft final proposals", "rewrite tools directly"],
    "coach": ["new tool implementations", "harness rewrites"],
    "architect": ["behavior policy rewrites", "self-approval"],
    "curator": ["final verification", "promotion decisions"],
    "experiment_designer": ["apply changes", "promote lessons"],
    "applier": ["scope expansion", "self-approval"],
    "verifier": ["inventing new lessons", "ignoring regressions"],
    "promotion_manager": ["performing verification", "rewriting changes"],
    "librarian": ["re-approving lessons", "altering production configs"],
    "historian": ["direct production proposals", "single-run over-indexing"],
    "adversary": ["final scoring", "direct system patching"],
}


class BaseRoleAgent(ABC, Generic[TInput, TOutput]):
    def __init__(
        self,
        name: str,
        mission: str,
        model_name: str,
        temperature: float,
    ) -> None:
        self.name = name
        self.mission = mission
        self.contract = RoleContract(
            role_name=name,
            mission=mission,
            allowed_actions=ROLE_ALLOWED_ACTIONS.get(name, [f"produce {name} output"]),
            forbidden_actions=ROLE_FORBIDDEN_ACTIONS.get(name, ["cross-role approvals"]),
            decision_rules=["prefer evidence", "preserve uncertainty"],
            quality_bar=["schema valid", "role boundaries respected"],
            failure_behavior=["return partial structured output"],
            tool_access=[],
            prompt_pack=default_prompt_pack(name),
            runtime_config=RoleRuntimeConfig(
                model_name=model_name,
                temperature=temperature,
                structured_output_required=True,
            ),
        )

    @abstractmethod
    def run(self, payload: TInput) -> TOutput:
        raise NotImplementedError

    def validate_scope(self, payload: TInput) -> bool:
        """Return True if payload is within role scope."""
        return payload is not None

    def validate_output(self, output: TOutput) -> bool:
        """Return True if output meets role quality bar."""
        return output is not None
