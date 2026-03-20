from __future__ import annotations

from pathlib import Path

from ash_hawk.improve_cycle.models import RolePromptPack


def default_prompt_pack(role_name: str) -> RolePromptPack:
    base = Path(__file__).resolve().parent / "prompts"
    role_base = base / "roles" / role_name
    system_prompt = role_base / "system.md"
    task_template = role_base / "task_template.md"
    rubric = role_base / "rubric.md"
    strong = role_base / "examples" / "strong_01.json"
    weak = role_base / "examples" / "weak_01.json"

    if not system_prompt.exists() or not task_template.exists() or not rubric.exists():
        system_prompt = base / "system.md"
        task_template = base / "task_template.md"
        rubric = base / "rubric.md"
        strong = base / "examples" / "strong_01.json"
        weak = base / "examples" / "weak_01.json"

    return RolePromptPack(
        system_prompt_path=str(system_prompt),
        task_template_path=str(task_template),
        rubric_path=str(rubric),
        example_paths=[str(strong), str(weak)],
    )


ROLE_NAMES = [
    "competitor",
    "translator",
    "analyst",
    "triage",
    "coach",
    "architect",
    "curator",
    "experiment_designer",
    "applier",
    "verifier",
    "promotion_manager",
    "librarian",
    "historian",
    "adversary",
]
