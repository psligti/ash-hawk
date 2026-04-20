from __future__ import annotations

from ash_hawk.thin_runtime.models import SkillSpec


class SkillRegistry:
    def __init__(self, skills: list[SkillSpec]) -> None:
        self._skills = {skill.name: skill for skill in skills}

    def get(self, name: str) -> SkillSpec:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise ValueError(f"Unknown thin runtime skill: {name}") from exc

    def list_skills(self) -> list[SkillSpec]:
        return list(self._skills.values())
