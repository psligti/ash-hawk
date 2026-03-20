from __future__ import annotations

from ash_hawk.improve_cycle.models import AnalystOutput, TriageOutput
from ash_hawk.improve_cycle.roles import TriageRole


class ClassificationService:
    def __init__(self, role: TriageRole | None = None) -> None:
        self._role = role or TriageRole()

    def classify(self, analyst_output: AnalystOutput) -> TriageOutput:
        return self._role.run(analyst_output)
