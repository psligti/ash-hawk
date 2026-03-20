from __future__ import annotations

from ash_hawk.improve_cycle.models import AdversarialScenario, AnalystOutput, VerificationReport
from ash_hawk.improve_cycle.roles import AdversaryRole


class AdversarialScenarioService:
    def __init__(self, role: AdversaryRole | None = None) -> None:
        self._role = role or AdversaryRole()

    def generate(
        self,
        analyst_output: AnalystOutput,
        verification_reports: list[VerificationReport],
    ) -> list[AdversarialScenario]:
        return self._role.run((analyst_output, verification_reports))
