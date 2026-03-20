from __future__ import annotations

from ash_hawk.improve_cycle.models import ChangeSet, ExperimentPlan, VerificationReport
from ash_hawk.improve_cycle.roles import VerifierRole


class VerificationService:
    def __init__(self, role: VerifierRole | None = None) -> None:
        self._role = role or VerifierRole()

    def verify(self, change_set: ChangeSet, plan: ExperimentPlan) -> VerificationReport:
        return self._role.run((change_set, plan))
