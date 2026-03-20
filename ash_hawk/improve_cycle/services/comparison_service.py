from __future__ import annotations

from ash_hawk.improve_cycle.models import CompetitorOutput, RunArtifactBundle
from ash_hawk.improve_cycle.roles import CompetitorRole


class ComparisonService:
    def __init__(self, role: CompetitorRole | None = None) -> None:
        self._role = role or CompetitorRole()

    def compare(self, run_bundle: RunArtifactBundle) -> CompetitorOutput:
        return self._role.run(run_bundle)
