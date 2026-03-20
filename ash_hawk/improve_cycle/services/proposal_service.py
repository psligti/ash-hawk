from __future__ import annotations

from ash_hawk.improve_cycle.models import AnalystOutput, ImprovementProposal, TriageOutput
from ash_hawk.improve_cycle.roles import ArchitectRole, CoachRole
from ash_hawk.improve_cycle.routing import should_run_architect, should_run_coach


class ProposalService:
    def __init__(self) -> None:
        self._coach = CoachRole()
        self._architect = ArchitectRole()

    def generate(
        self,
        analyst_output: AnalystOutput,
        triage_output: TriageOutput,
    ) -> list[ImprovementProposal]:
        proposals: list[ImprovementProposal] = []
        if should_run_coach(triage_output):
            proposals.extend(self._coach.run((analyst_output, triage_output)))
        if should_run_architect(triage_output):
            proposals.extend(self._architect.run((analyst_output, triage_output)))
        return proposals
