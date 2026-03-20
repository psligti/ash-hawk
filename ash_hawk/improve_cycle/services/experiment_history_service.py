from __future__ import annotations

from ash_hawk.improve_cycle.models import ExperimentHistorySummary
from ash_hawk.improve_cycle.storage import ImproveCycleStorage


class ExperimentHistoryService:
    def __init__(self, storage: ImproveCycleStorage) -> None:
        self._storage = storage

    def save_history(self, history: ExperimentHistorySummary) -> ExperimentHistorySummary:
        return history
