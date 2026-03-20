from __future__ import annotations

from ash_hawk.improve_cycle.models import CuratedLesson
from ash_hawk.improve_cycle.storage import ImproveCycleStorage


class LessonStoreService:
    def __init__(self, storage: ImproveCycleStorage) -> None:
        self._storage = storage

    def save_lessons(self, lessons: list[CuratedLesson]) -> None:
        for lesson in lessons:
            self._storage.lessons.upsert(lesson, CuratedLesson)
