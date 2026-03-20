from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from ash_hawk.improve_cycle.models import CuratedLesson
from ash_hawk.improve_cycle.storage import ImproveCycleStorage

console = Console()


@click.group(name="lessons")
def lessons() -> None:
    pass


@lessons.command(name="list")
@click.option("--agent", default=None, help="Filter by agent id")
def list_lessons(agent: str | None) -> None:
    store = ImproveCycleStorage()
    table = Table(title="Lessons")
    table.add_column("Lesson ID", style="cyan")
    table.add_column("Proposal ID", style="magenta")
    table.add_column("Target Surface", style="green")
    table.add_column("Risk")
    lessons = store.lessons.list_all(model_type=CuratedLesson)
    for lesson in lessons:
        if agent is not None and agent not in lesson.summary and agent not in lesson.title:
            continue
        table.add_row(
            lesson.lesson_id, lesson.proposal_id, lesson.target_surface, lesson.risk_level.value
        )
    console.print(table)
