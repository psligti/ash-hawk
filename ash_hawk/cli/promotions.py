from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from ash_hawk.improve_cycle.models import PromotionDecision
from ash_hawk.improve_cycle.storage import ImproveCycleStorage

console = Console()


@click.group(name="promotions")
def promotions() -> None:
    pass


@promotions.command(name="list")
@click.option("--agent", default=None, help="Filter by agent id")
def list_promotions(agent: str | None) -> None:
    store = ImproveCycleStorage()
    table = Table(title="Promotion Decisions")
    table.add_column("Decision ID", style="cyan")
    table.add_column("Lesson ID", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Scope")
    decisions = store.promotions.list_all(model_type=PromotionDecision)
    for decision in decisions:
        if agent is not None and agent not in decision.scope and agent not in decision.reason:
            continue
        table.add_row(
            decision.decision_id, decision.lesson_id, decision.status.value, decision.scope
        )
    console.print(table)
