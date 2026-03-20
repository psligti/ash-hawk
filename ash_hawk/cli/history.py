from __future__ import annotations

import click
from rich.console import Console

from ash_hawk.improve_cycle.models import RunArtifactBundle
from ash_hawk.improve_cycle.storage import ImproveCycleStorage

console = Console()


@click.group(name="history")
def history() -> None:
    pass


@history.command(name="show")
@click.option("--experiment", required=True, help="Experiment identifier")
def show_history(experiment: str) -> None:
    store = ImproveCycleStorage()
    runs = store.runs.list_all(model_type=RunArtifactBundle)
    matching = [run for run in runs if run.experiment_id == experiment]
    if not matching:
        console.print(f"No runs found for experiment '{experiment}'")
        return
    console.print(f"Experiment: {experiment}")
    for run in matching:
        console.print(
            f"- run_id={run.run_id} agent={run.agent_id} eval_pack={run.eval_pack_id} scenarios={len(run.scenario_ids)}"
        )
