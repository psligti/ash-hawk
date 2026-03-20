from __future__ import annotations

from datetime import UTC, datetime

import click
from rich.console import Console

from ash_hawk.improve_cycle.models import RunArtifactBundle
from ash_hawk.improve_cycle.orchestrator import ImproveCycleOrchestrator

console = Console()


@click.group(name="adversary")
def adversary() -> None:
    pass


@adversary.command(name="generate")
@click.option("--agent", required=True, help="Agent identifier")
@click.option("--eval-pack", required=True, help="Eval pack identifier")
def generate_adversarial(agent: str, eval_pack: str) -> None:
    orchestrator = ImproveCycleOrchestrator()
    run_bundle = RunArtifactBundle(
        run_id=f"adv-seed-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        experiment_id=f"adv-{agent}",
        agent_id=agent,
        eval_pack_id=eval_pack,
        scenario_ids=["baseline"],
        timestamp=datetime.now(UTC).isoformat(),
    )
    result = orchestrator.run_cycle(run_bundle)
    scenarios = result.adversarial_scenarios
    if not scenarios:
        console.print("No adversarial scenarios generated")
        return
    console.print(f"Generated {len(scenarios)} adversarial scenario(s)")
    for scenario in scenarios:
        console.print(f"- {scenario.scenario_id}: {scenario.title} ({scenario.target_weakness})")
