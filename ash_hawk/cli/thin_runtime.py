# mypy: disable-error-code=misc
# type-hygiene: skip-file
from __future__ import annotations

from pathlib import Path

import click

from ash_hawk.thin_runtime import RuntimeGoal, create_default_harness


@click.group(name="thin-runtime")
def thin_runtime() -> None:
    pass


@thin_runtime.command("run")
@click.argument("goal", required=False)
@click.option("--goal-id", default="thin-runtime-run", help="Stable goal identifier")
@click.option("--agent", "agent_name", default="coordinator", help="Thin runtime agent name")
@click.option("--max-iterations", default=8, type=int, help="Maximum runtime iterations")
@click.option(
    "--workdir",
    type=click.Path(exists=True),
    default=".",
    help="Working directory for the thin runtime",
)
@click.option(
    "--storage-root",
    type=click.Path(),
    default=None,
    help="Override storage root for thin runtime persistence",
)
@click.option(
    "--scenario-path",
    type=click.Path(exists=True),
    default=None,
    help="Optional scenario path to seed workspace evaluation context",
)
@click.option(
    "--skill",
    "requested_skills",
    multiple=True,
    help="Explicit skill(s) to activate",
)
@click.option(
    "--tool",
    "tool_execution_order",
    multiple=True,
    help="Explicit tool execution order override",
)
def run_thin_runtime(
    goal: str | None,
    goal_id: str,
    agent_name: str,
    max_iterations: int,
    workdir: str,
    storage_root: str | None,
    scenario_path: str | None,
    requested_skills: tuple[str, ...],
    tool_execution_order: tuple[str, ...],
) -> None:
    harness = create_default_harness(
        workdir=Path(workdir),
        storage_root=Path(storage_root) if storage_root else None,
    )

    resolved_goal = _resolve_runtime_goal(goal=goal, agent_name=agent_name, harness=harness)

    result: object = harness.execute(
        RuntimeGoal(goal_id=goal_id, description=resolved_goal, max_iterations=max_iterations),
        agent_name=agent_name,
        requested_skills=list(requested_skills) or None,
        tool_execution_order=list(tool_execution_order) or None,
        scenario_path=str(Path(scenario_path).resolve()) if scenario_path else None,
    )
    if getattr(result, "context", None) is None:
        run_id = getattr(result, "run_id", None)
        if isinstance(run_id, str) and run_id.strip():
            click.echo(run_id)


def _resolve_runtime_goal(*, goal: str | None, agent_name: str, harness: object) -> str:
    if isinstance(goal, str) and goal.strip():
        return goal.strip()

    catalog = getattr(harness, "catalog", None)
    agents = getattr(catalog, "agents", None)
    if isinstance(agents, list):
        for agent in agents:
            if getattr(agent, "name", None) != agent_name:
                continue
            default_goal_description = getattr(agent, "default_goal_description", None)
            if isinstance(default_goal_description, str) and default_goal_description.strip():
                return default_goal_description.strip()
            break

    raise click.UsageError(
        f"No goal provided and agent '{agent_name}' has no default_goal_description."
    )
