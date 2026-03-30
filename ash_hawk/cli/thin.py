"""CLI commands for thin telemetry bridge."""

from __future__ import annotations

import asyncio
from typing import Any
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group(name="thin")
def thin() -> None:
    """Thin telemetry bridge commands.

    These commands use the new thin bridge that runs real dawn-kestrel
    agents with RuntimeHook telemetry capture.
    """
    pass


@thin.command("run")
@click.argument("scenario_path", type=click.Path(exists=True))
@click.option(
    "--agent",
    "-a",
    "agent_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to agent directory (e.g., .dawn-kestrel/agents/bolt-merlin)",
)
@click.option(
    "--max-iterations",
    "-m",
    default=10,
    help="Maximum iterations for agent execution",
)
@click.option(
    "--workdir",
    "-w",
    type=click.Path(exists=True),
    default=".",
    help="Working directory for agent execution",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    default=None,
    help="Path to save transcript JSON",
)
def run_thin(
    scenario_path: str,
    agent_path: str,
    max_iterations: int,
    workdir: str,
    output_path: str | None,
) -> None:
    """Run a scenario using the thin telemetry bridge.

    This runs the real dawn-kestrel agent with RuntimeHook telemetry
    capture, bypassing the adapter registry.
    """
    from ash_hawk.bridge import TelemetrySink, run_real_agent
    from ash_hawk.scenario.loader import load_scenario

    scenario_file = Path(scenario_path)
    scenario = load_scenario(scenario_file)

    agent_dir = Path(agent_path)
    work_dir = Path(workdir)

    console.print(f"[cyan]Loading scenario:[/cyan] {scenario.id}")
    console.print(f"[cyan]Agent:[/cyan] {agent_dir.name}")

    class CollectingSink(TelemetrySink):
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        async def on_iteration_start(self, data: dict[str, Any]) -> None:
            self.events.append({"type": "iteration_start", **data})

        async def on_iteration_end(self, data: dict[str, Any]) -> None:
            self.events.append({"type": "iteration_end", **data})

        async def on_action_decision(self, data: dict[str, Any]) -> None:
            self.events.append({"type": "action_decision", **data})

        async def on_tool_result(self, data: dict[str, Any]) -> None:
            self.events.append({"type": "tool_result", **data})

        async def on_run_complete(self, data: dict[str, Any]) -> None:
            self.events.append({"type": "run_complete", **data})

    input_text = (
        scenario.inputs.get("prompt", scenario.description)
        if scenario.inputs
        else scenario.description
    )

    sink = CollectingSink()

    async def _run() -> None:
        result = await run_real_agent(
            agent_path=agent_dir,
            input=str(input_text),
            telemetry_sink=sink,
            max_iterations=max_iterations,
            workdir=work_dir,
        )

        console.print()
        if result.outcome.success:
            console.print(f"[green]✓ Success[/green] ({result.iterations} iterations)")
        else:
            console.print(f"[red]✗ Failed:[/red] {result.outcome.error or result.outcome.message}")

        console.print(
            f"[dim]Tokens: {result.transcript.token_usage.get('input', 0)} in / {result.transcript.token_usage.get('output', 0)} out[/dim]"
        )
        console.print(f"[dim]Duration: {result.transcript.duration_seconds:.2f}s[/dim]")

        if result.transcript.tool_calls:
            console.print(f"[dim]Tools used: {len(result.transcript.tool_calls)} calls[/dim]")

        if output_path:
            import json

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            transcript_data = {
                "run_id": result.run_id,
                "outcome": {
                    "success": result.outcome.success,
                    "message": result.outcome.message,
                    "error": result.outcome.error,
                },
                "transcript": {
                    "messages": result.transcript.messages,
                    "tool_calls": result.transcript.tool_calls,
                    "token_usage": result.transcript.token_usage,
                    "duration_seconds": result.transcript.duration_seconds,
                    "agent_response": result.transcript.agent_response,
                    "error_trace": result.transcript.error_trace,
                },
                "telemetry_events": sink.events,
            }

            with open(output_file, "w") as f:
                json.dump(transcript_data, f, indent=2, default=str)

            console.print(f"[green]Transcript saved to:[/green] {output_path}")

    asyncio.run(_run())


@thin.command("improve")
@click.argument("scenario_path", type=click.Path(exists=True))
@click.argument("transcript_path", type=click.Path(exists=True))
@click.option(
    "--agent",
    "-a",
    "agent_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to agent directory to improve",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show proposed diffs without applying",
)
@click.option(
    "--backup",
    is_flag=True,
    default=True,
    help="Create backup files before applying diffs",
)
def improve_thin(
    scenario_path: str,
    transcript_path: str,
    agent_path: str,
    dry_run: bool,
    backup: bool,
) -> None:
    """Analyze a failed transcript and propose improvements.

    This reads a transcript from a failed run and generates unified
    diffs to improve the agent.
    """
    import json

    from ash_hawk.improvement.applier import DiffApplier
    from ash_hawk.improvement.improver_agent import ImprovementContext, ImproverAgent

    transcript_file = Path(transcript_path)
    agent_dir = Path(agent_path)

    with open(transcript_file) as f:
        transcript_data = json.load(f)

    failed_grades = []
    if not transcript_data.get("outcome", {}).get("success", True):
        failed_grades.append(
            {
                "grader": "outcome",
                "score": 0.0,
                "feedback": transcript_data.get("outcome", {}).get("error", "Agent failed"),
            }
        )

    context = ImprovementContext(
        run_id=transcript_data.get("run_id", "unknown"),
        scenario_path=Path(scenario_path),
        transcript_path=transcript_file,
        grade_path=transcript_file,
        agent_files_dir=agent_dir,
        failed_grades=failed_grades,
    )

    improver = ImproverAgent()

    async def _improve() -> None:
        proposals = await improver.analyze_failures(context)

        if not proposals:
            console.print("[yellow]No improvements proposed[/yellow]")
            return

        applier = DiffApplier()

        for i, proposal in enumerate(proposals, 1):
            console.print(f"\n[cyan]Proposal {i}:[/cyan] {proposal.description}")
            console.print(f"[dim]File: {proposal.file_path}[/dim]")
            console.print()

            if dry_run:
                console.print("[yellow]Proposed diff:[/yellow]")
                console.print(proposal.diff)
                continue

            result = await applier.apply(
                proposal.file_path,
                proposal.diff,
                dry_run=False,
                backup=backup,
            )

            if result.success:
                console.print(f"[green]✓ Applied to {result.file_path}[/green]")
                if result.backup_path:
                    console.print(f"[dim]Backup: {result.backup_path}[/dim]")
            else:
                console.print(f"[red]✗ Failed: {result.error}[/red]")

    asyncio.run(_improve())


__all__ = ["thin", "run_thin", "improve_thin"]
