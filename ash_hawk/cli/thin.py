"""CLI commands for thin telemetry bridge."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

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
    default=None,
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
@click.option(
    "--grade",
    "-g",
    is_flag=True,
    default=False,
    help="Run graders defined in scenario YAML",
)
def run_thin(
    scenario_path: str,
    agent_path: str | None,
    max_iterations: int,
    workdir: str,
    output_path: str | None,
    grade: bool,
) -> None:
    """Run a scenario using the thin telemetry bridge.

    This runs the real dawn-kestrel agent with RuntimeHook telemetry
    capture, bypassing the adapter registry.
    """
    from ash_hawk.scenario.loader import load_scenario
    from ash_hawk.scenario.thin_runner import ThinScenarioRunner

    scenario_file = Path(scenario_path)
    scenario = load_scenario(scenario_file)
    work_dir = Path(workdir)

    console.print(f"[cyan]Loading scenario:[/cyan] {scenario.id}")

    if grade and scenario.graders:
        console.print(f"[cyan]Graders:[/cyan] {len(scenario.graders)} defined")

    runner = ThinScenarioRunner(
        workdir=work_dir,
        max_iterations=max_iterations,
    )

    async def _run() -> None:
        if grade:
            result = await runner.run_with_grading(scenario, scenario_file)
            _display_graded_result(result, output_path)
        else:
            run_result = await runner.run_scenario(scenario, scenario_file)
            _display_run_result(run_result, output_path)

    asyncio.run(_run())


def _display_run_result(result: Any, output_path: str | None) -> None:
    """Display run result without grading."""
    console.print()
    if result.outcome.success:
        console.print(f"[green]✓ Success[/green] ({result.iterations} iterations)")
    else:
        console.print(f"[red]✗ Failed:[/red] {result.outcome.error or result.outcome.message}")

    console.print(
        f"[dim]Tokens: {result.transcript.token_usage.get('input', 0)} in / "
        f"{result.transcript.token_usage.get('output', 0)} out[/dim]"
    )
    console.print(f"[dim]Duration: {result.transcript.duration_seconds:.2f}s[/dim]")

    if result.transcript.tool_calls:
        console.print(f"[dim]Tools used: {len(result.transcript.tool_calls)} calls[/dim]")

    if output_path:
        _save_transcript_json(result, output_path, grader_results=None)


def _display_graded_result(graded_result: Any, output_path: str | None) -> None:
    """Display graded result with rich table."""
    result = graded_result.run_result
    grader_results = graded_result.grader_results

    console.print()
    if result.outcome.success:
        console.print(f"[green]✓ Run completed[/green] ({result.iterations} iterations)")
    else:
        console.print(f"[red]✗ Run failed:[/red] {result.outcome.error or result.outcome.message}")

    console.print(
        f"[dim]Tokens: {result.transcript.token_usage.get('input', 0)} in / "
        f"{result.transcript.token_usage.get('output', 0)} out[/dim]"
    )
    console.print(f"[dim]Duration: {result.transcript.duration_seconds:.2f}s[/dim]")

    if grader_results:
        console.print()
        _display_grader_table(grader_results)

        if graded_result.all_passed():
            console.print()
            console.print(f"[green]✓ All {len(grader_results)} graders passed[/green]")
        else:
            passed = sum(1 for g in grader_results if g.passed)
            console.print()
            console.print(
                f"[yellow]⚠ {passed}/{len(grader_results)} graders passed "
                f"(score: {graded_result.aggregate_score:.2f})[/yellow]"
            )

    if output_path:
        _save_transcript_json(result, output_path, grader_results=grader_results)


def _display_grader_table(grader_results: list[Any]) -> None:
    """Display grader results in a rich table."""
    table = Table(title="Grader Results")
    table.add_column("Grader", style="cyan")
    table.add_column("Passed", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Details")

    for grader in grader_results:
        passed_str = "[green]✓[/green]" if grader.passed else "[red]✗[/red]"
        score_str = f"{grader.score:.2f}"

        details = ""
        if grader.error_message:
            details = f"[red]{grader.error_message}[/red]"
        elif hasattr(grader, "rationale") and grader.rationale:
            details = (
                grader.rationale[:50] + "..." if len(grader.rationale) > 50 else grader.rationale
            )

        table.add_row(
            grader.grader_type,
            passed_str,
            score_str,
            details,
        )

    console.print(table)


def _save_transcript_json(
    result: Any,
    output_path: str,
    grader_results: list[Any] | None,
) -> None:
    """Save transcript and optional grader results to JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    transcript_data: dict[str, Any] = {
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
    }

    if grader_results is not None:
        transcript_data["grader_results"] = [
            {
                "grader_type": g.grader_type,
                "passed": g.passed,
                "score": g.score,
                "error_message": g.error_message,
                "rationale": getattr(g, "rationale", None),
            }
            for g in grader_results
        ]
        transcript_data["all_passed"] = all(g.passed for g in grader_results)
        transcript_data["aggregate_score"] = (
            sum(g.score for g in grader_results) / len(grader_results) if grader_results else 0.0
        )

    with open(output_file, "w") as f:
        json.dump(transcript_data, f, indent=2, default=str)

    console.print(f"[green]Transcript saved to:[/green] {output_path}")


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
    from ash_hawk.improvement.applier import DiffApplier
    from ash_hawk.improvement.improver_agent import ImprovementContext, ImproverAgent
    from ash_hawk.scenario.loader import load_scenario
    from ash_hawk.scenario.thin_runner import ThinScenarioRunner

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

    for grader in transcript_data.get("grader_results", []):
        if not grader.get("passed", True):
            failed_grades.append(
                {
                    "grader": grader.get("grader_type", "unknown"),
                    "score": grader.get("score", 0.0),
                    "feedback": grader.get("error_message") or grader.get("rationale", "Failed"),
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
    baseline_score = _extract_baseline_score(transcript_data)

    async def _improve() -> None:
        proposals = await improver.analyze_failures(context)

        if not proposals:
            console.print("[yellow]No improvements proposed[/yellow]")
            return

        applier = DiffApplier()
        updated_targets: list[Path] = []
        reverted_targets: list[Path] = []
        apply_failures: list[tuple[Path, str]] = []

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
                updated_targets.append(result.file_path)
                if result.backup_path:
                    console.print(f"[dim]Backup: {result.backup_path}[/dim]")
            else:
                console.print(f"[red]✗ Failed: {result.error}[/red]")
                apply_failures.append((result.file_path, result.error or "unknown error"))
                if result.backup_path and result.backup_path.exists():
                    result.file_path.write_text(
                        result.backup_path.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                    reverted_targets.append(result.file_path)
                    console.print(f"[yellow]↺ Reverted {result.file_path} from backup[/yellow]")

        if not dry_run:
            _print_file_change_summary(updated_targets, reverted_targets, apply_failures)

            score_after = await _reevaluate_score(Path(scenario_path))
            _print_score_delta(baseline_score, score_after)

            if score_after is None:
                console.print(
                    "[yellow]⚠ Could not compute post-apply score (re-evaluation unavailable).[/yellow]"
                )

    async def _reevaluate_score(scenario_file: Path) -> float | None:
        try:
            scenario = load_scenario(scenario_file)
            if not scenario.graders:
                return None

            runner = ThinScenarioRunner(workdir=Path.cwd(), max_iterations=10)
            graded_result = await runner.run_with_grading(scenario, scenario_file)
            return float(graded_result.aggregate_score)
        except Exception as exc:
            console.print(f"[yellow]Re-evaluation failed:[/yellow] {exc}")
            return None

    asyncio.run(_improve())


def _extract_baseline_score(transcript_data: dict[str, Any]) -> float | None:
    score = transcript_data.get("aggregate_score")
    if isinstance(score, float | int):
        return float(score)
    return None


def _print_score_delta(score_before: float | None, score_after: float | None) -> None:
    console.print()
    console.rule("[bold]Score Delta[/bold]")

    before_text = f"{score_before:.3f}" if score_before is not None else "n/a"
    after_text = f"{score_after:.3f}" if score_after is not None else "n/a"
    console.print(f"Before: {before_text}")
    console.print(f"After:  {after_text}")

    if score_before is not None and score_after is not None:
        delta = score_after - score_before
        color = "green" if delta >= 0 else "red"
        console.print(f"[{color}]Delta:  {delta:+.3f}[/{color}]")


def _print_file_change_summary(
    updated_targets: list[Path],
    reverted_targets: list[Path],
    apply_failures: list[tuple[Path, str]],
) -> None:
    console.print()
    console.rule("[bold]Target File Changes[/bold]")

    if updated_targets:
        console.print("[green]Updated targets:[/green]")
        for path in updated_targets:
            console.print(f"  • {path}")
    else:
        console.print("[yellow]Updated targets:[/yellow] none")

    if reverted_targets:
        console.print("[yellow]Reverted targets:[/yellow]")
        for path in reverted_targets:
            console.print(f"  • {path}")
    else:
        console.print("[dim]Reverted targets: none[/dim]")

    if apply_failures:
        console.print("[red]Apply failures:[/red]")
        for path, error in apply_failures:
            console.print(f"  • {path}: {error}")


__all__ = ["thin", "run_thin", "improve_thin"]
