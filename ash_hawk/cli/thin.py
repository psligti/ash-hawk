"""CLI commands for thin telemetry bridge."""  # type-hygiene: skip-file

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from ash_hawk.bridge import DiffFieldChange, DiffReport, RunManifest
from ash_hawk.scenario.thin_runner import ThinScenarioRunner

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
    help="Path to save transcript JSON (legacy; artifacts auto-saved to .ash-hawk/thin/)",
)
@click.option(
    "--grade",
    "-g",
    is_flag=True,
    default=False,
    help="Run graders defined in scenario YAML",
)
@click.option(
    "--variant",
    "-V",
    default="",
    help="Free-form variant tag for provenance (e.g., 'baseline', 'v2-prompt')",
)
@click.option(
    "--storage-root",
    type=click.Path(),
    default=None,
    help="Override storage root (default: .ash-hawk/thin/)",
)
def run_thin(
    scenario_path: str,
    agent_path: str | None,
    max_iterations: int,
    workdir: str,
    output_path: str | None,
    grade: bool,
    variant: str,
    storage_root: str | None,
) -> None:
    """Run a scenario using the thin telemetry bridge.

    This runs the real dawn-kestrel agent with RuntimeHook telemetry
    capture, bypassing the adapter registry. Artifacts are auto-saved
    to .ash-hawk/thin/{scenario_stem}/{run_id}/.
    """
    from ash_hawk.scenario.loader import load_scenario

    scenario_file = Path(scenario_path)
    scenario = load_scenario(scenario_file)
    work_dir = Path(workdir)

    effective_storage = Path(storage_root) if storage_root else None

    console.print(f"[cyan]Loading scenario:[/cyan] {scenario.id}")
    if variant:
        console.print(f"[cyan]Variant:[/cyan] {variant}")

    if grade and scenario.graders:
        console.print(f"[cyan]Graders:[/cyan] {len(scenario.graders)} defined")

    runner = ThinScenarioRunner(
        workdir=work_dir,
        max_iterations=max_iterations,
        variant=variant,
        storage_root=effective_storage,
        agent_override_path=Path(agent_path) if agent_path else None,
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

    if result.manifest:
        console.print(f"[dim]Run ID: {result.manifest.run_id}[/dim]")

    if output_path:
        _save_run_artifacts_legacy(result, output_path, grader_results=None)


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

    if result.manifest:
        console.print(f"[dim]Run ID: {result.manifest.run_id}[/dim]")

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
        _save_run_artifacts_legacy(result, output_path, grader_results=grader_results)


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


def _save_run_artifacts_legacy(
    result: Any,
    output_path: str,
    grader_results: list[Any] | None,
) -> None:
    """Save transcript and optional grader results to a single JSON file.

    This is the legacy path used when ``--output`` is specified.
    The primary path is auto-persist via ThinScenarioRunner._persist_run_artifacts.
    """
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

    if result.manifest:
        transcript_data["manifest"] = result.manifest.model_dump()

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


@thin.command("diff")
@click.argument("baseline_run_id", required=False)
@click.argument("candidate_run_id", required=False)
@click.option(
    "--scenario",
    "scenario_stem",
    default=None,
    help="Scenario stem to search (defaults to auto-discover)",
)
@click.option(
    "--storage-root",
    type=click.Path(exists=True),
    default=None,
    help="Override storage root (default: .ash-hawk/thin/)",
)
def diff_thin(
    baseline_run_id: str | None,
    candidate_run_id: str | None,
    scenario_stem: str | None,
    storage_root: str | None,
) -> None:
    """Compare two thin runs and show provenance/score diff.

    If no run IDs are given, auto-discovers the two most recent runs
    for the scenario. If one run ID is given, compares against the
    most recent other run.
    """
    from ash_hawk.scenario.thin_runner import _DEFAULT_STORAGE_ROOT as default_storage

    effective_root = Path(storage_root) if storage_root else Path.cwd() / default_storage

    runs = ThinScenarioRunner.discover_runs(effective_root, scenario_stem=scenario_stem)

    if len(runs) < 2:
        console.print("[red]Need at least 2 runs to diff. Run 'thin run' first.[/red]")
        raise SystemExit(1)

    if baseline_run_id is None and candidate_run_id is None:
        baseline_entry = runs[-2]
        candidate_entry = runs[-1]
    elif baseline_run_id is not None and candidate_run_id is not None:
        baseline_entry = _find_run(runs, baseline_run_id)
        candidate_entry = _find_run(runs, candidate_run_id)
    elif baseline_run_id is not None:
        candidate_entry = _find_run(runs, baseline_run_id)
        others = [r for r in runs if r["run_id"] != baseline_run_id]
        if not others:
            console.print("[red]No other run found to compare against.[/red]")
            raise SystemExit(1)
        baseline_entry = others[-1]
    else:
        baseline_entry = runs[-2]
        candidate_entry = runs[-1]

    baseline_dir = Path(baseline_entry["path"])
    candidate_dir = Path(candidate_entry["path"])

    baseline_manifest = ThinScenarioRunner.load_manifest(baseline_dir)
    candidate_manifest = ThinScenarioRunner.load_manifest(candidate_dir)

    if baseline_manifest is None or candidate_manifest is None:
        console.print("[red]One or both runs missing manifest.json.[/red]")
        raise SystemExit(1)

    baseline_scores = _load_grades(baseline_dir)
    candidate_scores = _load_grades(candidate_dir)

    report = _build_diff_report(
        baseline_manifest=baseline_manifest,
        candidate_manifest=candidate_manifest,
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
    )

    _display_diff_report(report)


def _find_run(runs: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    for entry in runs:
        if entry["run_id"] == run_id:
            return entry
    console.print(f"[red]Run ID not found: {run_id}[/red]")
    raise SystemExit(1)


def _load_grades(run_dir: Path) -> list[dict[str, Any]]:
    grades_file = run_dir / "grades.json"
    if not grades_file.is_file():
        return []
    try:
        return list(json.loads(grades_file.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return []


def _build_diff_report(
    baseline_manifest: RunManifest,
    candidate_manifest: RunManifest,
    baseline_scores: list[dict[str, Any]],
    candidate_scores: list[dict[str, Any]],
) -> DiffReport:
    field_changes = _compute_field_changes(baseline_manifest, candidate_manifest)
    grader_deltas = _compute_grader_deltas(baseline_scores, candidate_scores)

    baseline_agg = _aggregate_score(baseline_scores)
    candidate_agg = _aggregate_score(candidate_scores)
    score_delta: float | None = None
    if baseline_agg is not None and candidate_agg is not None:
        score_delta = candidate_agg - baseline_agg

    recommendation = _compute_recommendation(score_delta, grader_deltas)

    return DiffReport(
        baseline_run_id=baseline_manifest.run_id,
        candidate_run_id=candidate_manifest.run_id,
        baseline_score=baseline_agg,
        candidate_score=candidate_agg,
        score_delta=score_delta,
        field_changes=field_changes,
        grader_deltas=grader_deltas,
        recommendation=recommendation,
        timestamp=datetime.now(UTC).isoformat(),
    )


def _compute_field_changes(
    baseline: RunManifest,
    candidate: RunManifest,
) -> list[DiffFieldChange]:
    changes: list[DiffFieldChange] = []
    comparable_fields = [
        "scenario_hash",
        "agent_hash",
        "model_name",
        "variant",
        "policy_hash",
    ]
    for field_name in comparable_fields:
        b_val = getattr(baseline, field_name, "")
        c_val = getattr(candidate, field_name, "")
        if b_val != c_val:
            changes.append(
                DiffFieldChange(
                    field=field_name,
                    baseline=str(b_val),
                    candidate=str(c_val),
                )
            )

    if baseline.skill_hashes != candidate.skill_hashes:
        for skill, b_hash in baseline.skill_hashes.items():
            c_hash = candidate.skill_hashes.get(skill, "")
            if b_hash != c_hash:
                changes.append(
                    DiffFieldChange(
                        field=f"skill_hashes.{skill}",
                        baseline=b_hash[:12],
                        candidate=c_hash[:12] if c_hash else "(missing)",
                    )
                )
        for skill in candidate.skill_hashes:
            if skill not in baseline.skill_hashes:
                changes.append(
                    DiffFieldChange(
                        field=f"skill_hashes.{skill}",
                        baseline="(missing)",
                        candidate=candidate.skill_hashes[skill][:12],
                    )
                )

    if baseline.tool_hashes != candidate.tool_hashes:
        for tool, b_hash in baseline.tool_hashes.items():
            c_hash = candidate.tool_hashes.get(tool, "")
            if b_hash != c_hash:
                changes.append(
                    DiffFieldChange(
                        field=f"tool_hashes.{tool}",
                        baseline=b_hash[:12],
                        candidate=c_hash[:12] if c_hash else "(missing)",
                    )
                )
        for tool in candidate.tool_hashes:
            if tool not in baseline.tool_hashes:
                changes.append(
                    DiffFieldChange(
                        field=f"tool_hashes.{tool}",
                        baseline="(missing)",
                        candidate=candidate.tool_hashes[tool][:12],
                    )
                )

    return changes


def _compute_grader_deltas(
    baseline_scores: list[dict[str, Any]],
    candidate_scores: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    deltas: dict[str, dict[str, Any]] = {}

    baseline_by_type = {s["grader_type"]: s for s in baseline_scores}
    candidate_by_type = {s["grader_type"]: s for s in candidate_scores}

    all_types = set(baseline_by_type) | set(candidate_by_type)
    for g_type in sorted(all_types):
        b = baseline_by_type.get(g_type, {})
        c = candidate_by_type.get(g_type, {})
        b_score = b.get("score", 0.0)
        c_score = c.get("score", 0.0)
        b_passed = b.get("passed", False)
        c_passed = c.get("passed", False)
        deltas[g_type] = {
            "baseline_score": b_score,
            "candidate_score": c_score,
            "delta": c_score - b_score,
            "flipped": b_passed != c_passed,
        }
    return deltas


def _aggregate_score(scores: list[dict[str, Any]]) -> float | None:
    if not scores:
        return None
    total: float = sum(float(s.get("score", 0.0)) for s in scores)
    return total / len(scores)


def _compute_recommendation(
    score_delta: float | None,
    grader_deltas: dict[str, dict[str, Any]],
) -> str:
    if score_delta is None:
        return "inconclusive"

    has_regression = any(d["delta"] < 0 for d in grader_deltas.values())
    has_improvement = any(d["delta"] > 0 for d in grader_deltas.values())
    has_flip_to_fail = any(
        d.get("flipped") and d["candidate_score"] < d["baseline_score"]
        for d in grader_deltas.values()
    )

    if has_flip_to_fail:
        return "reject"
    if has_regression and not has_improvement:
        return "reject"
    if has_improvement and not has_regression:
        return "keep"
    if has_improvement and has_regression:
        return "inconclusive"
    if score_delta > 0:
        return "keep"
    if score_delta < 0:
        return "reject"
    return "inconclusive"


def _display_diff_report(report: DiffReport) -> None:
    console.print()
    console.rule("[bold]Run Comparison[/bold]")

    console.print(f"[dim]Baseline:  {report.baseline_run_id}[/dim]")
    console.print(f"[dim]Candidate: {report.candidate_run_id}[/dim]")

    if report.score_delta is not None:
        color = "green" if report.score_delta >= 0 else "red"
        console.print(
            f"[{color}]Score delta: {report.score_delta:+.3f}"
            f" ({report.baseline_score:.3f} → {report.candidate_score:.3f})[/{color}]"
        )
    else:
        console.print("[yellow]Score delta: n/a (no grades found)[/yellow]")

    if report.field_changes:
        console.print()
        console.rule("[bold]Manifest Changes[/bold]")
        table = Table()
        table.add_column("Field", style="cyan")
        table.add_column("Baseline", style="dim")
        table.add_column("Candidate", style="dim")
        for change in report.field_changes:
            table.add_row(change.field, change.baseline, change.candidate)
        console.print(table)

    if report.grader_deltas:
        console.print()
        console.rule("[bold]Grader Deltas[/bold]")
        table = Table()
        table.add_column("Grader", style="cyan")
        table.add_column("Baseline", justify="right")
        table.add_column("Candidate", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Flipped", justify="center")
        for g_type, delta in report.grader_deltas.items():
            d_color = "green" if delta["delta"] >= 0 else "red"
            flipped_str = "[yellow]⚡[/yellow]" if delta["flipped"] else ""
            table.add_row(
                g_type,
                f"{delta['baseline_score']:.2f}",
                f"{delta['candidate_score']:.2f}",
                f"[{d_color}]{delta['delta']:+.2f}[/{d_color}]",
                flipped_str,
            )
        console.print(table)

    console.print()
    rec_color = {"keep": "green", "reject": "red", "inconclusive": "yellow"}
    console.print(
        f"[bold {rec_color.get(report.recommendation, 'white')}]"
        f"Recommendation: {report.recommendation.upper()}[/bold {rec_color.get(report.recommendation, 'white')}]"
    )


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
def improve_thin(
    scenario_path: str,
    transcript_path: str,
    agent_path: str,
    dry_run: bool,
) -> None:
    """Analyze a failed transcript and propose improvements."""
    console.print(
        "[yellow]The 'improve' command has been replaced by the auto-research loop.[/yellow]\n"
        "[dim]Use: ash-hawk auto-research run -s <scenario.yaml>[/dim]"
    )


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


__all__ = ["thin", "run_thin", "improve_thin", "diff_thin"]
