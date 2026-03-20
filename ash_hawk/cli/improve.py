from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

import click
from rich.console import Console
from rich.table import Table

from ash_hawk.adapters.artifact_adapter import ArtifactAdapter
from ash_hawk.contracts import (
    ImprovementProposal,
    ReviewMetrics,
    ReviewRequest,
    ReviewResult,
)
from ash_hawk.cycle.types import IterationResult
from ash_hawk.scenario.loader import discover_scenarios
from ash_hawk.services.review_service import ReviewService
from ash_hawk.storage import FileStorage

console = Console()


@click.group(name="improve")
def improve() -> None:
    """Cross-agent improvement pipeline commands.

    Run evaluations, generate improvement proposals, and curate lessons
    for Dawn Kestrel-based agents (iron-rook, bolt-merlin, vox-jay).
    """
    pass


@improve.command(name="review")
@click.argument("run_artifact_id")
@click.option(
    "--agent",
    "-a",
    required=True,
    help="Target agent to evaluate (e.g., iron-rook, bolt-merlin, vox-jay)",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["quick", "standard", "deep"]),
    default="standard",
    help="Review depth mode",
)
@click.option(
    "--persist",
    "-p",
    type=click.Choice(["none", "propose", "curate"]),
    default="propose",
    help="Persistence mode for findings",
)
@click.option(
    "--baseline",
    "-b",
    default=None,
    help="Baseline run ID for comparison",
)
@click.option(
    "--eval-suite",
    "-e",
    multiple=True,
    help="Specific evaluators to run (can specify multiple)",
)
def review_run(
    run_artifact_id: str,
    agent: str,
    mode: str,
    persist: str,
    baseline: str | None,
    eval_suite: tuple[str, ...],
) -> None:
    """Review a completed agent run and generate improvement proposals.

    RUN_ARTIFACT_ID: ID of the run artifact to review.
    """
    console.print(f"[cyan]Reviewing run:[/cyan] {run_artifact_id}")
    console.print(f"[cyan]Target agent:[/cyan] {agent}")
    console.print(f"[cyan]Review mode:[/cyan] {mode}")

    request = ReviewRequest(
        run_artifact_id=run_artifact_id,
        target_agent=agent,
        eval_suite=list(eval_suite) if eval_suite else [],
        review_mode=cast(Literal["quick", "standard", "deep"], mode),
        persistence_mode=cast(Literal["none", "propose", "curate"], persist),
        baseline_run_id=baseline,
    )

    result = _run_review(request)

    _display_review_result(result)


@improve.command(name="propose")
@click.argument("review_id")
@click.option(
    "--type",
    "-t",
    type=click.Choice(["policy", "skill", "tool", "harness", "eval"]),
    required=True,
    help="Type of improvement proposal",
)
@click.option(
    "--title",
    "-T",
    required=True,
    help="Title for the proposal",
)
@click.option(
    "--rationale",
    "-r",
    required=True,
    help="Rationale for the proposed change",
)
@click.option(
    "--risk",
    type=click.Choice(["low", "medium", "high"]),
    default="medium",
    help="Risk level of the change",
)
def create_proposal(
    review_id: str,
    type: str,
    title: str,
    rationale: str,
    risk: str,
) -> None:
    """Create an improvement proposal from a review.

    REVIEW_ID: ID of the review to create proposal from.
    """
    console.print(f"[cyan]Creating proposal from review:[/cyan] {review_id}")

    proposal = ImprovementProposal(
        proposal_id=f"prop-{uuid4().hex[:8]}",
        origin_run_id="",
        origin_review_id=review_id,
        target_agent="",
        proposal_type=cast(Literal["policy", "skill", "tool", "harness", "eval"], type),
        title=title,
        rationale=rationale,
        expected_benefit="",
        risk_level=cast(Literal["low", "medium", "high"], risk),
        created_at=datetime.now(UTC),
    )

    console.print(f"[green]Created proposal:[/green] {proposal.proposal_id}")
    _display_proposal(proposal)


@improve.command(name="curate")
@click.argument("proposal_id")
@click.option(
    "--action",
    "-a",
    type=click.Choice(["approve", "reject", "defer"]),
    required=True,
    help="Curation action to take",
)
@click.option(
    "--reason",
    "-r",
    default=None,
    help="Reason for the curation decision",
)
@click.option(
    "--applies-to",
    multiple=True,
    help="Agents this lesson applies to (for approve)",
)
def curate_proposal(
    proposal_id: str,
    action: str,
    reason: str | None,
    applies_to: tuple[str, ...],
) -> None:
    """Curate an improvement proposal (approve, reject, or defer).

    PROPOSAL_ID: ID of the proposal to curate.
    """
    from ash_hawk.contracts import ImprovementProposal
    from ash_hawk.services.lesson_service import LessonService

    console.print(f"[cyan]Curating proposal:[/cyan] {proposal_id}")
    console.print(f"[cyan]Action:[/cyan] {action}")

    if action == "approve":
        proposal = ImprovementProposal(
            proposal_id=proposal_id,
            origin_run_id="",
            origin_review_id="",
            target_agent="",
            proposal_type="skill",
            title=reason or "Approved proposal",
            rationale=reason or "",
            expected_benefit="",
            risk_level="medium",
            created_at=datetime.now(UTC),
        )
        service = LessonService()
        lesson = service.approve_proposal(
            proposal,
            applies_to_agents=list(applies_to) if applies_to else None,
        )
        console.print(f"[green]Created lesson:[/green] {lesson.lesson_id}")
        console.print(f"[green]Applies to:[/green] {', '.join(lesson.applies_to_agents)}")
    elif action == "reject":
        console.print(f"[red]Rejected proposal:[/red] {reason or 'No reason provided'}")
    else:
        console.print("[yellow]Deferred proposal[/yellow]")


@improve.command(name="list")
@click.option(
    "--agent",
    "-a",
    default=None,
    help="Filter by target agent",
)
@click.option(
    "--status",
    "-s",
    type=click.Choice(["pending", "approved", "rejected", "implemented"]),
    default=None,
    help="Filter by status",
)
@click.option(
    "--type",
    "-t",
    type=click.Choice(["proposal", "lesson"]),
    default="proposal",
    help="List proposals or lessons",
)
@click.option(
    "--strategy",
    "-S",
    default=None,
    help="Filter by strategy (e.g., policy-quality, tool-quality)",
)
@click.option(
    "--sub-strategy",
    "-ss",
    default=None,
    help="Filter by sub-strategy (e.g., tool-efficiency, error-recovery)",
)
def list_improvements(
    agent: str | None,
    status: str | None,
    type: str,
    strategy: str | None,
    sub_strategy: str | None,
) -> None:
    """List improvement proposals or curated lessons."""
    if type == "proposal":
        _list_proposals(agent, status, strategy, sub_strategy)
    else:
        _list_lessons(agent, status, strategy, sub_strategy)


@improve.command(name="rollback")
@click.argument("lesson_id")
@click.option(
    "--reason",
    "-r",
    required=True,
    help="Reason for rollback",
)
def rollback_lesson(lesson_id: str, reason: str) -> None:
    """Roll back a curated lesson.

    LESSON_ID: ID of the lesson to roll back.
    """
    console.print(f"[red]Rolling back lesson:[/red] {lesson_id}")
    console.print(f"[red]Reason:[/red] {reason}")
    from ash_hawk.services.lesson_service import LessonService

    service = LessonService()
    updated = service.deactivate_lesson(lesson_id)
    if updated is None:
        raise click.ClickException(f"Lesson not found: {lesson_id}")

    console.print(f"[green]Lesson rolled back:[/green] {updated.lesson_id}")


def _run_review(request: ReviewRequest) -> ReviewResult:
    storage_path = os.environ.get("ASH_HAWK_STORAGE_PATH", ".ash-hawk")
    storage = FileStorage(storage_path)
    adapter = ArtifactAdapter(storage)
    service = ReviewService()

    async def load_and_review() -> ReviewResult:
        artifact = await adapter.load_run_artifact(request.run_artifact_id)
        if artifact is None:
            return ReviewResult(
                review_id=f"review-{uuid4().hex[:8]}",
                request_id=request.run_artifact_id,
                run_artifact_id=request.run_artifact_id,
                target_agent=request.target_agent,
                status="failed",
                findings=[],
                metrics=ReviewMetrics(score=0.0),
                created_at=datetime.now(UTC),
                error_message=f"Run artifact not found: {request.run_artifact_id}",
            )
        return service.review(request, artifact)

    return asyncio.run(load_and_review())


def _display_review_result(result: ReviewResult) -> None:
    """Display review result in a formatted table."""
    console.print()
    console.print(f"[bold]Review:[/bold] {result.review_id}")
    console.print(f"[bold]Status:[/bold] {result.status}")
    console.print(f"[bold]Score:[/bold] {result.metrics.score:.2f}")

    if result.findings:
        table = Table(title="Findings")
        table.add_column("ID", style="cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Severity", style="red")
        table.add_column("Title")

        for finding in result.findings:
            severity_color = {
                "critical": "red",
                "warning": "yellow",
                "info": "blue",
            }.get(finding.severity, "white")
            table.add_row(
                finding.finding_id,
                finding.category,
                f"[{severity_color}]{finding.severity}[/{severity_color}]",
                finding.title,
            )

        console.print(table)


def _display_proposal(proposal: ImprovementProposal) -> None:
    """Display proposal details."""
    console.print()
    console.print(f"[bold]Proposal:[/bold] {proposal.proposal_id}")
    console.print(f"[bold]Type:[/bold] {proposal.proposal_type}")
    console.print(f"[bold]Title:[/bold] {proposal.title}")
    console.print(f"[bold]Risk:[/bold] {proposal.risk_level}")
    console.print(f"[bold]Rationale:[/bold] {proposal.rationale}")


def _list_proposals(
    agent: str | None,
    status: str | None,
    strategy: str | None = None,
    sub_strategy: str | None = None,
) -> None:
    console.print(
        "[yellow]Proposals are stored in-memory during review sessions.[/yellow]\n"
        "Use 'ash-hawk improve review <run-id>' to create proposals.\n"
        "Proposals are persisted as CuratedLessons after curation."
    )

    table = Table(title="Recent Proposals (from storage)")
    table.add_column("ID", style="cyan")
    table.add_column("Agent", style="magenta")
    table.add_column("Type", style="blue")
    table.add_column("Strategy", style="green")
    table.add_column("Status", style="green")
    table.add_column("Title")
    table.add_column("Created")

    table.add_row(
        "(none)",
        agent or "-",
        "-",
        strategy or "-",
        status or "-",
        "No pending proposals in persistent storage",
        "-",
    )

    console.print(table)


def _list_lessons(
    agent: str | None,
    status: str | None,
    strategy: str | None = None,
    sub_strategy: str | None = None,
) -> None:
    from ash_hawk.services.lesson_service import LessonService

    table = Table(title="Curated Lessons")
    table.add_column("ID", style="cyan")
    table.add_column("Agents", style="magenta")
    table.add_column("Type", style="blue")
    table.add_column("Strategy", style="green")
    table.add_column("Version", style="green")
    table.add_column("Title")
    table.add_column("Status")

    service = LessonService()
    status_filter = "approved" if status is None else status
    lessons = service.list_lessons(status=status_filter)

    if agent:
        lessons = [lesson for lesson in lessons if agent in lesson.applies_to_agents]
    if strategy:
        lessons = [lesson for lesson in lessons if getattr(lesson, "strategy", None) == strategy]
    if sub_strategy:
        lessons = [
            lesson for lesson in lessons if getattr(lesson, "sub_strategy", None) == sub_strategy
        ]

    if not lessons:
        console.print("[yellow]No lessons found[/yellow]")
        return

    for lesson in lessons[:50]:
        table.add_row(
            lesson.lesson_id,
            ", ".join(lesson.applies_to_agents),
            lesson.lesson_type,
            getattr(lesson, "strategy", "-") or "-",
            str(lesson.version),
            lesson.title[:40] if len(lesson.title) > 40 else lesson.title,
            lesson.validation_status,
        )

    console.print(table)


@improve.command(name="cycle")
@click.option(
    "--agent",
    "-a",
    required=True,
    help="Target agent to improve (e.g., bolt-merlin)",
)
@click.option(
    "--iterations",
    "-i",
    default=100,
    type=int,
    help="Maximum number of iterations (default: 100)",
)
@click.option(
    "--experiment",
    "-e",
    default=None,
    help="Experiment ID for lesson isolation (auto-generated if not provided)",
)
@click.option(
    "--convergence-threshold",
    "-c",
    default=0.01,
    type=float,
    help="Variance threshold for convergence detection",
)
@click.option(
    "--stop-on-convergence/--run-all-iterations",
    default=False,
    help="Stop early on convergence (default runs full iteration count)",
)
@click.option(
    "--promotion-threshold",
    "-p",
    default=3,
    type=int,
    help="Consecutive improvements needed to promote lessons",
)
@click.option(
    "--eval-pack",
    default=None,
    help="Specific eval pack to use (default: auto-detect by agent)",
)
@click.option(
    "--checkpoint-interval",
    default=10,
    type=int,
    help="Save checkpoint every N iterations",
)
@click.option(
    "--lessons-per-iteration",
    default=4,
    type=int,
    help="Maximum approved lessons to apply per iteration",
)
@click.option(
    "--scenario-path",
    multiple=True,
    help="Scenario file path(s) to run for per-iteration scoring",
)
@click.option(
    "--scenario-parallelism",
    type=int,
    default=None,
    help="Override parallel scenario workers for per-iteration scoring",
)
@click.option(
    "--queue-timeout-seconds",
    type=int,
    default=None,
    help="Override ASH_HAWK_DEFAULT_TIMEOUT_SECONDS for this run",
)
@click.option(
    "--llm-timeout-seconds",
    type=int,
    default=None,
    help="Override ASH_HAWK_LLM_TIMEOUT_SECONDS for this run",
)
@click.option(
    "--llm-max-workers",
    type=int,
    default=None,
    help="Override ASH_HAWK_LLM_MAX_WORKERS for this run",
)
@click.option(
    "--trial-max-workers",
    type=int,
    default=None,
    help="Override ASH_HAWK_TRIAL_MAX_WORKERS for this run",
)
@click.option(
    "--progress-logs/--no-progress-logs",
    default=True,
    help="Show iteration progress logs during cycle execution",
)
@click.option(
    "--disable-redis/--allow-redis",
    default=False,
    help="Use Redis queue/rate limiting by default; pass --disable-redis to force local backends",
)
@click.option("--enable-competitor/--disable-competitor", default=True)
@click.option("--enable-triage/--disable-triage", default=True)
@click.option("--enable-verifier/--disable-verifier", default=True)
@click.option("--enable-adversary/--disable-adversary", default=True)
@click.option("--cross-pack-eval-pack", multiple=True)
@click.option("--max-token-delta-pct", type=float, default=10.0)
@click.option("--max-latency-delta-pct", type=float, default=15.0)
@click.option("--min-verification-runs", type=int, default=3)
@click.option(
    "--promotion-scope",
    type=click.Choice(
        ["global", "agent-specific", "eval-pack-specific", "scenario-family-specific", "temporary"]
    ),
    default="agent-specific",
)
def run_cycle(
    agent: str,
    iterations: int,
    experiment: str | None,
    convergence_threshold: float,
    stop_on_convergence: bool,
    promotion_threshold: int,
    eval_pack: str | None,
    checkpoint_interval: int,
    lessons_per_iteration: int,
    scenario_path: tuple[str, ...],
    scenario_parallelism: int | None,
    queue_timeout_seconds: int | None,
    llm_timeout_seconds: int | None,
    llm_max_workers: int | None,
    trial_max_workers: int | None,
    progress_logs: bool,
    disable_redis: bool,
    enable_competitor: bool,
    enable_triage: bool,
    enable_verifier: bool,
    enable_adversary: bool,
    cross_pack_eval_pack: tuple[str, ...],
    max_token_delta_pct: float,
    max_latency_delta_pct: float,
    min_verification_runs: int,
    promotion_scope: str,
) -> None:
    """Run an N-iteration improvement cycle on a target agent.

    Each iteration runs evaluation, generates lessons, and applies them
    to subsequent iterations. The cycle stops when converged or max
    iterations reached.

    \b
    Example:
        ash-hawk improve cycle --agent bolt-merlin --iterations 100
    """
    import os

    from ash_hawk.config import reload_config
    from ash_hawk.cycle import CycleConfig, CycleRunner, create_cycle_id

    effective_queue_timeout_seconds = queue_timeout_seconds
    effective_scenario_parallelism = scenario_parallelism
    effective_llm_timeout_seconds = llm_timeout_seconds
    effective_llm_max_workers = llm_max_workers
    effective_trial_max_workers = trial_max_workers

    cycle_id = create_cycle_id()
    experiment_id = experiment or f"exp-{agent}-{cycle_id}"
    scenario_inputs = list(scenario_path) or _default_scenarios_for_agent(agent)
    scenario_paths = resolve_cycle_scenario_paths(scenario_inputs)

    if effective_scenario_parallelism is None and scenario_paths:
        if len(scenario_paths) >= 100:
            effective_scenario_parallelism = 1
        elif len(scenario_paths) >= 40:
            effective_scenario_parallelism = 2

    if effective_queue_timeout_seconds is None and scenario_paths:
        if len(scenario_paths) >= 100:
            effective_queue_timeout_seconds = 14400
        elif len(scenario_paths) >= 40:
            effective_queue_timeout_seconds = 5400

    if effective_llm_timeout_seconds is None and scenario_paths:
        if len(scenario_paths) >= 100:
            effective_llm_timeout_seconds = 420
        elif len(scenario_paths) >= 40:
            effective_llm_timeout_seconds = 240

    if scenario_paths and llm_max_workers is None and trial_max_workers is None:
        if len(scenario_paths) >= 100:
            effective_llm_max_workers = 1
            effective_trial_max_workers = 1
        elif len(scenario_paths) >= 40:
            effective_llm_max_workers = 2
            effective_trial_max_workers = 2

    if effective_queue_timeout_seconds is not None:
        os.environ["ASH_HAWK_DEFAULT_TIMEOUT_SECONDS"] = str(effective_queue_timeout_seconds)
    if effective_llm_timeout_seconds is not None:
        os.environ["ASH_HAWK_LLM_TIMEOUT_SECONDS"] = str(effective_llm_timeout_seconds)
    if effective_llm_max_workers is not None:
        os.environ["ASH_HAWK_LLM_MAX_WORKERS"] = str(effective_llm_max_workers)
    if effective_trial_max_workers is not None:
        os.environ["ASH_HAWK_TRIAL_MAX_WORKERS"] = str(effective_trial_max_workers)
    if disable_redis:
        os.environ["DAWN_KESTREL_REDIS_URL"] = ""
        os.environ["DAWN_KESTREL_QUEUE_BACKEND"] = "local"
        os.environ["DAWN_KESTREL_RATE_LIMIT_BACKEND"] = "local"

    if progress_logs:
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )
        else:
            root_logger.setLevel(logging.INFO)
            for handler in root_logger.handlers:
                handler.setLevel(logging.INFO)
        logging.getLogger("ash_hawk").setLevel(logging.INFO)
        logging.getLogger("dawn_kestrel").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)

    reload_config()

    console.print(f"[cyan]Starting improvement cycle:[/cyan] {cycle_id}")
    console.print(f"[cyan]Target agent:[/cyan] {agent}")
    console.print(f"[cyan]Experiment ID:[/cyan] {experiment_id}")
    console.print(f"[cyan]Max iterations:[/cyan] {iterations}")
    console.print(
        "[cyan]Convergence behavior:[/cyan]",
        "stop early" if stop_on_convergence else "run full iterations",
    )
    if scenario_paths:
        console.print(f"[cyan]Scenario paths:[/cyan] {len(scenario_paths)} resolved")
    if effective_scenario_parallelism is not None:
        console.print(f"[cyan]Scenario parallelism:[/cyan] {effective_scenario_parallelism}")
    if effective_queue_timeout_seconds is not None:
        console.print(f"[cyan]Queue timeout:[/cyan] {effective_queue_timeout_seconds}s")
    if effective_llm_timeout_seconds is not None:
        console.print(f"[cyan]LLM timeout:[/cyan] {effective_llm_timeout_seconds}s")
    if effective_llm_max_workers is not None:
        console.print(f"[cyan]LLM workers:[/cyan] {effective_llm_max_workers}")
    if effective_trial_max_workers is not None:
        console.print(f"[cyan]Trial workers:[/cyan] {effective_trial_max_workers}")
    if disable_redis:
        console.print("[cyan]Redis mode:[/cyan] disabled (local backends)")
    console.print(f"[cyan]Progress logs:[/cyan] {'enabled' if progress_logs else 'disabled'}")

    config = CycleConfig(
        cycle_id=cycle_id,
        experiment_id=experiment_id,
        target_agent=agent,
        max_iterations=iterations,
        convergence_threshold=convergence_threshold,
        stop_on_convergence=stop_on_convergence,
        promotion_success_threshold=promotion_threshold,
        eval_pack=eval_pack,
        checkpoint_interval=checkpoint_interval,
        max_lessons_per_iteration=lessons_per_iteration,
        scenario_paths=scenario_paths,
        scenario_parallelism=effective_scenario_parallelism,
        metadata={
            "enable_competitor": enable_competitor,
            "enable_triage": enable_triage,
            "enable_verifier": enable_verifier,
            "enable_adversary": enable_adversary,
            "cross_pack_eval_pack": list(cross_pack_eval_pack),
            "max_token_delta_pct": max_token_delta_pct,
            "max_latency_delta_pct": max_latency_delta_pct,
            "min_verification_runs": min_verification_runs,
            "promotion_scope": promotion_scope,
        },
    )

    runner = CycleRunner(config)

    async def _run() -> None:
        result = await runner.run_cycle()

        console.print()
        console.print("[bold green]Cycle Complete:[/bold green]", result.status.value)
        console.print("[bold]Total iterations:[/bold]", result.total_iterations)
        console.print("[bold]Best score:[/bold]", f"{result.best_score:.3f}")
        console.print("[bold]Final score:[/bold]", f"{result.final_score:.3f}")
        console.print("[bold]Improvement:[/bold]", f"{result.improvement_delta:+.3f}")
        console.print("[bold]Lessons generated:[/bold]", result.total_lessons_generated)
        _display_generated_lessons(result.iterations)
        console.print("[bold]Lessons promoted:[/bold]", len(result.lessons_promoted))
        console.print("[bold]Convergence:[/bold]", result.convergence_status.value)

        if result.iterations:
            _display_iteration_summary(result.iterations, result.total_iterations)
            _display_changes_under_test(result.iterations)
            _display_experiment_cards(result.iterations)
            _display_llm_agent_summaries(result.iterations)
            _display_change_set_outcomes(result.iterations)
            _display_best_experiment(result.iterations)

    asyncio.run(_run())


def _display_iteration_summary(
    iterations: list[IterationResult],
    total: int,
) -> None:
    table = Table(title=f"Iteration Summary (showing {len(iterations)}/{total})")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Score", style="green")
    table.add_column("Delta", style="yellow")
    table.add_column("Lessons", style="magenta")
    table.add_column("Status", style="blue")

    for it in iterations:
        delta_str = f"{it.score_delta:+.3f}" if it.score_delta is not None else "-"
        table.add_row(
            str(it.iteration_num),
            f"{it.score:.3f}",
            delta_str,
            str(it.lessons_generated),
            it.status.value,
        )

    console.print(table)


def _display_generated_lessons(iterations: list[IterationResult]) -> None:
    lesson_entries: list[str] = []
    for iteration in iterations:
        raw_titles = iteration.metadata.get("lesson_titles", [])
        titles = [t for t in raw_titles if isinstance(t, str) and t.strip()]
        for title in titles:
            lesson_entries.append(f"{iteration.iteration_num}. {title}")

    if not lesson_entries:
        return

    console.print("[bold]Lesson names:[/bold]")
    for entry in lesson_entries:
        console.print(f"  - {entry}")


def _display_changes_under_test(iterations: list[IterationResult]) -> None:
    console.print("[bold]Changes tested:[/bold]")
    for iteration in iterations:
        delta_str = f"{iteration.score_delta:+.3f}" if iteration.score_delta is not None else "-"
        raw_titles = iteration.metadata.get("tested_change_titles", [])
        raw_ids = iteration.metadata.get("tested_change_ids", [])
        titles = [t for t in raw_titles if isinstance(t, str) and t.strip()]
        lesson_ids = [
            lesson_id for lesson_id in raw_ids if isinstance(lesson_id, str) and lesson_id
        ]

        if not titles:
            console.print(
                f"  - {iteration.iteration_num}. Baseline (no prior lesson applied), "
                f"score={iteration.score:.3f}, delta={delta_str}"
            )
            continue

        console.print(
            f"  - {iteration.iteration_num}. score={iteration.score:.3f}, delta={delta_str}"
        )
        for idx, title in enumerate(titles):
            lesson_id = lesson_ids[idx] if idx < len(lesson_ids) else "unknown_lesson_id"
            console.print(f"    - ({lesson_id}) {title}")


def _display_experiment_cards(iterations: list[IterationResult]) -> None:
    console.print("[bold]Experiments:[/bold]")
    for iteration in iterations:
        delta_str = f"{iteration.score_delta:+.3f}" if iteration.score_delta is not None else "-"
        tested_titles_raw = iteration.metadata.get("tested_change_titles", [])
        tested_ids_raw = iteration.metadata.get("tested_change_ids", [])
        tested_rationales_raw = iteration.metadata.get("tested_change_rationales", [])

        tested_titles = [t for t in tested_titles_raw if isinstance(t, str) and t.strip()]
        tested_ids = [t for t in tested_ids_raw if isinstance(t, str) and t.strip()]
        tested_rationales = [r for r in tested_rationales_raw if isinstance(r, str) and r.strip()]

        console.print(
            f"  - Iteration {iteration.iteration_num}: score={iteration.score:.3f}, "
            f"delta={delta_str}, status={iteration.status.value}"
        )

        if not tested_titles:
            console.print("    - Applied changes: baseline (no prior lesson applied)")
            continue

        console.print("    - Applied changes:")
        for idx, title in enumerate(tested_titles):
            lesson_id = tested_ids[idx] if idx < len(tested_ids) else "(unknown lesson id)"
            console.print(f"      - ({lesson_id}) {title}")
            if idx < len(tested_rationales):
                console.print(f"        rationale: {tested_rationales[idx]}")


def _display_best_experiment(iterations: list[IterationResult]) -> None:
    if not iterations:
        return

    best = max(iterations, key=lambda it: it.score)
    delta_str = f"{best.score_delta:+.3f}" if best.score_delta is not None else "-"
    tested_titles_raw = best.metadata.get("tested_change_titles", [])
    tested_ids_raw = best.metadata.get("tested_change_ids", [])
    tested_rationales_raw = best.metadata.get("tested_change_rationales", [])

    tested_titles = [t for t in tested_titles_raw if isinstance(t, str) and t.strip()]
    tested_ids = [t for t in tested_ids_raw if isinstance(t, str) and t.strip()]
    tested_rationales = [r for r in tested_rationales_raw if isinstance(r, str) and r.strip()]

    console.print("[bold]Best experiment:[/bold]")
    console.print(
        f"  - Iteration {best.iteration_num} with score={best.score:.3f}, "
        f"delta={delta_str}, status={best.status.value}"
    )

    if not tested_titles:
        console.print("  - Applied changes: baseline (no prior lesson applied)")
        return

    console.print("  - Applied changes:")
    for idx, title in enumerate(tested_titles):
        lesson_id = tested_ids[idx] if idx < len(tested_ids) else "(unknown lesson id)"
        console.print(f"    - ({lesson_id}) {title}")
        if idx < len(tested_rationales):
            console.print(f"      rationale: {tested_rationales[idx]}")


def _display_change_set_outcomes(iterations: list[IterationResult]) -> None:
    if not iterations:
        return

    grouped: dict[tuple[str, ...], list[IterationResult]] = {}
    for iteration in iterations:
        raw_ids = iteration.metadata.get("tested_change_ids", [])
        ids = tuple(sorted([item for item in raw_ids if isinstance(item, str) and item]))
        grouped.setdefault(ids, []).append(iteration)

    console.print("[bold]Change-set outcomes:[/bold]")
    for ids, runs in sorted(
        grouped.items(), key=lambda item: max(r.score for r in item[1]), reverse=True
    ):
        scores = [run.score for run in runs]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        id_label = ", ".join(ids) if ids else "baseline"
        console.print(
            f"  - {id_label}: runs={len(runs)}, avg={avg_score:.3f}, "
            f"best={max_score:.3f}, worst={min_score:.3f}"
        )


def _coerce_str_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    raw = cast(dict[object, object], value)
    out: dict[str, str] = {}
    for key, item in raw.items():
        if isinstance(key, str) and isinstance(item, str):
            out[key] = item
    return out


def _display_llm_agent_summaries(iterations: list[IterationResult]) -> None:
    console.print("[bold]LLM agent summaries:[/bold]")
    for iteration in iterations:
        role_summaries = _coerce_str_map(iteration.metadata.get("role_summaries", {}))
        console.print(f"  - Iteration {iteration.iteration_num}:")
        if not role_summaries:
            console.print("    - no role summaries available")
            continue

        ordered_roles = ["competitor", "translator", "analyst", "coach", "architect", "curator"]
        for role in ordered_roles:
            summary = role_summaries.get(role)
            if isinstance(summary, str) and summary.strip():
                console.print(f"    - {role}: {summary}")


def _default_scenarios_for_agent(agent: str) -> list[str]:
    project_root = Path(__file__).resolve().parents[2]

    if agent == "bolt-merlin":
        candidate = project_root / "examples" / "scenarios" / "bolt-merlin"
        if candidate.exists():
            return [str(candidate)]

    return []


def resolve_cycle_scenario_paths(paths: list[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()

    for path in paths:
        candidate = Path(path)
        discovered: list[Path]
        if candidate.is_dir():
            discovered = discover_scenarios(candidate)
        elif candidate.is_file():
            discovered = [candidate.resolve()]
        else:
            raise click.ClickException(f"Scenario path not found: {path}")

        for scenario_path in discovered:
            normalized = str(scenario_path.resolve())
            if normalized in seen:
                continue
            seen.add(normalized)
            resolved.append(normalized)

    if not resolved:
        raise click.ClickException("No scenario files found in provided scenario paths")

    return resolved
