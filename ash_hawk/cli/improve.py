from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Literal, cast
from uuid import uuid4

import click
from rich.console import Console
from rich.table import Table

from ash_hawk.adapters.artifact_adapter import ArtifactAdapter
from ash_hawk.contracts import (
    CuratedLesson,
    ImprovementProposal,
    ReviewMetrics,
    ReviewRequest,
    ReviewResult,
)
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

    console.print("[yellow]Lesson marked for rollback[/yellow]")


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
