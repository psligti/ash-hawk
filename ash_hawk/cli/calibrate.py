"""Calibration CLI command for judge calibration against ground truth."""

import asyncio
import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ash_hawk.config import get_config
from ash_hawk.storage import FileStorage
from ash_hawk.types import CalibrationCurve, CalibrationSample

console = Console()


def _compute_recommended_threshold(samples: list[CalibrationSample]) -> tuple[float, str]:
    if not samples:
        return 0.5, "Default threshold (no samples)"

    predictions = [s.predicted for s in samples]
    if len(set(predictions)) == 1:
        return 0.5, "Default threshold (uniform predictions)"

    def calc_accuracy(threshold: float) -> float:
        correct = sum(1 for s in samples if (s.predicted >= threshold) == s.actual)
        return correct / len(samples)

    best_threshold = 0.5
    best_accuracy = calc_accuracy(0.5)
    default_accuracy = best_accuracy

    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        acc = calc_accuracy(t)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = t

    return best_threshold, (
        f"Maximizes accuracy at {best_accuracy * 100:.1f}% "
        f"(vs {default_accuracy * 100:.1f}% at default 0.5)"
    )


@click.command("calibrate")
@click.option(
    "--ground-truth",
    required=True,
    type=click.Path(exists=True),
    help="Path to ground truth file (JSON)",
)
@click.option(
    "--run",
    required=True,
    help="Run ID to calibrate",
)
@click.option(
    "--storage",
    type=click.Path(),
    default=None,
    help="Storage path (default from config)",
)
@click.option(
    "--suite",
    type=str,
    default=None,
    help="Suite ID (auto-detected if only one suite exists)",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output file for calibration JSON",
)
@click.option(
    "--grader",
    default="llm_judge",
    help="Grader to calibrate (default: llm_judge)",
)
def calibrate(
    ground_truth: str,
    run: str,
    storage: str | None,
    suite: str | None,
    output: str | None,
    grader: str,
) -> None:
    _run_calibration(ground_truth, run, storage, suite, output, grader)


def _run_calibration(
    ground_truth: str,
    run: str,
    storage_path: str | None,
    suite_id: str | None,
    output: str | None,
    grader: str,
) -> None:
    asyncio.run(_run_calibration_async(ground_truth, run, storage_path, suite_id, output, grader))


async def _run_calibration_async(
    ground_truth: str,
    run: str,
    storage_path: str | None,
    suite_id: str | None,
    output: str | None,
    grader: str,
) -> None:
    ground_truth_path = Path(ground_truth)

    with open(ground_truth_path) as f:
        gt_data = json.load(f)

    if not isinstance(gt_data, dict):
        console.print("[red]Error:[/red] Ground truth file must be a JSON object")
        raise SystemExit(1)

    config = get_config()
    effective_storage_path = storage_path or str(config.storage_path_resolved())
    storage = FileStorage(base_path=effective_storage_path)

    if suite_id is None:
        suite_ids = await storage.list_suites()
        if len(suite_ids) == 1:
            suite_id = suite_ids[0]
        elif len(suite_ids) == 0:
            console.print("[red]Error:[/red] No suites found in storage")
            raise SystemExit(1)
        else:
            console.print("[red]Error:[/red] Multiple suites found. Specify --suite")
            console.print(f"[dim]Available suites: {', '.join(suite_ids)}[/dim]")
            raise SystemExit(1)

    summary = await storage.load_summary(suite_id, run)
    if summary is None:
        console.print(f"[red]Error:[/red] Run {run} not found in suite {suite_id}")
        raise SystemExit(1)

    samples: list[CalibrationSample] = []
    unlabeled_trials: list[str] = []

    for trial in summary.trials:
        trial_id = trial.id

        if trial_id not in gt_data:
            unlabeled_trials.append(trial_id)
            continue

        actual = gt_data[trial_id]
        if not isinstance(actual, bool):
            console.print(
                f"[yellow]Warning:[/yellow] Ground truth for {trial_id} is not a boolean, skipping"
            )
            continue

        if trial.result is None:
            continue

        grader_result = None
        for gr in trial.result.grader_results:
            if gr.grader_type == grader:
                grader_result = gr
                break

        if grader_result is None:
            continue

        sample = CalibrationSample(
            predicted=grader_result.score,
            actual=actual,
            trial_id=trial_id,
        )
        samples.append(sample)

    if unlabeled_trials:
        console.print(
            f"[yellow]Warning:[/yellow] {len(unlabeled_trials)} trial(s) not in ground truth: "
            f"{', '.join(unlabeled_trials[:5])}{'...' if len(unlabeled_trials) > 5 else ''}"
        )

    curve = CalibrationCurve.compute(samples)
    recommended_threshold, rationale = _compute_recommended_threshold(samples)

    calibration_result = {
        "grader_name": grader,
        "run_id": run,
        "ground_truth_file": str(ground_truth_path),
        "samples_matched": len(samples),
        "samples_total": len(gt_data),
        "metrics": {
            "ece": curve.ece,
            "brier_score": curve.brier_score,
        },
        "recommended_threshold": recommended_threshold,
        "rationale": rationale,
    }

    console.print(
        Panel.fit(
            f"[bold]Calibration Report[/bold]\n"
            f"Run: [cyan]{run}[/cyan]\n"
            f"Grader: [yellow]{grader}[/yellow]\n"
            f"Ground Truth: [dim]{ground_truth}[/dim]"
        )
    )

    metrics_table = Table(
        title="Calibration Metrics",
        show_header=True,
        header_style="bold",
    )
    metrics_table.add_column("Metric")
    metrics_table.add_column("Value", justify="right")

    metrics_table.add_row("Expected Calibration Error (ECE)", f"{curve.ece:.4f}")
    metrics_table.add_row("Brier Score", f"{curve.brier_score:.4f}")
    metrics_table.add_row(
        "Recommended Threshold", f"{calibration_result['recommended_threshold']:.2f}"
    )
    metrics_table.add_row("Samples Matched", str(len(samples)))
    metrics_table.add_row("Samples in Ground Truth", str(len(gt_data)))

    console.print(metrics_table)

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(calibration_result, f, indent=2)
        console.print(f"[green]Calibration data written to:[/green] {output}")
    else:
        console.print("\n[bold]Calibration JSON:[/bold]")
        console.print(json.dumps(calibration_result, indent=2))
