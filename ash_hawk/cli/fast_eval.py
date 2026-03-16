from __future__ import annotations

import asyncio
from pathlib import Path

import click
import yaml
from rich.console import Console

from ash_hawk.execution.fast_eval import FastEvalRunner
from ash_hawk.reporting.fast_eval_report import (
    fast_eval_result_to_json,
    fast_eval_result_to_junit_xml,
    render_fast_eval_table,
)
from ash_hawk.types import FastEvalSuite

console = Console()


@click.command(name="fast-eval")
@click.argument("suite", type=click.Path(exists=True))
@click.option("--parallelism", "-p", type=int, default=None)
@click.option("--tag", "tags", multiple=True)
@click.option("--eval", "eval_ids", multiple=True)
@click.option(
    "--output", "output_format", type=click.Choice(["table", "json", "junit"]), default="table"
)
@click.option("--output-file", type=click.Path(), default=None)
def fast_eval(
    suite: str,
    parallelism: int | None,
    tags: tuple[str, ...],
    eval_ids: tuple[str, ...],
    output_format: str,
    output_file: str | None,
) -> None:
    asyncio.run(
        _run_fast_eval(
            suite_path=suite,
            parallelism=parallelism,
            tags=list(tags),
            eval_ids=list(eval_ids),
            output_format=output_format,
            output_file=output_file,
        )
    )


async def _run_fast_eval(
    suite_path: str,
    parallelism: int | None,
    tags: list[str],
    eval_ids: list[str],
    output_format: str,
    output_file: str | None,
) -> None:
    suite_file = Path(suite_path)
    if not suite_file.exists():
        console.print(f"[red]Error:[/red] Fast eval suite not found: {suite_path}")
        raise SystemExit(1)

    with open(suite_file) as f:
        suite_data = yaml.safe_load(f)

    try:
        suite = FastEvalSuite.model_validate(suite_data)
    except Exception as e:
        console.print(f"[red]Error parsing fast eval suite:[/red] {e}")
        raise SystemExit(1)

    runner = FastEvalRunner(suite=suite, parallelism=parallelism)
    result = await runner.run_suite(
        filter_tags=tags if tags else None,
        eval_ids=eval_ids if eval_ids else None,
    )

    if output_format == "table":
        render_fast_eval_table(console, result)
    elif output_format == "json":
        payload = fast_eval_result_to_json(result)
        if output_file:
            Path(output_file).write_text(payload)
            console.print(f"[green]Wrote:[/green] {output_file}")
        else:
            console.print(payload)
    else:
        payload = fast_eval_result_to_junit_xml(result)
        if output_file:
            Path(output_file).write_text(payload)
            console.print(f"[green]Wrote:[/green] {output_file}")
        else:
            console.print(payload)

    if result.failed_evals > 0:
        raise SystemExit(1)
