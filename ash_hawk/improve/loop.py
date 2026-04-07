# type-hygiene: skip-file
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ash_hawk.improve.diagnose import Diagnosis, diagnose_failures
from ash_hawk.improve.patch import ProposedPatch, propose_patch, write_patch

if TYPE_CHECKING:
    from ash_hawk.types import EvalRunSummary

logger = logging.getLogger(__name__)


@dataclass
class ImprovementResult:
    iterations: int
    initial_pass_rate: float
    final_pass_rate: float
    patches_proposed: list[ProposedPatch]
    patches_applied: list[str]
    trace_path: Path | None


async def improve(
    suite_path: str,
    agent_name: str = "build",
    target: float = 1.0,
    max_iterations: int = 5,
    trace_dir: Path | None = None,
    output_dir: Path | None = None,
    iteration_timeout_seconds: float = 300.0,
) -> ImprovementResult:
    patches: list[ProposedPatch] = []
    initial_pass_rate = 0.0
    final_pass_rate = 0.0

    for i in range(max_iterations):
        try:
            summary = await _run_eval(suite_path, agent_name, iteration_timeout_seconds)
        except Exception:
            logger.warning("Eval run failed in iteration %d", i, exc_info=True)
            continue

        pass_rate = summary.metrics.pass_rate
        if i == 0:
            initial_pass_rate = pass_rate
        final_pass_rate = pass_rate

        logger.info("Iteration %d: pass_rate=%.2f target=%.2f", i, pass_rate, target)

        if pass_rate >= target:
            logger.info("Target reached at iteration %d", i)
            break

        failures = [
            t for t in summary.trials if t.result is not None and not t.result.aggregate_passed
        ]
        if not failures:
            logger.info("No failures found, stopping")
            break

        try:
            diagnoses = await diagnose_failures(failures, trace_dir)
        except Exception:
            logger.warning("Diagnosis failed in iteration %d", i, exc_info=True)
            continue

        logger.info("Diagnosed %d failures in iteration %d", len(diagnoses), i)

        for diagnosis in diagnoses:
            try:
                patch = await propose_patch(diagnosis)
                patches.append(patch)
                patch_path = write_patch(patch, output_dir)
                logger.info("Patch proposed: %s for %s", patch_path, patch.file_path)
            except Exception:
                logger.warning("Patch proposal failed for %s", diagnosis.trial_id, exc_info=True)

    return ImprovementResult(
        iterations=max_iterations,
        initial_pass_rate=initial_pass_rate,
        final_pass_rate=final_pass_rate,
        patches_proposed=patches,
        patches_applied=[],
        trace_path=trace_dir,
    )


async def _run_eval(suite_path: str, agent_name: str, timeout: float) -> EvalRunSummary:
    from ash_hawk.scenario.runner import run_scenarios_async

    return await asyncio.wait_for(
        run_scenarios_async([suite_path]),
        timeout=timeout,
    )
