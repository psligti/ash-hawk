# type-hygiene: skip-file
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ash_hawk.improve.diagnose import diagnose_failures
from ash_hawk.improve.patch import ProposedPatch, propose_patch, write_patch

if TYPE_CHECKING:
    from ash_hawk.agents.agent_mutator import AgentMutator
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
    mutation_history: list[dict[str, Any]] = field(default_factory=list)
    convergence_achieved: bool = False


async def improve(
    suite_path: str,
    agent_name: str = "build",
    agent_path: Path | None = None,
    target: float = 1.0,
    max_iterations: int = 5,
    trace_dir: Path | None = None,
    output_dir: Path | None = None,
    iteration_timeout_seconds: float = 300.0,
    eval_repeats: int = 3,
    score_threshold: float = 0.02,
) -> ImprovementResult:
    patches: list[ProposedPatch] = []
    initial_pass_rate = 0.0
    final_pass_rate = 0.0
    mutation_history: list[dict[str, Any]] = []
    convergence_achieved = False

    mutator: AgentMutator | None = None
    snapshot_hash: str | None = None
    agent_content: dict[str, str] | None = None

    if agent_path is not None:
        from ash_hawk.agents.agent_mutator import AgentMutator

        uuid_hex = uuid.uuid4().hex
        mutator = AgentMutator(agent_path, run_id=f"improve-{uuid_hex[:8]}")
        snapshot_hash = mutator.snapshot()
        agent_content = mutator.scan()

    try:
        for i in range(max_iterations):
            # --- Run evals N times and compute mean pass rate ---
            pass_rates: list[float] = []
            for _ in range(eval_repeats):
                try:
                    summary = await _run_eval(
                        suite_path, agent_name, iteration_timeout_seconds, agent_path
                    )
                    pass_rates.append(summary.metrics.pass_rate)
                except Exception:
                    logger.warning("Eval run failed in iteration %d", i, exc_info=True)

            if not pass_rates:
                continue

            mean_pass_rate = sum(pass_rates) / len(pass_rates)
            if i == 0:
                initial_pass_rate = mean_pass_rate
            final_pass_rate = mean_pass_rate

            logger.info(
                "Iteration %d: mean_pass_rate=%.4f (runs=%d) target=%.2f",
                i,
                mean_pass_rate,
                len(pass_rates),
                target,
            )

            if mean_pass_rate >= target:
                logger.info("Target reached at iteration %d", i)
                convergence_achieved = True
                break

            # --- Gather failures from the last eval run ---
            last_summary = await _run_eval(
                suite_path, agent_name, iteration_timeout_seconds, agent_path
            )
            failures = [
                t
                for t in last_summary.trials
                if t.result is not None and not t.result.aggregate_passed
            ]
            if not failures:
                logger.info("No failures found, stopping")
                break

            # --- Diagnose with agent content context ---
            try:
                diagnoses = await diagnose_failures(failures, trace_dir, agent_content)
            except Exception:
                logger.warning("Diagnosis failed in iteration %d", i, exc_info=True)
                continue

            logger.info("Diagnosed %d failures in iteration %d", len(diagnoses), i)

            # --- Propose and apply patches ---
            mean_old = mean_pass_rate
            for diagnosis in diagnoses:
                try:
                    patch = await propose_patch(diagnosis, agent_content)
                    patches.append(patch)

                    if mutator is not None and patch.agent_relative_path and patch.content:
                        mutator.write_file(patch.agent_relative_path, patch.content)
                        logger.info("Mutated agent file: %s", patch.agent_relative_path)
                    else:
                        patch_path = write_patch(patch, output_dir)
                        logger.info("Patch proposed: %s for %s", patch_path, patch.file_path)
                except Exception:
                    logger.warning(
                        "Patch proposal failed for %s", diagnosis.trial_id, exc_info=True
                    )

            # --- Re-evaluate after mutations ---
            if mutator is not None and any(
                p.agent_relative_path for p in patches[-len(diagnoses) :]
            ):
                new_pass_rates: list[float] = []
                for _ in range(eval_repeats):
                    try:
                        new_summary = await _run_eval(
                            suite_path, agent_name, iteration_timeout_seconds, agent_path
                        )
                        new_pass_rates.append(new_summary.metrics.pass_rate)
                    except Exception:
                        logger.warning(
                            "Post-mutation eval failed in iteration %d", i, exc_info=True
                        )

                if new_pass_rates:
                    mean_new = sum(new_pass_rates) / len(new_pass_rates)
                    improvement = mean_new - mean_old

                    iteration_record: dict[str, Any] = {
                        "iteration": i,
                        "mean_pass_rate_before": mean_old,
                        "mean_pass_rate_after": mean_new,
                        "improvement": improvement,
                        "kept": False,
                    }

                    if improvement > score_threshold:
                        logger.info(
                            "Mutation improved score by %.4f (> threshold %.4f), keeping",
                            improvement,
                            score_threshold,
                        )
                        iteration_record["kept"] = True
                        final_pass_rate = mean_new
                        # Re-snapshot for next iteration baseline
                        snapshot_hash = mutator.snapshot()
                        agent_content = mutator.scan()
                    else:
                        logger.info(
                            "Mutation did not improve enough (%.4f <= threshold %.4f), reverting",
                            improvement,
                            score_threshold,
                        )
                        mutator.revert_all()
                        if snapshot_hash is not None:
                            snapshot_hash = mutator.snapshot()
                            agent_content = mutator.scan()

                    mutation_history.append(iteration_record)

    finally:
        if mutator is not None:
            mutator.cleanup()

    return ImprovementResult(
        iterations=max_iterations,
        initial_pass_rate=initial_pass_rate,
        final_pass_rate=final_pass_rate,
        patches_proposed=patches,
        patches_applied=[],
        trace_path=trace_dir,
        mutation_history=mutation_history,
        convergence_achieved=convergence_achieved,
    )


async def _run_eval(
    suite_path: str,
    agent_name: str,
    timeout: float,
    agent_path: Path | None = None,
) -> EvalRunSummary:
    from ash_hawk.scenario.runner import run_scenarios_async

    return await asyncio.wait_for(
        run_scenarios_async([suite_path], agent_path=agent_path),
        timeout=timeout,
    )
