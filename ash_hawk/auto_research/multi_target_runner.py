"""Multi-target parallel improvement cycle runner.

Wraps the core ``run_cycle`` function and executes improvement cycles for
multiple targets concurrently, controlled by an ``asyncio.Semaphore``.
Each target receives its own ``DawnKestrelInjector`` to avoid cache
contamination between parallel runs.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ash_hawk.auto_research.cycle_runner import run_cycle
from ash_hawk.auto_research.types import (
    CycleResult,
    CycleStatus,
    MultiTargetResult,
)

logger = logging.getLogger(__name__)


class TargetCandidate(Protocol):
    name: str
    discovered_path: Path
    target_type: Any


class MultiTargetCycleRunner:
    """Run improvement cycles for multiple targets in parallel.

    Uses ``asyncio.Semaphore`` to bound the number of concurrently executing
    cycles, and aggregates results into a ``MultiTargetResult``.

    Args:
        project_root: Root of the project being evaluated.
        llm_client: Optional LLM client instance.  When *None*, each
            ``run_cycle`` call will create its own client.
        max_concurrent: Maximum number of targets to evaluate in parallel.
        convergence_window: Number of recent iterations to consider for
            convergence detection (passed through to ``run_cycle``).
        convergence_variance_threshold: Score-variance threshold that
            triggers convergence (informational — stored for reference).
    """

    def __init__(
        self,
        project_root: Path,
        llm_client: Any = None,
        max_concurrent: int = 4,
        convergence_window: int = 5,
        convergence_variance_threshold: float = 0.001,
    ) -> None:
        self._project_root = project_root
        self._llm_client = llm_client
        self._max_concurrent = max_concurrent
        self._convergence_window = convergence_window
        self._convergence_variance_threshold = convergence_variance_threshold
        self._semaphore: asyncio.Semaphore | None = None

    async def run_all_targets(
        self,
        scenarios: list[Path],
        targets: list[TargetCandidate],
        iterations_per_target: int = 50,
        threshold: float = 0.02,
        storage_path: Path | None = None,
    ) -> MultiTargetResult:
        """Run improvement cycles for all targets in parallel.

        Args:
            scenarios: Scenario YAML paths to evaluate against.
            targets: Improvement targets to optimise.
            iterations_per_target: Max iterations per target cycle.
            threshold: Minimum score delta to keep an iteration.
            storage_path: Base directory for artefacts.  When *None* a
                default under ``.ash-hawk/auto-research`` is used.

        Returns:
            Aggregated ``MultiTargetResult`` with per-target results.
        """
        if not targets:
            return MultiTargetResult(
                agent_name="unknown",
                completed_at=datetime.now(UTC),
            )

        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        base_storage = storage_path or Path(".ash-hawk/auto-research")

        agent_name = targets[0].name

        started_at = datetime.now(UTC)

        tasks = [
            self._run_single_target(
                target=target,
                scenarios=scenarios,
                iterations=iterations_per_target,
                threshold=threshold,
                storage_path=base_storage / target.name,
            )
            for target in targets
        ]

        completed = await asyncio.gather(*tasks, return_exceptions=True)

        target_results: dict[str, CycleResult] = {}
        for target, outcome in zip(targets, completed):
            if isinstance(outcome, BaseException):
                target_results[target.name] = CycleResult(
                    agent_name=target.name,
                    target_path=str(target.discovered_path),
                    target_type=target.target_type,
                    scenario_paths=[str(s) for s in scenarios],
                    status=CycleStatus.ERROR,
                    error_message=str(outcome),
                    completed_at=datetime.now(UTC),
                )
            else:
                target_results[target.name] = outcome

        overall_improvement = self._compute_overall_improvement(target_results)

        best_target = ""
        best_delta = -float("inf")
        for name, result in target_results.items():
            if result.improvement_delta > best_delta:
                best_delta = result.improvement_delta
                best_target = name

        all_converged = all(r.status == CycleStatus.CONVERGED for r in target_results.values())

        return MultiTargetResult(
            agent_name=agent_name,
            target_results=target_results,
            overall_improvement=overall_improvement,
            best_target=best_target,
            converged=all_converged,
            started_at=started_at,
            completed_at=datetime.now(UTC),
        )

    async def _run_single_target(
        self,
        target: TargetCandidate,
        scenarios: list[Path],
        iterations: int,
        threshold: float,
        storage_path: Path,
    ) -> CycleResult:
        """Run cycle for a single target, gated by the semaphore.

        Each invocation passes ``explicit_targets`` so ``run_cycle`` knows
        which file to improve.  The caller is responsible for ensuring each
        ``ImprovementTarget`` carries its own ``DawnKestrelInjector``.
        """
        assert self._semaphore is not None  # noqa: S101

        async with self._semaphore:
            logger.info(
                "Starting cycle for target %s (%s)",
                target.name,
                target.target_type.value,
            )
            return await run_cycle(
                scenarios=scenarios,
                iterations=iterations,
                threshold=threshold,
                storage_path=storage_path,
                llm_client=self._llm_client,
                project_root=self._project_root,
                explicit_targets=[target.discovered_path],
            )

    def _compute_overall_improvement(self, results: dict[str, CycleResult]) -> float:
        """Compute weighted average improvement across targets.

        Weights are based on the number of applied iterations for each
        target.  Targets with zero applied iterations contribute zero
        weight.  If no target has applied iterations the simple mean of
        ``improvement_delta`` values is returned instead.
        """
        if not results:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for result in results.values():
            weight = float(len(result.applied_iterations))
            weighted_sum += weight * result.improvement_delta
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight

        deltas = [r.improvement_delta for r in results.values()]
        return sum(deltas) / len(deltas)
