from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from ash_hawk.research.diagnosis import DiagnosisEngine, DiagnosisReport
from ash_hawk.research.strategy_promoter import StrategyPromoter
from ash_hawk.research.target_registry import TargetRegistry
from ash_hawk.research.types import (
    ResearchAction,
    ResearchDecision,
    ResearchLoopConfig,
    ResearchLoopResult,
)
from ash_hawk.research.uncertainty import UncertaintyModel

logger = logging.getLogger(__name__)


class ResearchLoop:
    """Orchestrates the research cycle: diagnose → decide → act → learn."""

    def __init__(
        self,
        config: ResearchLoopConfig | None = None,
        llm_client: object | None = None,
        storage_path: Path | None = None,
    ) -> None:
        self._config = config or ResearchLoopConfig()
        self._llm_client = llm_client
        self._storage_path = storage_path or self._config.storage_path
        self._diagnosis_engine = DiagnosisEngine(llm_client=llm_client)
        self._uncertainty = UncertaintyModel(
            storage_path=storage_path / "uncertainty" if storage_path else None,
            max_hypotheses=self._config.max_hypotheses,
            max_evidence=self._config.max_evidence_per_hypothesis,
        )
        self._target_registry = TargetRegistry(
            storage_path=storage_path / "targets" if storage_path else None,
            min_active=self._config.min_active_targets,
        )
        self._strategy_promoter = StrategyPromoter(
            storage_path=storage_path / "strategies" if storage_path else None,
        )
        self._diagnosis_count = 0

    async def run(
        self,
        scenarios: list[Path],
        project_root: Path | None = None,
    ) -> ResearchLoopResult:
        """Run the full research loop.

        1. Initialize components, load persistent state
        2. Run baseline evaluation
        3. For each iteration:
           a. Diagnose: why did current state happen?
           b. Update uncertainty model
           c. Decide: observe, fix, experiment, or promote?
           d. Execute action (with human approval gate if enabled)
           e. Every d_step_interval: discover new targets
           f. Every prune_interval: prune low-correlation targets
        4. Save all state
        5. Return ResearchLoopResult
        """
        result = ResearchLoopResult(started_at=datetime.now(UTC))
        self._storage_path.mkdir(parents=True, exist_ok=True)

        if self._storage_path:
            self._uncertainty = UncertaintyModel.load(
                self._storage_path / "uncertainty",
                max_hypotheses=self._config.max_hypotheses,
                max_evidence=self._config.max_evidence_per_hypothesis,
            )
            self._target_registry = TargetRegistry.load(
                self._storage_path / "targets",
                min_active=self._config.min_active_targets,
            )
            self._strategy_promoter = StrategyPromoter.load(
                self._storage_path / "strategies",
            )

        result.uncertainty_before = self._uncertainty.uncertainty_level

        for i in range(self._config.iterations):
            if self._diagnosis_count >= self._config.max_diagnoses_per_run:
                logger.info("LLM budget exhausted, stopping research loop")
                break

            diagnosis = await self._diagnose(scenarios, i)
            if diagnosis is None:
                continue

            self._uncertainty.update_from_diagnosis(diagnosis)

            decision = self._decide(diagnosis, i)
            result.decisions.append(decision)

            await self._execute_decision(decision, scenarios, project_root, i)

            if i > 0 and i % self._config.d_step_interval == 0 and project_root:
                new_targets = self._target_registry.discover_targets(project_root)
                logger.info("Discovered %d new targets", len(new_targets))

            if i > 0 and i % self._config.prune_interval == 0:
                pruned = self._target_registry.prune_low_correlation()
                if pruned:
                    logger.info("Pruned %d low-correlation targets", len(pruned))

        await self._save_state()

        result.diagnoses_count = self._diagnosis_count
        result.uncertainty_after = self._uncertainty.uncertainty_level
        result.improvement_delta = result.uncertainty_before - result.uncertainty_after
        result.completed_at = datetime.now(UTC)
        return result

    async def _diagnose(self, scenarios: list[Path], iteration: int) -> DiagnosisReport | None:
        """Run diagnosis. Returns None if budget exhausted or no data."""
        if self._diagnosis_count >= self._config.max_diagnoses_per_run:
            return None
        self._diagnosis_count += 1

        report = await self._diagnosis_engine.diagnose(
            eval_results={},
            trace_events=[],
            scores={"iteration": float(iteration)},
            experiment_log_path=None,
        )
        return report

    def _decide(self, diagnosis: DiagnosisReport, iteration: int) -> ResearchDecision:
        """Decide next action based on diagnosis and uncertainty state."""
        if self._uncertainty.should_observe_before_fixing(self._config.uncertainty_threshold):
            action = ResearchAction.OBSERVE
        elif diagnosis.recommended_action == "promote":
            action = ResearchAction.PROMOTE
        elif self._has_competing_hypotheses():
            action = ResearchAction.EXPERIMENT
        else:
            action = ResearchAction.FIX

        return ResearchDecision(
            action=action,
            rationale=diagnosis.recommended_action,
            target=None,
            expected_info_gain=1.0 - self._uncertainty.uncertainty_level,
            confidence=1.0 - diagnosis.uncertainty_level,
        )

    def _has_competing_hypotheses(self) -> bool:
        competing = self._uncertainty.get_competing_hypotheses()
        return len(competing) >= 2 and all(h.confidence < 0.4 for h in competing)

    async def _execute_decision(
        self,
        decision: ResearchDecision,
        scenarios: list[Path],
        project_root: Path | None,
        iteration: int,
    ) -> None:
        """Execute the decided action."""
        if decision.action == ResearchAction.PROMOTE:
            await self._execute_promote()
        elif decision.action == ResearchAction.OBSERVE:
            logger.info("Iteration %d: observe (high uncertainty)", iteration)
        elif decision.action == ResearchAction.EXPERIMENT:
            logger.info("Iteration %d: experiment (competing hypotheses)", iteration)
        elif decision.action == ResearchAction.FIX:
            if self._config.human_approval_required:
                logger.info(
                    "Iteration %d: fix requires human approval (skipping in auto mode)",
                    iteration,
                )
            else:
                logger.info("Iteration %d: fix (applying mutation)", iteration)

    async def _execute_promote(self) -> None:
        """Check for patterns to promote."""
        patterns = self._strategy_promoter.detect_patterns([])
        for pattern in patterns:
            if self._strategy_promoter.should_promote(pattern):
                await self._strategy_promoter.promote(pattern)

    async def _save_state(self) -> None:
        """Save all persistent state."""
        await self._uncertainty.save()
        await self._target_registry.save()
        await self._strategy_promoter.save()
