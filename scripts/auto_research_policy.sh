#!/usr/bin/env bash
set -euo pipefail

AGENT="${1:-bolt-merlin}"
ITERATIONS="${2:-100}"
PROMOTION_THRESHOLD="${3:-3}"
EVAL_DIR="${4:-evals}"

STAMP="$(date +%Y%m%d-%H%M%S)"
EXPERIMENT_ID="exp-${AGENT}-policy-${STAMP}"

printf "\n[auto-research] agent=%s iterations=%s promotion_threshold=%s eval_dir=%s\n" \
  "$AGENT" "$ITERATIONS" "$PROMOTION_THRESHOLD" "$EVAL_DIR"
printf "[auto-research] experiment_id=%s\n\n" "$EXPERIMENT_ID"

uv run python -c "
import asyncio
from pathlib import Path
from ash_hawk.auto_research.cycle_runner import run_cycle, CycleConfig
from ash_hawk.auto_research.knowledge_promotion import PromotionCriteria
from ash_hawk.improvement.guardrails import GuardrailConfig

async def main():
    config = CycleConfig(
        max_iterations=${ITERATIONS},
        target_pass_rate=1.0,
        score_threshold=0.02,
        eval_repeats=3,
        train_ratio=0.7,
        seed=42,
        guardrail_config=GuardrailConfig(
            max_consecutive_holdout_drops=3,
            max_reverts=${ITERATIONS},
            plateau_window=5,
            plateau_threshold=0.02,
        ),
        convergence_window=5,
        convergence_variance_threshold=0.001,
        max_iterations_without_improvement=15,
        promotion_criteria=PromotionCriteria(
            min_improvement=0.05,
            min_consecutive_successes=${PROMOTION_THRESHOLD},
            max_regression=0.02,
        ),
        lessons_dir=Path('.ash-hawk/lessons'),
    )

    result = await run_cycle(
        suite_path='${EVAL_DIR}',
        agent_name='${AGENT}',
        config=config,
    )

    print()
    print(f'Status: {result.status.value}')
    print(f'Score: {result.initial_score:.3f} -> {result.final_score:.3f} ({result.improvement_delta:+.3f})')
    print(f'Iterations: {result.total_iterations} (applied={result.applied_count} reverted={result.reverted_count})')
    print(f'Promoted lessons: {result.promoted_lessons}')
    print(f'Duration: {result.duration_seconds:.1f}s')

asyncio.run(main())
"

printf "\n[auto-research] done. experiment_id=%s\n" "$EXPERIMENT_ID"
