#!/bin/bash
# Self-improvement cycle runner for python-bugfix skill
# Uses the orchestrated cycle_runner with train/holdout split,
# guardrails, convergence detection, lesson tracking, and knowledge promotion.

set -e  # Exit on error

ITERATIONS=${1:-10}
AGENT=${2:-build}
PROJECT_ROOT=${3:-$(pwd)}

echo "=== Self-Improvement Cycle ==="
echo "Iterations: $ITERATIONS"
echo "Agent: $AGENT"
echo "Project root: $PROJECT_ROOT"
echo ""

# Check that fixtures exist
FIXTURE_DIR="evals/python-bugfix"
if [ ! -d "$FIXTURE_DIR" ]; then
    echo "ERROR: Fixture directory not found: $FIXTURE_DIR"
    exit 1
fi

# Run the improvement cycle
echo ""
echo "Starting improvement cycle..."
echo ""

uv run python -c "
import asyncio
from pathlib import Path
from ash_hawk.auto_research.cycle_runner import run_cycle, CycleConfig
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
            max_reverts=5,
            plateau_window=5,
            plateau_threshold=0.02,
        ),
        convergence_window=5,
        convergence_variance_threshold=0.001,
        max_iterations_without_improvement=10,
        lessons_dir=Path('.ash-hawk/lessons'),
    )

    result = await run_cycle(
        suite_path='${FIXTURE_DIR}',
        agent_name='${AGENT}',
        agent_path=None,
        config=config,
    )

    print()
    print('=== Cycle Complete ===')
    print(f'  Status: {result.status.value}')
    print(f'  Total iterations: {result.total_iterations}')
    print(f'  Initial score: {result.initial_score:.3f}')
    print(f'  Final score: {result.final_score:.3f}')
    print(f'  Improvement: {result.improvement_delta:+.3f} ({result.improvement_delta * 100:+.1f} points)')
    print(f'  Applied: {result.applied_count}  Reverted: {result.reverted_count}')
    print(f'  Promoted lessons: {result.promoted_lessons}')
    print(f'  Duration: {result.duration_seconds:.1f}s')

    if result.convergence_result:
        print(f'  Convergence: {result.convergence_result.reason}')

    if result.guardrail_reason:
        print(f'  Guardrail: {result.guardrail_reason}')

    improvement = result.improvement_delta * 100
    if improvement >= 20:
        print('SUCCESS: Achieved +20pt improvement goal!')
    else:
        print(f'GOAL: Need {20 - improvement:.1f} more points')

asyncio.run(main())
"

echo ""
echo "Cycle complete!"
