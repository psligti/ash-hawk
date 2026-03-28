#!/bin/bash
# Self-improvement cycle runner for python-bugfix skill

set -e  # Exit on error

ITERATIONS=${1:-10}
PROMOTION_THRESHOLD=${2:-3}
PROJECT_ROOT=${3:-$(pwd)}

echo "=== Self-Improvement Cycle ==="
echo "Iterations: $ITERATIONS"
echo "Promotion threshold: $PROMOTION_THRESHOLD"
echo "Project root: $PROJECT_ROOT"
echo ""

# Check that fixtures exist
FIXTURE_DIR="evals/python-bugfix"
if [ ! -d "$FIXTURE_DIR" ]; then
    echo "ERROR: Fixture directory not found: $FIXTURE_DIR"
    exit 1
fi

# Count fixtures
FIXTURE_COUNT=$(find "$FIXTURE_DIR" -maxdepth 1 -type d -name "*-*" | wc -l)
echo "Found $FIXTURE_COUNT fixtures"

if [ "$FIXTURE_COUNT" -lt 10 ]; then
    echo "ERROR: Need at least 10 fixtures for train/heldout split"
    exit 1
fi

# Run the improvement cycle
echo ""
echo "Starting improvement cycle..."
echo ""

uv run python -c "
import asyncio
from pathlib import Path
from ash_hawk.improvement.fixture_splitter import FixtureSplitter
from ash_hawk.improvement.guardrails import GuardrailConfig
from ash_hawk.auto_research.cycle_runner import run_cycle

async def main():
    # Get all fixture paths
    fixture_dir = Path('evals/python-bugfix')
    fixtures = sorted([d for d in fixture_dir.iterdir() if d.is_dir() and d.name.endswith('-') or d.name[0].isdigit()])
    
    # Split into train/heldout
    splitter = FixtureSplitter(seed=42, train_ratio=0.7)
    split = splitter.split(fixtures)
    
    print(f'Training fixtures: {len(split.train)}')
    print(f'Held-out fixtures: {len(split.heldout)}')
    print(f'Training: {[f.name for f in split.train]}')
    print(f'Held-out: {[f.name for f in split.heldout]}')
    
    # Configure guardrails
    guardrail_config = GuardrailConfig(
        max_consecutive_holdout_drops=3,
        max_reverts=5,
        plateau_window=5,
        plateau_threshold=0.02,
    )
    
    # Run the improvement cycle
    result = await run_cycle(
        scenarios=split.train,
        heldout_scenarios=split.heldout,
        iterations=${ITERATIONS},
        threshold=0.02,
        project_root=Path('${PROJECT_ROOT}'),
        guardrail_config=guardrail_config,
    )
    
    print()
    print('=== Cycle Complete ===')
    print(f'Status: {result.status.value}')
    print(f'Total iterations: {result.total_iterations}')
    print(f'Baseline score: {result.initial_score:.3f}')
    print(f'Final score: {result.final_score:.3f}')
    print(f'Improvement: {result.improvement_delta:+.3f}')
    
    if result.final_score is not None and result.initial_score is not None:
        improvement = (result.final_score - result.initial_score) * 100
        print(f'Improvement: {improvement:.1f} absolute points')
        
        if improvement >= 20:
            print('SUCCESS: Achieved +20pt improvement goal!')
        else:
            print(f'GOAL: Need {20 - improvement:.1f} more points')

asyncio.run(main())
"

echo ""
echo "Cycle complete!"
