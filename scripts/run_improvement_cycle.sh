#!/bin/bash
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
echo "Starting improvement cycle via 'ash-hawk improve'..."
echo ""

exec uv run ash-hawk improve "$FIXTURE_DIR" \
  --agent "$AGENT" \
  --max-iterations "$ITERATIONS" \
  --threshold 0.02 \
  --eval-repeats 3 \
  --integrity-repeats 3
