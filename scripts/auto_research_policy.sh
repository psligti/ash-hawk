#!/usr/bin/env bash
set -euo pipefail

AGENT="${1:-bolt-merlin}"
ITERATIONS="${2:-100}"
PROMOTION_THRESHOLD="${3:-3}"
EVAL_DIR="${4:-evals}"

printf "\n[improve-wrapper] agent=%s iterations=%s promotion_threshold=%s eval_dir=%s\n" \
  "$AGENT" "$ITERATIONS" "$PROMOTION_THRESHOLD" "$EVAL_DIR"
printf "[improve-wrapper] legacy auto-research modules were removed; forwarding to the live CLI.\n"
printf "[improve-wrapper] promotion_threshold is ignored by the current improve loop.\n\n"

exec uv run ash-hawk improve "$EVAL_DIR" \
  --agent "$AGENT" \
  --max-iterations "$ITERATIONS" \
  --threshold 0.02 \
  --eval-repeats 3 \
  --integrity-repeats 3
