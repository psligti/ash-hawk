#!/usr/bin/env bash
set -euo pipefail

AGENT="${1:-bolt-merlin}"
ITERATIONS="${2:-100}"
PROMOTION_THRESHOLD="${3:-3}"
EVAL_PACK="${4:-bolt-merlin-eval}"

STAMP="$(date +%Y%m%d-%H%M%S)"
EXPERIMENT_ID="exp-${AGENT}-policy-${STAMP}"

printf "\n[auto-research] agent=%s iterations=%s promotion_threshold=%s eval_pack=%s\n" \
  "$AGENT" "$ITERATIONS" "$PROMOTION_THRESHOLD" "$EVAL_PACK"
printf "[auto-research] experiment_id=%s\n\n" "$EXPERIMENT_ID"

uv run ash-hawk improve cycle \
  --agent "$AGENT" \
  --iterations "$ITERATIONS" \
  --experiment "$EXPERIMENT_ID" \
  --promotion-threshold "$PROMOTION_THRESHOLD" \
  --eval-pack "$EVAL_PACK"

printf "\n[auto-research] latest curated output for %s\n" "$AGENT"
uv run ash-hawk improve list --agent "$AGENT"

printf "\n[auto-research] done. experiment_id=%s\n" "$EXPERIMENT_ID"
