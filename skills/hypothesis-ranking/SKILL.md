---
name: hypothesis-ranking
description: Prioritizes or reorders candidate hypotheses using durable knowledge
  and fresh evidence.
version: 1.0.0
metadata:
  id: skill.hypothesis_ranking
  name: hypothesis-ranking
  kind: skill
  version: 1.0.0
  status: active
  summary: Prioritizes or reorders candidate hypotheses using durable knowledge and
    fresh evidence.
  goal: Rank current hypotheses into a useful execution order and re-rank them when
    new evidence changes the tradeoffs.
  file: skills/hypothesis-ranking.md
  category: reasoning
  scope: narrow
  when_to_use:
  - When multiple hypotheses exist
  - When memory should influence prioritization
  - When fresh evidence should reshuffle the current order
  when_not_to_use:
  - When only one hypothesis exists
  - When no new evidence changes relative priority
  triggers:
  - need execution order for hypotheses
  - new evidence changes ranking confidence
  anti_triggers:
  - no meaningful hypothesis set
  prerequisites:
  - failure_context is available
  - memory_context is available
  inputs_expected:
  - current hypotheses
  - episodic memory
  - semantic memory
  - fresh evidence already reflected in failure or evaluation context
  procedure:
  - Read hypotheses and memory
  - Rank or re-rank the hypotheses
  - Update failure context
  decision_points:
  - which hypothesis should be first
  - whether new evidence materially changes the existing order
  fallback_strategy:
  - Prefer the simplest high-confidence hypothesis
  - Keep the current ordering when new evidence is inconclusive
  outputs:
    required_elements:
    - ranked hypotheses
  completion_criteria:
  - Ranked hypotheses are explicit
  escalation_rules:
  - Escalate when hypotheses are too weak to rank credibly
  dependencies:
    tools: &id001
    - call_llm_structured
    related_skills: []
    related_agents:
    - researcher
    - coordinator
  examples:
  - description: Rank candidate hypotheses
    input: 'The runtime has several hypotheses and needs a ranked order.

      '
    expected_behavior:
    - uses memory
    - returns ranked hypotheses
  tool_names: *id001
  input_contexts:
  - failure_context
  - memory_context
  output_contexts:
  - failure_context
  memory_read_scopes:
  - episodic_memory
  - semantic_memory
  memory_write_scopes: []
  description: Prioritizes or reorders candidate hypotheses using durable knowledge
    and fresh evidence.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/hypothesis-ranking.md
allowed-tools:
- call_llm_structured
---

# Purpose
Maintain one ranked hypothesis list that stays useful after new evidence lands.

# Procedure
1. Start from the existing ranked list if one already exists.
2. Promote hypotheses backed by fresh evidence or prior successful memory matches.
3. Demote hypotheses contradicted by the latest failure or evaluation signals.
4. Return one explicit ordering and state whether the ranking materially changed.

# Output Contract
- ranked hypotheses
- whether fresh evidence changed the order
