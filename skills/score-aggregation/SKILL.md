---
name: score-aggregation
description: Aggregates repeated evaluation results into a single score.
version: 1.0.0
metadata:
  id: skill.score_aggregation
  name: score-aggregation
  kind: skill
  version: 1.0.0
  status: active
  summary: Aggregates repeated evaluation results into a single score.
  goal: Produce a deterministic aggregated score from repeated evaluation runs.
  file: skills/score-aggregation.md
  category: verification
  scope: narrow
  when_to_use:
  - When repeated evaluation results exist
  - When one aggregated score is needed
  when_not_to_use:
  - When only a single result exists and no aggregation is needed
  triggers:
  - multiple evaluation results
  anti_triggers:
  - no repeated evaluations
  prerequisites:
  - evaluation_context is available
  inputs_expected:
  - repeated evaluation outputs
  procedure:
  - Aggregate scores
  - Update evaluation context
  decision_points:
  - whether enough data exists to aggregate
  fallback_strategy:
  - Use the most credible single score when aggregation is not meaningful
  outputs:
    required_elements:
    - aggregated score
  completion_criteria:
  - Aggregated score is explicit
  escalation_rules:
  - Escalate when repeated results are inconsistent beyond interpretation
  dependencies:
    tools: &id001
    - aggregate_scores
    related_skills:
    - baseline-evaluation
    related_agents:
    - verifier
  examples:
  - description: Aggregate repeated evals
    input: 'Repeated validation produced multiple scores that must be combined.

      '
    expected_behavior:
    - aggregates scores
    - updates evaluation context
  tool_names: *id001
  input_contexts:
  - evaluation_context
  output_contexts:
  - evaluation_context
  memory_read_scopes: []
  memory_write_scopes: []
  description: Aggregates repeated evaluation results into a single score.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/score-aggregation.md
allowed-tools:
- aggregate_scores
---

# Purpose
Produce a deterministic aggregated score from repeated evaluation runs.

# Use This Skill When
- When repeated evaluation results exist
- When one aggregated score is needed

# Do Not Use This Skill When
- When only a single result exists and no aggregation is needed

# Triggers
- multiple evaluation results

# Anti-Triggers
- no repeated evaluations

# Prerequisites
- evaluation_context is available

# Inputs Expected
- repeated evaluation outputs

# Procedure
1. Confirm the skill applies.
2. Aggregate scores
3. Update evaluation context

# Decision Points
- whether enough data exists to aggregate

# Fallback Strategy
- Use the most credible single score when aggregation is not meaningful

# Tool Contract
## Available Tools
- aggregate_scores

## Required Input Contexts
- evaluation_context

## Produced Output Contexts
- evaluation_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- aggregated score

# Completion Criteria
- Aggregated score is explicit

# Escalation Rules
- Escalate when repeated results are inconsistent beyond interpretation

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
