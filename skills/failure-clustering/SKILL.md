---
name: failure-clustering
description: Groups related failures into actionable clusters.
version: 1.0.0
metadata:
  id: skill.failure_clustering
  name: failure-clustering
  kind: skill
  version: 1.0.0
  status: active
  summary: Groups related failures into actionable clusters.
  goal: Reduce many related failures into clearer clusters for downstream action.
  file: skills/failure-clustering.md
  category: reasoning
  scope: narrow
  when_to_use:
  - When multiple related failures exist
  - When clustering improves next-step choice
  when_not_to_use:
  - When only one failure exists and clustering adds no value
  triggers:
  - related failures
  - clustering step
  anti_triggers:
  - single isolated failure
  prerequisites:
  - failure_context is available
  inputs_expected:
  - failure set
  procedure:
  - Inspect failures
  - Cluster them
  - Update failure context
  decision_points:
  - clustering adds value vs not
  fallback_strategy:
  - Keep failures separate if clustering is not meaningful
  outputs:
    required_elements:
    - clustered failures
  completion_criteria:
  - Clustered failures are explicit
  escalation_rules:
  - Escalate when failures cannot be clustered meaningfully
  dependencies:
    tools: &id001
    - call_llm_structured
    related_skills: []
    related_agents:
    - researcher
  examples:
  - description: Cluster a set of related failures
    input: 'The runtime has multiple failure patterns that may share a root cause.

      '
    expected_behavior:
    - clusters failures into actionable groups
  tool_names: *id001
  input_contexts:
  - failure_context
  output_contexts:
  - failure_context
  memory_read_scopes: []
  memory_write_scopes: []
  description: Groups related failures into actionable clusters.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/failure-clustering.md
allowed-tools:
- call_llm_structured
---

# Purpose
Reduce many related failures into clearer clusters for downstream action.

# Use This Skill When
- When multiple related failures exist
- When clustering improves next-step choice

# Do Not Use This Skill When
- When only one failure exists and clustering adds no value

# Triggers
- related failures
- clustering step

# Anti-Triggers
- single isolated failure

# Prerequisites
- failure_context is available

# Inputs Expected
- failure set

# Procedure
1. Confirm the skill applies.
2. Inspect failures
3. Cluster them
4. Update failure context

# Decision Points
- clustering adds value vs not

# Fallback Strategy
- Keep failures separate if clustering is not meaningful

# Tool Contract
## Available Tools
- call_llm_structured

## Required Input Contexts
- failure_context

## Produced Output Contexts
- failure_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- clustered failures

# Completion Criteria
- Clustered failures are explicit

# Escalation Rules
- Escalate when failures cannot be clustered meaningfully

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
