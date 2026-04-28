---
name: baseline-evaluation
description: Establishes the current measured baseline.
version: 1.0.0
metadata:
  id: skill.baseline_evaluation
  name: baseline-evaluation
  kind: skill
  version: 1.0.0
  status: active
  summary: Establishes the current measured baseline.
  goal: Run baseline evaluation and record the current state before deeper actions.
  file: skills/baseline-evaluation.md
  category: verification
  scope: narrow
  when_to_use:
  - When no current baseline exists
  - When verification depends on a known starting score
  when_not_to_use:
  - When a fresh baseline already exists and is still valid
  triggers:
  - missing baseline
  - pre-verification measurement
  anti_triggers:
  - irrelevant when no evaluation path exists
  prerequisites:
  - goal_context is available
  - workspace_context is available
  inputs_expected:
  - goal metadata
  - workspace state
  procedure:
  - Run the baseline evaluation path
  - Repeat if needed
  - Aggregate the resulting score
  decision_points:
  - whether repeated evaluation is needed
  fallback_strategy:
  - Use the simplest baseline path available
  outputs:
    required_elements:
    - baseline summary
    - audit trace
  completion_criteria:
  - Baseline summary is present
  - Audit evidence of the evaluation exists
  escalation_rules:
  - Escalate when evaluation cannot be run credibly
  dependencies:
    tools: &id001
    - run_eval
    - run_baseline_eval
    - run_eval_repeated
    - aggregate_scores
    related_skills:
    - verification
    related_agents:
    - verifier
  examples:
  - description: Establish a baseline before verification
    input: 'The run needs a baseline before acceptance can be decided.

      '
    expected_behavior:
    - runs baseline evaluation
    - aggregates the result if needed
  tool_names: *id001
  input_contexts:
  - goal_context
  - workspace_context
  output_contexts:
  - evaluation_context
  - audit_context
  memory_read_scopes: []
  memory_write_scopes: []
  description: Establishes the current measured baseline.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/baseline-evaluation.md
allowed-tools:
- run_eval
- run_baseline_eval
- run_eval_repeated
- aggregate_scores
---

# Purpose
Run baseline evaluation and record the current state before deeper actions.

# Use This Skill When
- When no current baseline exists
- When verification depends on a known starting score

# Do Not Use This Skill When
- When a fresh baseline already exists and is still valid

# Triggers
- missing baseline
- pre-verification measurement

# Anti-Triggers
- irrelevant when no evaluation path exists

# Prerequisites
- goal_context is available
- workspace_context is available

# Inputs Expected
- goal metadata
- workspace state

# Procedure
1. Confirm the skill applies.
2. Run the baseline evaluation path
3. Repeat if needed
4. Aggregate the resulting score

# Decision Points
- whether repeated evaluation is needed

# Fallback Strategy
- Use the simplest baseline path available

# Tool Contract
## Available Tools
- run_eval
- run_baseline_eval
- run_eval_repeated
- aggregate_scores

## Required Input Contexts
- goal_context
- workspace_context

## Produced Output Contexts
- evaluation_context
- audit_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- baseline summary
- audit trace

# Completion Criteria
- Baseline summary is present
- Audit evidence of the evaluation exists

# Escalation Rules
- Escalate when evaluation cannot be run credibly

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
