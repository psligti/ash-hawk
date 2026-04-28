---
name: skip-decision
description: Decides whether a hypothesis should be skipped.
version: 1.0.0
metadata:
  id: skill.skip_decision
  name: skip-decision
  kind: skill
  version: 1.0.0
  status: active
  summary: Decides whether a hypothesis should be skipped.
  goal: Avoid low-value hypotheses using durable memory signals.
  file: skills/skip-decision.md
  category: reasoning
  scope: narrow
  when_to_use:
  - When choosing among hypotheses
  - When memory signals suggest skipping
  when_not_to_use:
  - When there is only one hypothesis or no relevant memory
  triggers:
  - hypothesis skip check
  anti_triggers:
  - no failure context
  prerequisites:
  - memory_context is available
  - failure_context is available
  inputs_expected:
  - memory signals
  - failure state
  procedure:
  - Check skip conditions
  - Update failure context
  decision_points:
  - skip vs keep
  fallback_strategy:
  - Keep the hypothesis when skip evidence is weak
  outputs:
    required_elements:
    - skip decision
  completion_criteria:
  - Skip decision is explicit
  escalation_rules:
  - Escalate when skip confidence is too low for a safe decision
  dependencies:
    tools: &id001 []
    related_skills: []
    related_agents:
    - memory_manager
    - coordinator
  examples:
  - description: Decide whether to skip a hypothesis
    input: 'The runtime has memory evidence suggesting a hypothesis may be low value.

      '
    expected_behavior:
    - returns a skip decision
  tool_names: *id001
  input_contexts:
  - memory_context
  - failure_context
  output_contexts:
  - failure_context
  memory_read_scopes: []
  memory_write_scopes: []
  description: Decides whether a hypothesis should be skipped.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/skip-decision.md
---

# Purpose
Avoid low-value hypotheses using durable memory signals.

# Use This Skill When
- When choosing among hypotheses
- When memory signals suggest skipping

# Do Not Use This Skill When
- When there is only one hypothesis or no relevant memory

# Triggers
- hypothesis skip check

# Anti-Triggers
- no failure context

# Prerequisites
- memory_context is available
- failure_context is available

# Inputs Expected
- memory signals
- failure state

# Procedure
1. Confirm the skill applies.
2. Check skip conditions
3. Update failure context

# Decision Points
- skip vs keep

# Fallback Strategy
- Keep the hypothesis when skip evidence is weak

# Tool Contract
## Available Tools
- None

## Required Input Contexts
- memory_context
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
- skip decision

# Completion Criteria
- Skip decision is explicit

# Escalation Rules
- Escalate when skip confidence is too low for a safe decision

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
