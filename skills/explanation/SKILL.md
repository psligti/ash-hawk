---
name: explanation
description: Produces a structured explanation for the current failure state.
version: 1.0.0
metadata:
  id: skill.explanation
  name: explanation
  kind: skill
  version: 1.0.0
  status: active
  summary: Produces a structured explanation for the current failure state.
  goal: Turn failure and audit signals into a readable, useful diagnosis.
  file: skills/explanation.md
  category: reasoning
  scope: narrow
  when_to_use:
  - When a failure needs explanation
  - When audit context can support interpretation
  when_not_to_use:
  - When no failure context exists
  triggers:
  - need for diagnosis
  - handoff explanation
  anti_triggers:
  - pure execution tasks
  prerequisites:
  - failure_context is available
  - audit_context is available
  inputs_expected:
  - failure state
  - audit trace
  procedure:
  - Read failure state
  - Generate explanation
  - Update failure context
  decision_points:
  - sufficient evidence vs escalation
  fallback_strategy:
  - State uncertainty explicitly when evidence is weak
  outputs:
    required_elements:
    - structured explanation
  completion_criteria:
  - Explanation is explicit and grounded in available context
  escalation_rules:
  - Escalate when evidence is too weak for a responsible explanation
  dependencies:
    tools: &id001
    - call_llm_structured
    related_skills: []
    related_agents:
    - researcher
  examples:
  - description: Explain why a run failed
    input: 'The runtime has failure context and needs a human-readable diagnosis.

      '
    expected_behavior:
    - generates a structured explanation
  tool_names: *id001
  input_contexts:
  - failure_context
  - audit_context
  output_contexts:
  - failure_context
  memory_read_scopes: []
  memory_write_scopes: []
  description: Produces a structured explanation for the current failure state.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/explanation.md
allowed-tools:
- call_llm_structured
---

# Purpose
Turn failure and audit signals into a readable, useful diagnosis.

# Use This Skill When
- When a failure needs explanation
- When audit context can support interpretation

# Do Not Use This Skill When
- When no failure context exists

# Triggers
- need for diagnosis
- handoff explanation

# Anti-Triggers
- pure execution tasks

# Prerequisites
- failure_context is available
- audit_context is available

# Inputs Expected
- failure state
- audit trace

# Procedure
1. Confirm the skill applies.
2. Read failure state
3. Generate explanation
4. Update failure context

# Decision Points
- sufficient evidence vs escalation

# Fallback Strategy
- State uncertainty explicitly when evidence is weak

# Tool Contract
## Available Tools
- call_llm_structured

## Required Input Contexts
- failure_context
- audit_context

## Produced Output Contexts
- failure_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- structured explanation

# Completion Criteria
- Explanation is explicit and grounded in available context

# Escalation Rules
- Escalate when evidence is too weak for a responsible explanation

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
