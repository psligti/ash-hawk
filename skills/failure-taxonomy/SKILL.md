---
name: failure-taxonomy
description: Classifies failures into reusable families.
version: 1.0.0
metadata:
  id: skill.failure_taxonomy
  name: failure-taxonomy
  kind: skill
  version: 1.0.0
  status: active
  summary: Classifies failures into reusable families.
  goal: Turn failures into stable categories for downstream reasoning.
  file: skills/failure-taxonomy.md
  category: reasoning
  scope: narrow
  when_to_use:
  - When failures need family classification
  - When audit evidence exists
  when_not_to_use:
  - When no failure context exists
  triggers:
  - family assignment
  anti_triggers:
  - weak or missing evidence
  prerequisites:
  - failure_context is available
  - audit_context is available
  inputs_expected:
  - failure evidence
  - audit evidence
  procedure:
  - Inspect failure state
  - Classify the family
  - Update failure context
  decision_points:
  - which family best fits
  fallback_strategy:
  - Prefer unknown over weak family assignment
  outputs:
    required_elements:
    - failure family
  completion_criteria:
  - Failure family is explicit
  escalation_rules:
  - Escalate when family assignment is too weak
  dependencies:
    tools: &id001
    - call_llm_structured
    related_skills: []
    related_agents:
    - researcher
  examples:
  - description: Classify a failure into a family
    input: 'The runtime has failure evidence and needs a stable family assignment.

      '
    expected_behavior:
    - classifies the failure family
  tool_names: *id001
  input_contexts:
  - failure_context
  - audit_context
  output_contexts:
  - failure_context
  memory_read_scopes: []
  memory_write_scopes: []
  description: Classifies failures into reusable families.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/failure-taxonomy.md
allowed-tools:
- call_llm_structured
---

# Purpose
Turn failures into stable categories for downstream reasoning.

# Use This Skill When
- When failures need family classification
- When audit evidence exists

# Do Not Use This Skill When
- When no failure context exists

# Triggers
- family assignment

# Anti-Triggers
- weak or missing evidence

# Prerequisites
- failure_context is available
- audit_context is available

# Inputs Expected
- failure evidence
- audit evidence

# Procedure
1. Confirm the skill applies.
2. Inspect failure state
3. Classify the family
4. Update failure context

# Decision Points
- which family best fits

# Fallback Strategy
- Prefer unknown over weak family assignment

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
- failure family

# Completion Criteria
- Failure family is explicit

# Escalation Rules
- Escalate when family assignment is too weak

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
