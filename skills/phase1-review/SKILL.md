---
name: phase1-review
description: Applies deterministic review to suspicious-run traces.
version: 1.0.0
metadata:
  id: skill.phase1_review
  name: phase1-review
  kind: skill
  version: 1.0.0
  status: active
  summary: Applies deterministic review to suspicious-run traces.
  goal: Turn audit traces into explicit suspicious-run and failure-family signals.
  file: skills/phase1-review.md
  category: reasoning
  scope: narrow
  when_to_use:
  - When audit traces need deterministic review
  - When suspicious behavior is possible
  when_not_to_use:
  - When there is no audit or failure context
  triggers:
  - suspicious-run investigation
  anti_triggers:
  - pure verification tasks
  prerequisites:
  - audit_context is available
  - failure_context is available
  inputs_expected:
  - trace evidence
  - failure context
  procedure:
  - Review trace
  - Classify failure family
  - Update failure context
  decision_points:
  - suspicious vs normal
  - family assignment
  fallback_strategy:
  - Prefer unknown family over weak assignment
  outputs:
    required_elements:
    - failure family
    - suspicious review evidence
  completion_criteria:
  - Failure family is explicit
  - suspicious review state is explicit
  escalation_rules:
  - Escalate when the trace cannot support a family assignment
  dependencies:
    tools: &id001
    - call_llm_structured
    related_skills:
    - triage
    related_agents:
    - researcher
  examples:
  - description: Review a suspicious trace
    input: 'A run may have failed for a suspicious or policy-relevant reason.

      '
    expected_behavior:
    - reviews the trace
    - classifies a failure family
  tool_names: *id001
  input_contexts:
  - audit_context
  - failure_context
  output_contexts:
  - failure_context
  memory_read_scopes: []
  memory_write_scopes: []
  description: Applies deterministic review to suspicious-run traces.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/phase1-review.md
allowed-tools:
- call_llm_structured
---

# Purpose
Turn audit traces into explicit suspicious-run and failure-family signals.

# Use This Skill When
- When audit traces need deterministic review
- When suspicious behavior is possible

# Do Not Use This Skill When
- When there is no audit or failure context

# Triggers
- suspicious-run investigation

# Anti-Triggers
- pure verification tasks

# Prerequisites
- audit_context is available
- failure_context is available

# Inputs Expected
- trace evidence
- failure context

# Procedure
1. Confirm the skill applies.
2. Review trace
3. Classify failure family
4. Update failure context

# Decision Points
- suspicious vs normal
- family assignment

# Fallback Strategy
- Prefer unknown family over weak assignment

# Tool Contract
## Available Tools
- call_llm_structured

## Required Input Contexts
- audit_context
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
- failure family
- suspicious review evidence

# Completion Criteria
- Failure family is explicit
- suspicious review state is explicit

# Escalation Rules
- Escalate when the trace cannot support a family assignment

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
