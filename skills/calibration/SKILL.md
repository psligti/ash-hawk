---
name: calibration
description: Computes execution confidence from prior evidence.
version: 1.0.0
metadata:
  id: skill.calibration
  name: calibration
  kind: skill
  version: 1.0.0
  status: active
  summary: Computes execution confidence from prior evidence.
  goal: Adjust confidence using prior outcomes and memory context.
  file: skills/calibration.md
  category: reasoning
  scope: narrow
  when_to_use:
  - When confidence should be informed by prior outcomes
  - When execution certainty matters
  when_not_to_use:
  - When no relevant memory exists
  triggers:
  - confidence adjustment
  - prior-outcome influence
  anti_triggers:
  - no memory context
  prerequisites:
  - memory_context is available
  - failure_context is available
  inputs_expected:
  - memory state
  - failure state
  procedure:
  - Read memory and failure state
  - Compute calibration factor
  - Update memory context
  decision_points:
  - whether calibration should alter behavior
  fallback_strategy:
  - Use conservative confidence when evidence is limited
  outputs:
    required_elements:
    - calibration factor
  completion_criteria:
  - Calibration factor is explicit
  escalation_rules:
  - Escalate when confidence cannot be estimated credibly
  dependencies:
    tools: &id001 []
    related_skills: []
    related_agents:
    - memory_manager
  examples:
  - description: Calibrate confidence from prior outcomes
    input: 'The runtime wants confidence adjusted by prior evidence.

      '
    expected_behavior:
    - computes calibration factor
  tool_names: *id001
  input_contexts:
  - memory_context
  - failure_context
  output_contexts:
  - memory_context
  memory_read_scopes: []
  memory_write_scopes: []
  description: Computes execution confidence from prior evidence.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/calibration.md
---

# Purpose
Adjust confidence using prior outcomes and memory context.

# Use This Skill When
- When confidence should be informed by prior outcomes
- When execution certainty matters

# Do Not Use This Skill When
- When no relevant memory exists

# Triggers
- confidence adjustment
- prior-outcome influence

# Anti-Triggers
- no memory context

# Prerequisites
- memory_context is available
- failure_context is available

# Inputs Expected
- memory state
- failure state

# Procedure
1. Confirm the skill applies.
2. Read memory and failure state
3. Compute calibration factor
4. Update memory context

# Decision Points
- whether calibration should alter behavior

# Fallback Strategy
- Use conservative confidence when evidence is limited

# Tool Contract
## Available Tools
- None

## Required Input Contexts
- memory_context
- failure_context

## Produced Output Contexts
- memory_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- calibration factor

# Completion Criteria
- Calibration factor is explicit

# Escalation Rules
- Escalate when confidence cannot be estimated credibly

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
