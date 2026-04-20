---
id: skill.phase2_gate
name: phase2-gate
kind: skill
version: 1.0.0
status: active
summary: Evaluates a higher-level quality gate over current and historical evidence.
goal: Decide whether the run passes the phase2 quality gate.
file: skills/phase2-gate.md
category: reporting
scope: narrow
when_to_use:
- When final quality gating is needed
- When audit and memory context should be combined
when_not_to_use:
- When the run is too incomplete for gate evaluation
triggers:
- quality gate decision
anti_triggers:
- missing audit or memory context
prerequisites:
- audit_context is available
- memory_context is available
inputs_expected:
- audit state
- memory state
procedure:
- Evaluate phase2 gate
- Update audit/evaluation context
decision_points:
- gate pass vs fail
fallback_strategy:
- Escalate when the gate cannot be evaluated responsibly
outputs:
  required_elements:
  - phase2 gate result
completion_criteria:
- Gate result is explicit
escalation_rules:
- Escalate when gate outcome is too uncertain
dependencies:
  tools: &id001 []
  related_skills:
  - phase2-analysis
  related_agents:
  - reviewer
examples:
- description: Decide whether the run passes the quality gate
  input: 'The runtime has enough evidence to evaluate the phase2 gate.

    '
  expected_behavior:
  - evaluates the gate
  - updates context
tool_names: *id001
input_contexts:
- audit_context
- memory_context
output_contexts:
- audit_context
memory_read_scopes: []
memory_write_scopes: []
description: Evaluates a higher-level quality gate over current and historical evidence.
---
# Purpose
Decide whether the run passes the phase2 quality gate.

# Use This Skill When
- When final quality gating is needed
- When audit and memory context should be combined

# Do Not Use This Skill When
- When the run is too incomplete for gate evaluation

# Triggers
- quality gate decision

# Anti-Triggers
- missing audit or memory context

# Prerequisites
- audit_context is available
- memory_context is available

# Inputs Expected
- audit state
- memory state

# Procedure
1. Confirm the skill applies.
2. Evaluate phase2 gate
3. Update audit/evaluation context

# Decision Points
- gate pass vs fail

# Fallback Strategy
- Escalate when the gate cannot be evaluated responsibly

# Tool Contract
## Available Tools
- None

## Required Input Contexts
- audit_context
- memory_context

## Produced Output Contexts
- audit_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- phase2 gate result

# Completion Criteria
- Gate result is explicit

# Escalation Rules
- Escalate when gate outcome is too uncertain

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
