---
id: skill.stop_conditions
name: stop-conditions
kind: skill
version: 1.0.0
status: active
summary: Determines whether the run should continue, stop, or escalate.
goal: Apply explicit stop, retry, continue, or escalate reasoning to runtime state.
file: skills/stop-conditions.md
category: reasoning
scope: narrow
when_to_use:
- When the runtime may need to stop
- When iteration or success boundaries should be checked
- When escalation may be required
when_not_to_use:
- When the next action is already clear and safe
- When runtime and evaluation context are absent
triggers:
- max iteration check
- possible completion
- possible escalation
anti_triggers:
- no meaningful runtime state to inspect
prerequisites:
- runtime_context is available
- evaluation_context is available when relevant
inputs_expected:
- runtime counters
- evaluation or outcome state
procedure:
- Inspect runtime and evaluation signals
- Check success and stop boundaries
- Return stop or continue clearly
decision_points:
- continue vs stop
- stop vs escalate
fallback_strategy:
- Stop conservatively when safety is unclear
outputs:
  required_elements:
  - explicit stop or continue signal
completion_criteria:
- Stop reasoning is explicit and justified
escalation_rules:
- Escalate when stop reason is unclear but risk is rising
dependencies:
  tools: &id001 []
  related_skills: []
  related_agents:
  - coordinator
examples:
- description: Decide whether the run should continue
  input: 'Iteration limits and evaluation state suggest a possible stop.

    '
  expected_behavior:
  - checks the stop conditions
  - returns an explicit stop or continue signal
tool_names: *id001
input_contexts:
- runtime_context
- evaluation_context
output_contexts:
- runtime_context
memory_read_scopes: []
memory_write_scopes: []
description: Determines whether the run should continue, stop, or escalate.
---
# Purpose
Apply explicit stop, retry, continue, or escalate reasoning to runtime state.

# Use This Skill When
- When the runtime may need to stop
- When iteration or success boundaries should be checked
- When escalation may be required

# Do Not Use This Skill When
- When the next action is already clear and safe
- When runtime and evaluation context are absent

# Triggers
- max iteration check
- possible completion
- possible escalation

# Anti-Triggers
- no meaningful runtime state to inspect

# Prerequisites
- runtime_context is available
- evaluation_context is available when relevant

# Inputs Expected
- runtime counters
- evaluation or outcome state

# Procedure
1. Confirm the skill applies.
2. Inspect runtime and evaluation signals
3. Check success and stop boundaries
4. Return stop or continue clearly

# Decision Points
- continue vs stop
- stop vs escalate

# Fallback Strategy
- Stop conservatively when safety is unclear

# Tool Contract
## Available Tools
- None

## Required Input Contexts
- runtime_context
- evaluation_context

## Produced Output Contexts
- runtime_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- explicit stop or continue signal

# Completion Criteria
- Stop reasoning is explicit and justified

# Escalation Rules
- Escalate when stop reason is unclear but risk is rising

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
