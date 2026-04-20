---
id: skill.regression_detection
name: regression-detection
kind: skill
version: 1.0.0
status: active
summary: Detects regressions on unaffected areas.
goal: Identify regressions before final acceptance.
file: skills/regression-detection.md
category: verification
scope: narrow
when_to_use:
- When evaluation context exists
- When regression risk must be checked
when_not_to_use:
- When no evaluation evidence exists
triggers:
- regression check
anti_triggers:
- missing evaluation context
prerequisites:
- evaluation_context is available
inputs_expected:
- evaluation state
procedure:
- Run regression detection
- Update evaluation context
decision_points:
- regression found vs no regression
fallback_strategy:
- Escalate when regression signal is unclear
outputs:
  required_elements:
  - regression state
completion_criteria:
- Regression state is explicit
escalation_rules:
- Escalate when regressions cannot be assessed credibly
dependencies:
  tools: &id001
  - detect_regressions
  related_skills: []
  related_agents:
  - verifier
examples:
- description: Detect regressions before acceptance
  input: 'The run needs to know whether any regressions were introduced.

    '
  expected_behavior:
  - detects regressions
  - updates evaluation context
tool_names: *id001
input_contexts:
- evaluation_context
output_contexts:
- evaluation_context
memory_read_scopes: []
memory_write_scopes: []
description: Detects regressions on unaffected areas.
---
# Purpose
Identify regressions before final acceptance.

# Use This Skill When
- When evaluation context exists
- When regression risk must be checked

# Do Not Use This Skill When
- When no evaluation evidence exists

# Triggers
- regression check

# Anti-Triggers
- missing evaluation context

# Prerequisites
- evaluation_context is available

# Inputs Expected
- evaluation state

# Procedure
1. Confirm the skill applies.
2. Run regression detection
3. Update evaluation context

# Decision Points
- regression found vs no regression

# Fallback Strategy
- Escalate when regression signal is unclear

# Tool Contract
## Available Tools
- detect_regressions

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
- regression state

# Completion Criteria
- Regression state is explicit

# Escalation Rules
- Escalate when regressions cannot be assessed credibly

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
