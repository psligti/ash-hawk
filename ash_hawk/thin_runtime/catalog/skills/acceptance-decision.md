---
id: skill.acceptance_decision
name: acceptance-decision
kind: skill
version: 1.0.0
status: active
summary: Decides whether current results should be accepted.
goal: Compute keep or revert decisions from evaluation and regression evidence.
file: skills/acceptance-decision.md
category: verification
scope: narrow
when_to_use:
- When evaluation evidence exists
- When acceptance must be explicit
when_not_to_use:
- When baseline or verification is still missing
triggers:
- acceptance decision
- regression review
anti_triggers:
- insufficient evaluation evidence
prerequisites:
- evaluation_context is available
- workspace_context is available
inputs_expected:
- evaluation evidence
- workspace context
procedure:
- Detect regressions
- Compute acceptance
- Record the acceptance outcome
decision_points:
- accept vs reject
fallback_strategy:
- Escalate when acceptance cannot be supported credibly
outputs:
  required_elements:
  - acceptance decision
  - regression state
completion_criteria:
- Acceptance decision is explicit
- Regression state is explicit
escalation_rules:
- Escalate when evidence is too weak for acceptance
dependencies:
  tools: &id001
  - detect_regressions
  related_skills:
  - verification
  related_agents:
  - verifier
examples:
- description: Decide whether to accept a candidate
  input: 'The run has verification evidence and must decide keep or revert.

    '
  expected_behavior:
  - detects regressions
  - computes acceptance
tool_names: *id001
input_contexts:
- evaluation_context
- workspace_context
output_contexts:
- evaluation_context
memory_read_scopes: []
memory_write_scopes: []
description: Decides whether current results should be accepted.
---
# Purpose
Compute keep or revert decisions from evaluation and regression evidence.

# Use This Skill When
- When evaluation evidence exists
- When acceptance must be explicit

# Do Not Use This Skill When
- When baseline or verification is still missing

# Triggers
- acceptance decision
- regression review

# Anti-Triggers
- insufficient evaluation evidence

# Prerequisites
- evaluation_context is available
- workspace_context is available

# Inputs Expected
- evaluation evidence
- workspace context

# Procedure
1. Confirm the skill applies.
2. Detect regressions
3. Compute acceptance
4. Record the acceptance outcome

# Decision Points
- accept vs reject

# Fallback Strategy
- Escalate when acceptance cannot be supported credibly

# Tool Contract
## Available Tools
- detect_regressions

## Required Input Contexts
- evaluation_context
- workspace_context

## Produced Output Contexts
- evaluation_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- acceptance decision
- regression state

# Completion Criteria
- Acceptance decision is explicit
- Regression state is explicit

# Escalation Rules
- Escalate when evidence is too weak for acceptance

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
