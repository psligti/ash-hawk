---
id: skill.retry_strategy
name: retry-strategy
kind: skill
version: 1.0.0
status: active
summary: Retries transient failures with bounded policy.
goal: Decide whether and how to retry a transient action safely.
file: skills/retry-strategy.md
category: reasoning
scope: narrow
when_to_use:
- When a transient error may be recoverable
- When retry policy can add signal
when_not_to_use:
- When an error is clearly non-retryable
- When retries have stopped adding value
triggers:
- transient failure
- retryable tool error
anti_triggers:
- invalid input
- authorization failure
prerequisites:
- runtime_context is available
- audit_context is available
inputs_expected:
- retry count
- failure classification
procedure:
- Inspect retryability
- Retry with bounded policy
- Update runtime state
decision_points:
- retry vs stop
fallback_strategy:
- Escalate when retries are no longer useful
outputs:
  required_elements:
  - updated retry state
completion_criteria:
- Retry decision is explicit and bounded
escalation_rules:
- Escalate when repeated retries fail the same way
dependencies:
  tools: &id001 []
  related_skills: []
  related_agents:
  - coordinator
  - executor
examples:
- description: Retry a transient tool failure
  input: 'A tool timed out and may succeed on retry.

    '
  expected_behavior:
  - retries through policy
  - updates retry count
tool_names: *id001
input_contexts:
- runtime_context
- audit_context
output_contexts:
- runtime_context
memory_read_scopes: []
memory_write_scopes: []
description: Retries transient failures with bounded policy.
---
# Purpose
Decide whether and how to retry a transient action safely.

# Use This Skill When
- When a transient error may be recoverable
- When retry policy can add signal

# Do Not Use This Skill When
- When an error is clearly non-retryable
- When retries have stopped adding value

# Triggers
- transient failure
- retryable tool error

# Anti-Triggers
- invalid input
- authorization failure

# Prerequisites
- runtime_context is available
- audit_context is available

# Inputs Expected
- retry count
- failure classification

# Procedure
1. Confirm the skill applies.
2. Inspect retryability
3. Retry with bounded policy
4. Update runtime state

# Decision Points
- retry vs stop

# Fallback Strategy
- Escalate when retries are no longer useful

# Tool Contract
## Available Tools
- None

## Required Input Contexts
- runtime_context
- audit_context

## Produced Output Contexts
- runtime_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- updated retry state

# Completion Criteria
- Retry decision is explicit and bounded

# Escalation Rules
- Escalate when repeated retries fail the same way

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
