---
id: skill.triage
name: triage
kind: skill
version: 1.0.0
status: active
summary: Buckets failures and suspicious traces into actionable groupings.
goal: Identify the current failure shape and reduce ambiguity for the next step.
file: skills/triage.md
category: reasoning
scope: narrow
when_to_use:
- When failure context exists
- When suspicious behavior should be bucketed
when_not_to_use:
- When no failure or audit context exists
triggers:
- failure investigation
- suspicious run analysis
anti_triggers:
- pure execution tasks
prerequisites:
- failure_context is available
- audit_context is available
inputs_expected:
- failure state
- audit trace
procedure:
- Bucket failures
- Review suspicious traces
- Update failure context
decision_points:
- bucket choice
- suspicious vs normal
fallback_strategy:
- Use an unknown bucket rather than overfit
outputs:
  required_elements:
  - failure bucket
  - suspicious review state
completion_criteria:
- Failure is bucketed explicitly
escalation_rules:
- Escalate when the trace is too ambiguous to bucket credibly
dependencies:
  tools: &id001
  - call_llm_structured
  related_skills: []
  related_agents:
  - researcher
examples:
- description: Bucket a suspicious failed run
  input: 'A run failed and the trace may be suspicious.

    '
  expected_behavior:
  - buckets the failure
  - records suspicious review state
tool_names: *id001
input_contexts:
- failure_context
- audit_context
output_contexts:
- failure_context
memory_read_scopes: []
memory_write_scopes: []
description: Buckets failures and suspicious traces into actionable groupings.
---
# Purpose
Identify the current failure shape and reduce ambiguity for the next step.

# Use This Skill When
- When failure context exists
- When suspicious behavior should be bucketed

# Do Not Use This Skill When
- When no failure or audit context exists

# Triggers
- failure investigation
- suspicious run analysis

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
2. Bucket failures
3. Review suspicious traces
4. Update failure context

# Decision Points
- bucket choice
- suspicious vs normal

# Fallback Strategy
- Use an unknown bucket rather than overfit

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
- failure bucket
- suspicious review state

# Completion Criteria
- Failure is bucketed explicitly

# Escalation Rules
- Escalate when the trace is too ambiguous to bucket credibly

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
