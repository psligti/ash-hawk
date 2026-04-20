---
id: skill.report_generation
name: report-generation
kind: skill
version: 1.0.0
status: active
summary: Produces operator-facing summaries and diff reports.
goal: Turn audit and evaluation state into a clear report.
file: skills/report-generation.md
category: reporting
scope: narrow
when_to_use:
- When a summary or diff report is needed
- When reporting completes the run
when_not_to_use:
- When evidence is too incomplete to summarize credibly
triggers:
- report writing
- diff summary
anti_triggers:
- no audit or evaluation context
prerequisites:
- audit_context is available
- evaluation_context is available
inputs_expected:
- audit state
- evaluation state
procedure:
- Write run summary
- Build diff report
- Update audit context
decision_points:
- what level of reporting detail is appropriate
fallback_strategy:
- Produce the minimal trustworthy summary first
outputs:
  required_elements:
  - run summary
  - diff report
completion_criteria:
- Report outputs are explicit and current
escalation_rules:
- Escalate when evidence is too weak for a trustworthy summary
dependencies:
  tools: &id001 []
  related_skills: []
  related_agents:
  - reviewer
examples:
- description: Generate a final summary report
  input: 'The run has enough evidence to produce a final summary and diff report.

    '
  expected_behavior:
  - writes a summary
  - produces a diff report
tool_names: *id001
input_contexts:
- audit_context
- evaluation_context
output_contexts:
- audit_context
memory_read_scopes: []
memory_write_scopes: []
description: Produces operator-facing summaries and diff reports.
---
# Purpose
Turn audit and evaluation state into a clear report.

# Use This Skill When
- When a summary or diff report is needed
- When reporting completes the run

# Do Not Use This Skill When
- When evidence is too incomplete to summarize credibly

# Triggers
- report writing
- diff summary

# Anti-Triggers
- no audit or evaluation context

# Prerequisites
- audit_context is available
- evaluation_context is available

# Inputs Expected
- audit state
- evaluation state

# Procedure
1. Confirm the skill applies.
2. Write run summary
3. Build diff report
4. Update audit context

# Decision Points
- what level of reporting detail is appropriate

# Fallback Strategy
- Produce the minimal trustworthy summary first

# Tool Contract
## Available Tools
- None

## Required Input Contexts
- audit_context
- evaluation_context

## Produced Output Contexts
- audit_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- run summary
- diff report

# Completion Criteria
- Report outputs are explicit and current

# Escalation Rules
- Escalate when evidence is too weak for a trustworthy summary

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
