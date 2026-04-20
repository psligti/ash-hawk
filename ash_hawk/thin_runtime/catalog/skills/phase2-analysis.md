---
id: skill.phase2_analysis
name: phase2-analysis
kind: skill
version: 1.0.0
status: active
summary: Computes higher-level quality metrics over the run.
goal: Produce phase2-style metrics from audit and evaluation evidence.
file: skills/phase2-analysis.md
category: reporting
scope: narrow
when_to_use:
- When higher-level metrics are needed
- When audit and evaluation context exist
when_not_to_use:
- When basic reporting is sufficient and phase2 metrics add no value
triggers:
- quality metric computation
anti_triggers:
- missing audit or evaluation context
prerequisites:
- audit_context is available
- evaluation_context is available
inputs_expected:
- audit state
- evaluation state
procedure:
- Compute phase2 metrics
- Update audit/evaluation context
decision_points:
- whether phase2 metrics are necessary
fallback_strategy:
- Skip phase2 metrics when evidence is insufficient
outputs:
  required_elements:
  - phase2 metrics
completion_criteria:
- Phase2 metrics are explicit
escalation_rules:
- Escalate when metrics cannot be computed credibly
dependencies:
  tools: &id001 []
  related_skills: []
  related_agents:
  - reviewer
examples:
- description: Compute phase2 metrics
  input: 'The run needs higher-level quality metrics before final review.

    '
  expected_behavior:
  - computes phase2 metrics
tool_names: *id001
input_contexts:
- audit_context
- evaluation_context
output_contexts:
- audit_context
memory_read_scopes: []
memory_write_scopes: []
description: Computes higher-level quality metrics over the run.
---
# Purpose
Produce phase2-style metrics from audit and evaluation evidence.

# Use This Skill When
- When higher-level metrics are needed
- When audit and evaluation context exist

# Do Not Use This Skill When
- When basic reporting is sufficient and phase2 metrics add no value

# Triggers
- quality metric computation

# Anti-Triggers
- missing audit or evaluation context

# Prerequisites
- audit_context is available
- evaluation_context is available

# Inputs Expected
- audit state
- evaluation state

# Procedure
1. Confirm the skill applies.
2. Compute phase2 metrics
3. Update audit/evaluation context

# Decision Points
- whether phase2 metrics are necessary

# Fallback Strategy
- Skip phase2 metrics when evidence is insufficient

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
- phase2 metrics

# Completion Criteria
- Phase2 metrics are explicit

# Escalation Rules
- Escalate when metrics cannot be computed credibly

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
