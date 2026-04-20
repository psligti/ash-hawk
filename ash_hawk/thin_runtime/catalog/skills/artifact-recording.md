---
id: skill.artifact_recording
name: artifact-recording
kind: skill
version: 1.0.0
status: active
summary: Persists artifacts, events, and logs for the run.
goal: Record durable run artifacts at meaningful checkpoints.
file: skills/artifact-recording.md
category: reporting
scope: narrow
when_to_use:
- When the run reaches a meaningful checkpoint
- When audit artifacts should be persisted
when_not_to_use:
- When there is nothing meaningful to record yet
triggers:
- artifact write
- checkpoint logging
anti_triggers:
- no meaningful state change
prerequisites:
- audit_context is available
- runtime_context is available
inputs_expected:
- audit state
- runtime state
procedure:
- Record artifacts
- Write iteration log
- Write run summary
- Record event
decision_points:
- whether the checkpoint is meaningful enough to persist
fallback_strategy:
- Persist only the minimum useful artifact set when uncertain
outputs:
  required_elements:
  - artifact and event outputs
completion_criteria:
- Artifacts are written explicitly
escalation_rules:
- Escalate when artifact persistence repeatedly fails
dependencies:
  tools: &id001
  - todoread
  - todowrite
  related_skills: []
  related_agents:
  - coordinator
  - reviewer
examples:
- description: Record run artifacts at a checkpoint
  input: 'The runtime reached a checkpoint and should persist the audit state.

    '
  expected_behavior:
  - records artifacts
  - writes logs and summaries
tool_names: *id001
input_contexts:
- audit_context
- runtime_context
output_contexts:
- audit_context
memory_read_scopes: []
memory_write_scopes:
- artifact_memory
description: Persists artifacts, events, and logs for the run.
---
# Purpose
Record durable run artifacts at meaningful checkpoints.

# Use This Skill When
- When the run reaches a meaningful checkpoint
- When audit artifacts should be persisted

# Do Not Use This Skill When
- When there is nothing meaningful to record yet

# Triggers
- artifact write
- checkpoint logging

# Anti-Triggers
- no meaningful state change

# Prerequisites
- audit_context is available
- runtime_context is available

# Inputs Expected
- audit state
- runtime state

# Procedure
1. Confirm the skill applies.
2. Record artifacts
3. Write iteration log
4. Write run summary
5. Record event

# Decision Points
- whether the checkpoint is meaningful enough to persist

# Fallback Strategy
- Persist only the minimum useful artifact set when uncertain

# Tool Contract
## Available Tools
- todoread
- todowrite

## Required Input Contexts
- audit_context
- runtime_context

## Produced Output Contexts
- audit_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- artifact_memory

# Output Contract
## Required Elements
- artifact and event outputs

# Completion Criteria
- Artifacts are written explicitly

# Escalation Rules
- Escalate when artifact persistence repeatedly fails

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
