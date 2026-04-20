---
id: skill.lesson_consolidation
name: lesson-consolidation
kind: skill
version: 1.0.0
status: active
summary: Converts current outcomes into durable lessons.
goal: Turn evidence-backed outcomes into semantic lessons and artifact records.
file: skills/lesson-consolidation.md
category: reasoning
scope: narrow
when_to_use:
- When a run produced reusable learning
- When durable lesson recording is appropriate
when_not_to_use:
- When evidence is too weak to justify a lesson
triggers:
- lesson capture
- semantic update
anti_triggers:
- weak evidence
- purely transient findings
prerequisites:
- memory_context is available
- evaluation_context is available
inputs_expected:
- memory state
- evaluation evidence
procedure:
- Save lesson
- Consolidate lessons
- Update semantic and artifact memory
decision_points:
- whether a finding is durable enough for consolidation
fallback_strategy:
- Keep information session-local when durability is unclear
outputs:
  required_elements:
  - lesson state
  - semantic update
completion_criteria:
- Lesson recording is explicit and evidence-backed
escalation_rules:
- Escalate when lesson quality is too weak for semantic promotion
dependencies:
  tools: &id001 []
  related_skills: []
  related_agents:
  - memory_manager
  - reviewer
examples:
- description: Consolidate a proven lesson
  input: 'The run produced a reusable and verified lesson.

    '
  expected_behavior:
  - saves the lesson
  - updates durable memory
tool_names: *id001
input_contexts:
- memory_context
- evaluation_context
output_contexts:
- memory_context
memory_read_scopes: []
memory_write_scopes:
- semantic_memory
- artifact_memory
description: Converts current outcomes into durable lessons.
---
# Purpose
Turn evidence-backed outcomes into semantic lessons and artifact records.

# Use This Skill When
- When a run produced reusable learning
- When durable lesson recording is appropriate

# Do Not Use This Skill When
- When evidence is too weak to justify a lesson

# Triggers
- lesson capture
- semantic update

# Anti-Triggers
- weak evidence
- purely transient findings

# Prerequisites
- memory_context is available
- evaluation_context is available

# Inputs Expected
- memory state
- evaluation evidence

# Procedure
1. Confirm the skill applies.
2. Save lesson
3. Consolidate lessons
4. Update semantic and artifact memory

# Decision Points
- whether a finding is durable enough for consolidation

# Fallback Strategy
- Keep information session-local when durability is unclear

# Tool Contract
## Available Tools
- None

## Required Input Contexts
- memory_context
- evaluation_context

## Produced Output Contexts
- memory_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- semantic_memory
- artifact_memory

# Output Contract
## Required Elements
- lesson state
- semantic update

# Completion Criteria
- Lesson recording is explicit and evidence-backed

# Escalation Rules
- Escalate when lesson quality is too weak for semantic promotion

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
