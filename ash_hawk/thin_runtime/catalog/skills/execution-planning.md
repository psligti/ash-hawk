---
id: skill.execution_planning
name: execution-planning
kind: skill
version: 1.0.0
status: active
summary: Converts concepts and failures into concrete execution steps.
goal: Produce a deterministic next action from current goal and failure context.
file: skills/execution-planning.md
category: reasoning
scope: narrow
when_to_use:
- When the runtime has candidate concepts and must act
- When failure context should shape the next execution step
- When a concrete action plan is needed
when_not_to_use:
- When failure context is absent
- When only reporting remains
- When verification rather than execution is needed
triggers:
- Need to turn concepts into action
- Need to choose a plan from current failure state
anti_triggers:
- No actionable concept exists
- Another skill already determines the exact next tool
prerequisites:
- goal_context is available
- failure_context is available
inputs_expected:
- goal description
- current failure or concept state
procedure:
- Read goal and failure context
- Choose the most appropriate next action
- Express the plan through deterministic tools
decision_points:
- Whether the next move is execution or further analysis
fallback_strategy:
- Prefer the simplest actionable step
- Escalate if failure context is too weak
outputs:
  required_elements:
  - selected action
  - runtime progression signal
completion_criteria:
- Chosen action is concrete and valid
- Output fits current failure state
escalation_rules:
- Escalate when concepts are too weak or conflicting
dependencies:
  tools: &id001 []
  related_skills: []
  related_agents:
  - coordinator
  - executor
examples:
- description: Choose the next execution move from a failure concept
  input: 'A concept exists and the runtime needs a concrete next action.

    '
  expected_behavior:
  - selects the action
  - expresses it as an execution step
tool_names: *id001
input_contexts:
- goal_context
- failure_context
output_contexts:
- runtime_context
memory_read_scopes: []
memory_write_scopes: []
description: Converts concepts and failures into concrete execution steps.
---
# Purpose
Produce a deterministic next action from current goal and failure context.

# Use This Skill When
- When the runtime has candidate concepts and must act
- When failure context should shape the next execution step
- When a concrete action plan is needed

# Do Not Use This Skill When
- When failure context is absent
- When only reporting remains
- When verification rather than execution is needed

# Triggers
- Need to turn concepts into action
- Need to choose a plan from current failure state

# Anti-Triggers
- No actionable concept exists
- Another skill already determines the exact next tool

# Prerequisites
- goal_context is available
- failure_context is available

# Inputs Expected
- goal description
- current failure or concept state

# Procedure
1. Confirm the skill applies.
2. Read goal and failure context
3. Choose the most appropriate next action
4. Express the plan through deterministic tools

# Decision Points
- Whether the next move is execution or further analysis

# Fallback Strategy
- Prefer the simplest actionable step
- Escalate if failure context is too weak

# Tool Contract
## Available Tools
- None

## Required Input Contexts
- goal_context
- failure_context

## Produced Output Contexts
- runtime_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- selected action
- runtime progression signal

# Completion Criteria
- Chosen action is concrete and valid
- Output fits current failure state

# Escalation Rules
- Escalate when concepts are too weak or conflicting

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
