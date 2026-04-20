---
id: skill.process_control
name: process-control
kind: skill
version: 1.0.0
status: active
summary: Governs sequencing, action choice, and stop checks.
goal: Help the agent decide the next valid action and determine whether the run should
  continue.
file: skills/process-control.md
category: reasoning
scope: narrow
when_to_use:
- When the runtime needs the next justified action
- When stop conditions should be checked
- When sequencing discipline matters
when_not_to_use:
- When a specialized execution or verification skill is the direct need
- When no runtime context is available
- When a more specific skill fully determines the next step
triggers:
- Need to pick the next tool
- Need to check whether the run should stop
- Need to maintain ordered progress
anti_triggers:
- Task is trivial and already complete
- Required runtime context is missing
- Another skill has a narrower, sufficient procedure
prerequisites:
- goal_context is available
- runtime_context is available
inputs_expected:
- goal description
- current runtime state
- available tools and context
procedure:
- Inspect goal and runtime state
- Choose the next valid action
- Check whether stop conditions are met
- Return the chosen direction clearly
decision_points:
- Whether to act now or stop
- Whether the next tool should be selected
fallback_strategy:
- Use the simplest valid action when uncertainty remains
- Escalate if no safe next action exists
outputs:
  required_elements:
  - next action signal
  - stop signal when relevant
completion_criteria:
- Next action is explicit
- Stop reasoning is explicit when stopping
- Output is consistent with current runtime state
escalation_rules:
- Escalate when no safe next action exists
- Escalate when stop reasoning is ambiguous
dependencies:
  tools: &id001 []
  related_skills: []
  related_agents:
  - coordinator
examples:
- description: Pick the next action and check whether to stop
  input: 'The run has partial progress and needs the next valid step.

    '
  expected_behavior:
  - selects a valid next action
  - checks whether the run should stop
tool_names: *id001
input_contexts:
- goal_context
- runtime_context
output_contexts:
- runtime_context
memory_read_scopes: []
memory_write_scopes: []
description: Governs sequencing, action choice, and stop checks.
---
# Purpose
Help the agent decide the next valid action and determine whether the run should continue.

# Use This Skill When
- When the runtime needs the next justified action
- When stop conditions should be checked
- When sequencing discipline matters

# Do Not Use This Skill When
- When a specialized execution or verification skill is the direct need
- When no runtime context is available
- When a more specific skill fully determines the next step

# Triggers
- Need to pick the next tool
- Need to check whether the run should stop
- Need to maintain ordered progress

# Anti-Triggers
- Task is trivial and already complete
- Required runtime context is missing
- Another skill has a narrower, sufficient procedure

# Prerequisites
- goal_context is available
- runtime_context is available

# Inputs Expected
- goal description
- current runtime state
- available tools and context

# Procedure
1. Confirm the skill applies.
2. Inspect goal and runtime state
3. Choose the next valid action
4. Check whether stop conditions are met
5. Return the chosen direction clearly

# Decision Points
- Whether to act now or stop
- Whether the next tool should be selected

# Fallback Strategy
- Use the simplest valid action when uncertainty remains
- Escalate if no safe next action exists

# Tool Contract
## Available Tools
- None

## Required Input Contexts
- goal_context
- runtime_context

## Produced Output Contexts
- runtime_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- next action signal
- stop signal when relevant

# Completion Criteria
- Next action is explicit
- Stop reasoning is explicit when stopping
- Output is consistent with current runtime state

# Escalation Rules
- Escalate when no safe next action exists
- Escalate when stop reasoning is ambiguous

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
