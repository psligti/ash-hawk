---
id: skill.context_assembly
name: context-assembly
kind: skill
version: 1.0.0
status: active
summary: Assembles the runtime context bundle.
goal: Build a complete current context across goal, runtime, workspace, evaluation,
  failure, memory, tool, and audit sections.
file: skills/context-assembly.md
category: reasoning
scope: narrow
when_to_use:
- When the runtime needs a fresh context snapshot
- When workspace state should be loaded
when_not_to_use:
- When current context is already valid and fresh
triggers:
- context bootstrap
- workspace load
anti_triggers:
- no available goal or memory context
prerequisites:
- goal_context is available
- memory_context is available
- workspace_context is available
inputs_expected:
- goal state
- memory state
- workspace state
procedure:
- Assemble context
- Load workspace state
- Publish updated context sections
decision_points:
- whether context refresh is necessary
fallback_strategy:
- Refresh only the missing sections if full assembly is unnecessary
outputs:
  required_elements:
  - assembled context sections
completion_criteria:
- Context sections are explicit and current
escalation_rules:
- Escalate when required context sections cannot be assembled
dependencies:
  tools: &id001
  - load_workspace_state
  related_skills: []
  related_agents:
  - executor
  - coordinator
examples:
- description: Assemble the current runtime context
  input: 'The runtime needs a fresh context snapshot before choosing the next action.

    '
  expected_behavior:
  - assembles context
  - loads workspace state
tool_names: *id001
input_contexts:
- goal_context
- memory_context
- workspace_context
output_contexts:
- goal_context
- runtime_context
- workspace_context
- evaluation_context
- failure_context
- memory_context
- tool_context
- audit_context
memory_read_scopes: []
memory_write_scopes: []
description: Assembles the runtime context bundle.
---
# Purpose
Build a complete current context across goal, runtime, workspace, evaluation, failure, memory, tool, and audit sections.

# Use This Skill When
- When the runtime needs a fresh context snapshot
- When workspace state should be loaded

# Do Not Use This Skill When
- When current context is already valid and fresh

# Triggers
- context bootstrap
- workspace load

# Anti-Triggers
- no available goal or memory context

# Prerequisites
- goal_context is available
- memory_context is available
- workspace_context is available

# Inputs Expected
- goal state
- memory state
- workspace state

# Procedure
1. Confirm the skill applies.
2. Assemble context
3. Load workspace state
4. Publish updated context sections

# Decision Points
- whether context refresh is necessary

# Fallback Strategy
- Refresh only the missing sections if full assembly is unnecessary

# Tool Contract
## Available Tools
- load_workspace_state

## Required Input Contexts
- goal_context
- memory_context
- workspace_context

## Produced Output Contexts
- goal_context
- runtime_context
- workspace_context
- evaluation_context
- failure_context
- memory_context
- tool_context
- audit_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- assembled context sections

# Completion Criteria
- Context sections are explicit and current

# Escalation Rules
- Escalate when required context sections cannot be assembled

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
