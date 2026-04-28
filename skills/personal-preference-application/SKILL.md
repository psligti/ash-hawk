---
name: personal-preference-application
description: Applies operator preferences to the current run.
version: 1.0.0
metadata:
  id: skill.personal_preference_application
  name: personal-preference-application
  kind: skill
  version: 1.0.0
  status: active
  summary: Applies operator preferences to the current run.
  goal: Load and apply personal preferences so runtime choices match operator intent.
  file: skills/personal-preference-application.md
  category: reasoning
  scope: narrow
  when_to_use:
  - When operator preferences should shape the run
  - When personal memory is relevant
  when_not_to_use:
  - When preferences do not materially affect the task
  triggers:
  - preference load
  anti_triggers:
  - no personal preference relevance
  prerequisites:
  - memory_context is available
  - goal_context is available
  inputs_expected:
  - goal
  - personal memory
  procedure:
  - Load personal preferences
  - Apply them to tool or runtime context
  decision_points:
  - whether preferences affect the current choice
  fallback_strategy:
  - Ignore preferences that are not relevant to the current goal
  outputs:
    required_elements:
    - tool context updates
  completion_criteria:
  - Relevant preferences are surfaced explicitly
  escalation_rules:
  - Escalate when preferences conflict with stronger instructions
  dependencies:
    tools: &id001 []
    related_skills: []
    related_agents:
    - memory_manager
  examples:
  - description: Apply operator style preferences
    input: 'The runtime should incorporate known operator preferences before acting.

      '
    expected_behavior:
    - loads and applies preferences
  tool_names: *id001
  input_contexts:
  - memory_context
  - goal_context
  output_contexts:
  - tool_context
  memory_read_scopes: []
  memory_write_scopes: []
  description: Applies operator preferences to the current run.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/personal-preference-application.md
---

# Purpose
Load and apply personal preferences so runtime choices match operator intent.

# Use This Skill When
- When operator preferences should shape the run
- When personal memory is relevant

# Do Not Use This Skill When
- When preferences do not materially affect the task

# Triggers
- preference load

# Anti-Triggers
- no personal preference relevance

# Prerequisites
- memory_context is available
- goal_context is available

# Inputs Expected
- goal
- personal memory

# Procedure
1. Confirm the skill applies.
2. Load personal preferences
3. Apply them to tool or runtime context

# Decision Points
- whether preferences affect the current choice

# Fallback Strategy
- Ignore preferences that are not relevant to the current goal

# Tool Contract
## Available Tools
- None

## Required Input Contexts
- memory_context
- goal_context

## Produced Output Contexts
- tool_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- tool context updates

# Completion Criteria
- Relevant preferences are surfaced explicitly

# Escalation Rules
- Escalate when preferences conflict with stronger instructions

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
