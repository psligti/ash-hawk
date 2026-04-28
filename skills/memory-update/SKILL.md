---
name: memory-update
description: Writes runtime memory safely and explicitly.
version: 1.0.0
metadata:
  id: skill.memory_update
  name: memory-update
  kind: skill
  version: 1.0.0
  status: active
  summary: Writes runtime memory safely and explicitly.
  goal: Update working, session, episodic, and artifact memory within allowed scopes.
  file: skills/memory-update.md
  category: reasoning
  scope: narrow
  when_to_use:
  - When new memory should be recorded
  - When runtime findings should be preserved
  when_not_to_use:
  - When evidence is too weak for a memory write
  triggers:
  - new finding
  - episode capture
  - snapshot need
  anti_triggers:
  - no meaningful new information
  prerequisites:
  - memory_context is available
  - audit_context is available
  inputs_expected:
  - memory state
  - audit state
  procedure:
  - Write memory entry
  - Record episode
  - Snapshot or resume run state
  - Record artifacts
  decision_points:
  - which memory scope to update
  fallback_strategy:
  - Prefer narrower memory scopes when uncertainty remains
  outputs:
    required_elements:
    - updated memory state
  completion_criteria:
  - Memory updates are explicit and scoped correctly
  escalation_rules:
  - Escalate when a desired write exceeds scope permissions
  dependencies:
    tools: &id001 []
    related_skills: []
    related_agents:
    - memory_manager
  examples:
  - description: Write a new episode and snapshot state
    input: 'The runtime discovered a new finding worth preserving.

      '
    expected_behavior:
    - writes memory
    - records episode
    - snapshots state
  tool_names: *id001
  input_contexts:
  - memory_context
  - audit_context
  output_contexts:
  - memory_context
  memory_read_scopes: []
  memory_write_scopes:
  - working_memory
  - session_memory
  - episodic_memory
  - artifact_memory
  description: Writes runtime memory safely and explicitly.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/memory-update.md
---

# Purpose
Update working, session, episodic, and artifact memory within allowed scopes.

# Use This Skill When
- When new memory should be recorded
- When runtime findings should be preserved

# Do Not Use This Skill When
- When evidence is too weak for a memory write

# Triggers
- new finding
- episode capture
- snapshot need

# Anti-Triggers
- no meaningful new information

# Prerequisites
- memory_context is available
- audit_context is available

# Inputs Expected
- memory state
- audit state

# Procedure
1. Confirm the skill applies.
2. Write memory entry
3. Record episode
4. Snapshot or resume run state
5. Record artifacts

# Decision Points
- which memory scope to update

# Fallback Strategy
- Prefer narrower memory scopes when uncertainty remains

# Tool Contract
## Available Tools
- None

## Required Input Contexts
- memory_context
- audit_context

## Produced Output Contexts
- memory_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- working_memory
- session_memory
- episodic_memory
- artifact_memory

# Output Contract
## Required Elements
- updated memory state

# Completion Criteria
- Memory updates are explicit and scoped correctly

# Escalation Rules
- Escalate when a desired write exceeds scope permissions

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
