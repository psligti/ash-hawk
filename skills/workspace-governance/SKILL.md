---
name: workspace-governance
description: Governs workspace scope, isolation, and sync behavior.
version: 1.0.0
metadata:
  id: skill.workspace_governance
  name: workspace-governance
  kind: skill
  version: 1.0.0
  status: active
  summary: Governs workspace scope, isolation, and sync behavior.
  goal: Keep workspace operations safe, scoped, and auditable.
  file: skills/workspace-governance.md
  category: execution
  scope: narrow
  when_to_use:
  - When workspace operations are needed
  - When isolation or sync should be controlled
  when_not_to_use:
  - When no workspace interaction is required
  triggers:
  - workspace inspection
  - isolated execution
  - sync need
  anti_triggers:
  - pure reasoning tasks
  prerequisites:
  - workspace_context is available
  inputs_expected:
  - workspace state
  procedure:
  - Scope the workspace
  - Prepare isolated workspace
  - Sync or diff changes as needed
  decision_points:
  - isolation vs direct scope work
  fallback_strategy:
  - Prefer inspection and diffing before sync behavior
  outputs:
    required_elements:
    - workspace updates
    - audit trail
  completion_criteria:
  - Workspace actions are explicit and within scope
  escalation_rules:
  - Escalate when requested workspace behavior exceeds permissions
  dependencies:
    tools: &id001
    - load_workspace_state
    - scope_workspace
    - prepare_isolated_workspace
    - diff_workspace_changes
    - sync_workspace_changes
    - commit_workspace_changes
    - mutate_agent_files
    - detect_agent_config
    - read
    - write
    - edit
    - glob
    - grep
    - bash
    - test
    related_skills: []
    related_agents:
    - executor
  examples:
  - description: Prepare and inspect a scoped workspace
    input: 'The runtime needs to isolate workspace actions and inspect the resulting
      changes.

      '
    expected_behavior:
    - scopes workspace
    - prepares isolation
    - diffs changes
  tool_names: *id001
  input_contexts:
  - workspace_context
  output_contexts:
  - workspace_context
  - audit_context
  memory_read_scopes: []
  memory_write_scopes: []
  description: Governs workspace scope, isolation, and sync behavior.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/workspace-governance.md
allowed-tools:
- load_workspace_state
- scope_workspace
- prepare_isolated_workspace
- diff_workspace_changes
- sync_workspace_changes
- commit_workspace_changes
- mutate_agent_files
- detect_agent_config
- read
- write
- edit
- glob
- grep
- bash
- test
---

# Purpose
Keep workspace operations safe, scoped, and auditable.

# Use This Skill When
- When workspace operations are needed
- When isolation or sync should be controlled

# Do Not Use This Skill When
- When no workspace interaction is required

# Triggers
- workspace inspection
- isolated execution
- sync need

# Anti-Triggers
- pure reasoning tasks

# Prerequisites
- workspace_context is available

# Inputs Expected
- workspace state

# Procedure
1. Confirm the skill applies.
2. Scope the workspace
3. Prepare isolated workspace
4. Sync or diff changes as needed

# Decision Points
- isolation vs direct scope work

# Fallback Strategy
- Prefer inspection and diffing before sync behavior

# Tool Contract
## Available Tools
- load_workspace_state
- scope_workspace
- prepare_isolated_workspace
- diff_workspace_changes
- sync_workspace_changes
- commit_workspace_changes
- mutate_agent_files
- detect_agent_config
- read
- write
- edit
- glob
- grep
- bash
- test

## Required Input Contexts
- workspace_context

## Produced Output Contexts
- workspace_context
- audit_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- workspace updates
- audit trail

# Completion Criteria
- Workspace actions are explicit and within scope
- Candidate changes are not synced back to the primary workspace until validation passes

# Escalation Rules
- Escalate when requested workspace behavior exceeds permissions

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
