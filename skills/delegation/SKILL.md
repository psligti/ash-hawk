---
name: delegation
description: Delegates work to a more suitable agent with a bounded handoff.
version: 1.0.0
metadata:
  id: skill.delegation
  name: delegation
  kind: skill
  version: 1.0.0
  status: active
  summary: Delegates work to a more suitable agent with a bounded handoff.
  goal: Route a subtask to another agent when specialization improves outcome quality
    or speed.
  file: skills/delegation.md
  category: reasoning
  scope: narrow
  when_to_use:
  - When a subtask boundary is clear
  - When another agent is more specialized
  - When delegation improves quality or speed
  when_not_to_use:
  - When the current agent can finish directly
  - When the delegation boundary is unclear
  - When oversight cannot be maintained
  triggers:
  - Need specialist execution or verification
  - Need to offload a bounded subtask
  anti_triggers:
  - Delegation would add ambiguity
  - Delegation would exceed depth or breadth limits
  prerequisites:
  - goal_context is available
  - runtime_context is available
  inputs_expected:
  - current goal
  - active runtime state
  - candidate target agent
  procedure:
  - Confirm delegation improves the outcome
  - Identify the right target agent, requested skills, and requested tool surface
  - Dispatch the handoff and record it in audit context
  decision_points:
  - Whether delegation is warranted
  - Which agent should receive the subtask
  fallback_strategy:
  - Keep ownership local if delegation is not clearly better
  - Escalate if no suitable target exists
  outputs:
    required_elements:
    - delegated target
    - audit record
  completion_criteria:
  - Delegation is explicit and bounded
  - Audit context records the handoff
  escalation_rules:
  - Escalate when the right target agent is unclear
  dependencies:
    tools: &id001
    - delegate_task
    related_skills: []
    related_agents:
    - coordinator
    - researcher
    - executor
    - verifier
    - reviewer
    - memory_manager
  examples:
  - description: Hand off verification to a specialized agent
    input: 'The current agent has reached a point where verification should take over.

      '
    expected_behavior:
    - resolves the target agent
    - delegates the task with context
  tool_names: *id001
  input_contexts:
  - goal_context
  - runtime_context
  output_contexts:
  - runtime_context
  - audit_context
  memory_read_scopes: []
  memory_write_scopes: []
  description: Delegates work to a more suitable agent with a bounded handoff.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/delegation.md
allowed-tools:
- delegate_task
---

# Purpose
Route a subtask to another agent when specialization improves outcome quality or speed.

# Use This Skill When
- When a subtask boundary is clear
- When another agent is more specialized
- When delegation improves quality or speed

# Do Not Use This Skill When
- When the current agent can finish directly
- When the delegation boundary is unclear
- When oversight cannot be maintained

# Triggers
- Need specialist execution or verification
- Need to offload a bounded subtask

# Anti-Triggers
- Delegation would add ambiguity
- Delegation would exceed depth or breadth limits

# Prerequisites
- goal_context is available
- runtime_context is available

# Inputs Expected
- current goal
- active runtime state
- candidate target agent

# Procedure
1. Confirm the skill applies.
2. Confirm delegation improves the outcome
3. Identify the right target agent and requested skills
4. Identify requested tool constraints for the delegated run
5. Dispatch the handoff and record it in audit context

# Decision Points
- Whether delegation is warranted
- Which agent should receive the subtask

# Fallback Strategy
- Keep ownership local if delegation is not clearly better
- Escalate if no suitable target exists

# Tool Contract
## Available Tools
- delegate_task

## Required Input Contexts
- goal_context
- runtime_context

## Produced Output Contexts
- runtime_context
- audit_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- delegated target
- audit record

# Completion Criteria
- Delegation is explicit and bounded
- Audit context records the handoff

# Escalation Rules
- Escalate when the right target agent is unclear

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
