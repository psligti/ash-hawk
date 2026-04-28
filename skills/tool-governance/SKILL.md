---
name: tool-governance
description: Governs the allowed tool surface and policy checks.
version: 1.0.0
metadata:
  id: skill.tool_governance
  name: tool-governance
  kind: skill
  version: 1.0.0
  status: active
  summary: Governs the allowed tool surface and policy checks.
  goal: Ensure only the correct tools are resolved and used under current runtime
    policy.
  file: skills/tool-governance.md
  category: execution
  scope: narrow
  when_to_use:
  - When tool access must be resolved or checked
  - When policy boundaries matter
  when_not_to_use:
  - When tool surface is already known and stable
  triggers:
  - tool resolution
  - policy enforcement
  - tool registration
  anti_triggers:
  - no tool interaction is required
  prerequisites:
  - tool_context is available
  - runtime_context is available
  inputs_expected:
  - tool state
  - runtime state
  procedure:
  - List tools
  - Resolve toolset
  - Register MCP tools if needed
  - Enforce tool policy
  - Record tool usage
  decision_points:
  - which tool surface is valid under policy
  fallback_strategy:
  - Prefer the smallest valid tool surface
  outputs:
    required_elements:
    - updated tool context
    - audit trail
  completion_criteria:
  - Allowed tool surface is explicit
  - policy status is explicit
  escalation_rules:
  - Escalate when the requested action exceeds tool policy
  dependencies:
    tools: &id001 []
    related_skills: []
    related_agents:
    - executor
    - coordinator
  examples:
  - description: Resolve and validate the tool surface
    input: 'The runtime needs to know which tools are allowed under current policy.

      '
    expected_behavior:
    - lists tools
    - resolves toolset
    - enforces policy
  tool_names: *id001
  input_contexts:
  - tool_context
  - runtime_context
  output_contexts:
  - tool_context
  - audit_context
  memory_read_scopes: []
  memory_write_scopes: []
  description: Governs the allowed tool surface and policy checks.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/tool-governance.md
---

# Purpose
Ensure only the correct tools are resolved and used under current runtime policy.

# Use This Skill When
- When tool access must be resolved or checked
- When policy boundaries matter

# Do Not Use This Skill When
- When tool surface is already known and stable

# Triggers
- tool resolution
- policy enforcement
- tool registration

# Anti-Triggers
- no tool interaction is required

# Prerequisites
- tool_context is available
- runtime_context is available

# Inputs Expected
- tool state
- runtime state

# Procedure
1. Confirm the skill applies.
2. List tools
3. Resolve toolset
4. Register MCP tools if needed
5. Enforce tool policy
6. Record tool usage

# Decision Points
- which tool surface is valid under policy

# Fallback Strategy
- Prefer the smallest valid tool surface

# Tool Contract
## Available Tools
- None

## Required Input Contexts
- tool_context
- runtime_context

## Produced Output Contexts
- tool_context
- audit_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- updated tool context
- audit trail

# Completion Criteria
- Allowed tool surface is explicit
- policy status is explicit

# Escalation Rules
- Escalate when the requested action exceeds tool policy

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
