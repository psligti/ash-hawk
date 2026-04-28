---
name: diagnosis-routing
description: Chooses the diagnosis mode and fallback chain.
version: 1.0.0
metadata:
  id: skill.diagnosis_routing
  name: diagnosis-routing
  kind: skill
  version: 1.0.0
  status: active
  summary: Chooses the diagnosis mode and fallback chain.
  goal: Route diagnosis through the most appropriate mode for the current failure
    state.
  file: skills/diagnosis-routing.md
  category: reasoning
  scope: narrow
  when_to_use:
  - When failure diagnosis needs routing
  - When different diagnosis modes are available
  when_not_to_use:
  - When failure analysis is already complete
  - When tool context is missing
  triggers:
  - diagnosis mode choice
  - diagnosis fallback
  anti_triggers:
  - no failure context
  prerequisites:
  - failure_context is available
  - tool_context is available
  inputs_expected:
  - failure state
  - available tools
  procedure:
  - Inspect failure state
  - Choose diagnosis mode
  - Invoke fallback if needed
  decision_points:
  - tool diagnosis vs subprocess vs llm path
  fallback_strategy:
  - Use the simplest credible diagnosis mode first
  outputs:
    required_elements:
    - diagnosis mode
    - audit trail
  completion_criteria:
  - Diagnosis mode is explicit
  - audit trail records the route
  escalation_rules:
  - Escalate when no diagnosis path is credible
  dependencies:
    tools: &id001 []
    related_skills: []
    related_agents:
    - researcher
  examples:
  - description: Choose a diagnosis route
    input: 'The runtime needs to decide which diagnosis mode to use.

      '
    expected_behavior:
    - selects a diagnosis mode
    - records the route
  tool_names: *id001
  input_contexts:
  - failure_context
  - tool_context
  output_contexts:
  - failure_context
  - audit_context
  memory_read_scopes: []
  memory_write_scopes: []
  description: Chooses the diagnosis mode and fallback chain.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/diagnosis-routing.md
---

# Purpose
Route diagnosis through the most appropriate mode for the current failure state.

# Use This Skill When
- When failure diagnosis needs routing
- When different diagnosis modes are available

# Do Not Use This Skill When
- When failure analysis is already complete
- When tool context is missing

# Triggers
- diagnosis mode choice
- diagnosis fallback

# Anti-Triggers
- no failure context

# Prerequisites
- failure_context is available
- tool_context is available

# Inputs Expected
- failure state
- available tools

# Procedure
1. Confirm the skill applies.
2. Inspect failure state
3. Choose diagnosis mode
4. Invoke fallback if needed

# Decision Points
- tool diagnosis vs subprocess vs llm path

# Fallback Strategy
- Use the simplest credible diagnosis mode first

# Tool Contract
## Available Tools
- None

## Required Input Contexts
- failure_context
- tool_context

## Produced Output Contexts
- failure_context
- audit_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- diagnosis mode
- audit trail

# Completion Criteria
- Diagnosis mode is explicit
- audit trail records the route

# Escalation Rules
- Escalate when no diagnosis path is credible

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
