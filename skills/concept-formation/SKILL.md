---
name: concept-formation
description: Produces candidate concepts for the next action.
version: 1.0.0
metadata:
  id: skill.concept_formation
  name: concept-formation
  kind: skill
  version: 1.0.0
  status: active
  summary: Produces candidate concepts for the next action.
  goal: Turn failure and memory state into useful candidate next actions.
  file: skills/concept-formation.md
  category: reasoning
  scope: narrow
  when_to_use:
  - When failures are understood enough to propose next moves
  - When memory may shape new concepts
  when_not_to_use:
  - When failure state is still too weak or ambiguous
  triggers:
  - need candidate concepts
  anti_triggers:
  - missing failure context
  prerequisites:
  - failure_context is available
  - memory_context is available
  inputs_expected:
  - failure state
  - memory state
  procedure:
  - Read failure and memory state
  - Propose concepts
  - Update failure context
  decision_points:
  - which concepts are plausible enough to keep
  fallback_strategy:
  - Return a small conservative concept set if confidence is low
  outputs:
    required_elements:
    - concept list
  completion_criteria:
  - Concepts are explicit and usable for planning
  escalation_rules:
  - Escalate when no plausible concept can be formed
  dependencies:
    tools: &id001
    - call_llm_structured
    related_skills: []
    related_agents:
    - researcher
  examples:
  - description: Produce next-step concepts from a failure
    input: 'The runtime understands the failure enough to propose possible next steps.

      '
    expected_behavior:
    - proposes concepts
    - updates failure context
  tool_names: *id001
  input_contexts:
  - failure_context
  - memory_context
  output_contexts:
  - failure_context
  memory_read_scopes: []
  memory_write_scopes: []
  description: Produces candidate concepts for the next action.
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/concept-formation.md
allowed-tools:
- call_llm_structured
---

# Purpose
Turn failure and memory state into useful candidate next actions.

# Use This Skill When
- When failures are understood enough to propose next moves
- When memory may shape new concepts

# Do Not Use This Skill When
- When failure state is still too weak or ambiguous

# Triggers
- need candidate concepts

# Anti-Triggers
- missing failure context

# Prerequisites
- failure_context is available
- memory_context is available

# Inputs Expected
- failure state
- memory state

# Procedure
1. Confirm the skill applies.
2. Read failure and memory state
3. Propose concepts
4. Update failure context

# Decision Points
- which concepts are plausible enough to keep

# Fallback Strategy
- Return a small conservative concept set if confidence is low

# Tool Contract
## Available Tools
- call_llm_structured

## Required Input Contexts
- failure_context
- memory_context

## Produced Output Contexts
- failure_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- concept list

# Completion Criteria
- Concepts are explicit and usable for planning

# Escalation Rules
- Escalate when no plausible concept can be formed

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
