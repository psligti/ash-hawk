---
id: skill.hypothesis_ranking
name: hypothesis-ranking
kind: skill
version: 1.0.0
status: active
summary: Prioritizes candidate hypotheses using durable knowledge.
goal: Rank current hypotheses into a useful execution order.
file: skills/hypothesis-ranking.md
category: reasoning
scope: narrow
when_to_use:
- When multiple hypotheses exist
- When memory should influence prioritization
when_not_to_use:
- When only one hypothesis exists
triggers:
- need execution order for hypotheses
anti_triggers:
- no meaningful hypothesis set
prerequisites:
- failure_context is available
- memory_context is available
inputs_expected:
- current hypotheses
- episodic memory
- semantic memory
procedure:
- Read hypotheses and memory
- Rank the hypotheses
- Update failure context
decision_points:
- which hypothesis should be first
fallback_strategy:
- Prefer the simplest high-confidence hypothesis
outputs:
  required_elements:
  - ranked hypotheses
completion_criteria:
- Ranked hypotheses are explicit
escalation_rules:
- Escalate when hypotheses are too weak to rank credibly
dependencies:
  tools: &id001
  - call_llm_structured
  related_skills: []
  related_agents:
  - researcher
  - coordinator
examples:
- description: Rank candidate hypotheses
  input: 'The runtime has several hypotheses and needs a ranked order.

    '
  expected_behavior:
  - uses memory
  - returns ranked hypotheses
tool_names: *id001
input_contexts:
- failure_context
- memory_context
output_contexts:
- failure_context
memory_read_scopes:
- episodic_memory
- semantic_memory
memory_write_scopes: []
description: Prioritizes candidate hypotheses using durable knowledge.
---
# Purpose
Rank current hypotheses into a useful execution order.

# Use This Skill When
- When multiple hypotheses exist
- When memory should influence prioritization

# Do Not Use This Skill When
- When only one hypothesis exists

# Triggers
- need execution order for hypotheses

# Anti-Triggers
- no meaningful hypothesis set

# Prerequisites
- failure_context is available
- memory_context is available

# Inputs Expected
- current hypotheses
- episodic memory
- semantic memory

# Procedure
1. Confirm the skill applies.
2. Read hypotheses and memory
3. Rank the hypotheses
4. Update failure context

# Decision Points
- which hypothesis should be first

# Fallback Strategy
- Prefer the simplest high-confidence hypothesis

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
- episodic_memory
- semantic_memory

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- ranked hypotheses

# Completion Criteria
- Ranked hypotheses are explicit

# Escalation Rules
- Escalate when hypotheses are too weak to rank credibly

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
