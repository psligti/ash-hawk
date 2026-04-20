---
id: skill.hypothesis_reranking
name: hypothesis-reranking
kind: skill
version: 1.0.0
status: active
summary: Reorders hypotheses after new evidence changes the situation.
goal: Re-rank candidate hypotheses when evaluation or failure state has changed.
file: skills/hypothesis-reranking.md
category: reasoning
scope: narrow
when_to_use:
- When new evidence changes the previous ranking
- When evaluation results alter priority
when_not_to_use:
- When ranking has not materially changed
triggers:
- new evaluation evidence
- changed failure state
anti_triggers:
- no hypothesis set exists
prerequisites:
- failure_context is available
- evaluation_context is available
inputs_expected:
- ranked hypotheses
- new evaluation evidence
procedure:
- Read updated evidence
- Re-rank hypotheses
- Update failure context
decision_points:
- whether ranking should change
fallback_strategy:
- Keep the previous order when evidence is inconclusive
outputs:
  required_elements:
  - re-ranked hypotheses
completion_criteria:
- Re-ranked hypotheses are explicit
escalation_rules:
- Escalate when evidence is too inconsistent to rerank responsibly
dependencies:
  tools: &id001
  - call_llm_structured
  related_skills:
  - hypothesis-ranking
  related_agents:
  - coordinator
  - researcher
examples:
- description: Re-rank after verification results change
  input: 'New evaluation evidence suggests the old ranking is no longer correct.

    '
  expected_behavior:
  - re-ranks hypotheses
  - updates failure context
tool_names: *id001
input_contexts:
- failure_context
- evaluation_context
output_contexts:
- failure_context
memory_read_scopes:
- episodic_memory
- semantic_memory
memory_write_scopes: []
description: Reorders hypotheses after new evidence changes the situation.
---
# Purpose
Re-rank candidate hypotheses when evaluation or failure state has changed.

# Use This Skill When
- When new evidence changes the previous ranking
- When evaluation results alter priority

# Do Not Use This Skill When
- When ranking has not materially changed

# Triggers
- new evaluation evidence
- changed failure state

# Anti-Triggers
- no hypothesis set exists

# Prerequisites
- failure_context is available
- evaluation_context is available

# Inputs Expected
- ranked hypotheses
- new evaluation evidence

# Procedure
1. Confirm the skill applies.
2. Read updated evidence
3. Re-rank hypotheses
4. Update failure context

# Decision Points
- whether ranking should change

# Fallback Strategy
- Keep the previous order when evidence is inconclusive

# Tool Contract
## Available Tools
- call_llm_structured

## Required Input Contexts
- failure_context
- evaluation_context

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
- re-ranked hypotheses

# Completion Criteria
- Re-ranked hypotheses are explicit

# Escalation Rules
- Escalate when evidence is too inconsistent to rerank responsibly

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
