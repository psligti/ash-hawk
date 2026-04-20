---
id: skill.knowledge_retrieval
name: knowledge-retrieval
kind: skill
version: 1.0.0
status: active
summary: Retrieves relevant lessons, rules, and prior episodes.
goal: Bring useful memory into the current run before acting.
file: skills/knowledge-retrieval.md
category: reasoning
scope: narrow
when_to_use:
- When prior knowledge may improve decisions
- When memory should be hydrated into context
when_not_to_use:
- When no useful memory is expected
- When the task is purely local and trivial
triggers:
- memory search
- prior-knowledge lookup
anti_triggers:
- no goal or failure context
prerequisites:
- goal_context is available
- failure_context is available
inputs_expected:
- goal
- failure state
procedure:
- Search knowledge
- Load episodic memory
- Load semantic memory
- Load personal preferences
decision_points:
- which memory sources matter
fallback_strategy:
- Use the most relevant available memory source first
outputs:
  required_elements:
  - memory context updates
completion_criteria:
- Relevant memory has been surfaced
escalation_rules:
- Escalate when available memory is too weak to inform the task
dependencies:
  tools: &id001
  - search_knowledge
  - read
  - glob
  - grep
  related_skills: []
  related_agents:
  - researcher
  - memory_manager
examples:
- description: Retrieve prior knowledge for a failure investigation
  input: 'The runtime should search prior lessons before choosing a concept.

    '
  expected_behavior:
  - loads relevant memory into context
tool_names: *id001
input_contexts:
- goal_context
- failure_context
output_contexts:
- memory_context
memory_read_scopes: []
memory_write_scopes: []
description: Retrieves relevant lessons, rules, and prior episodes.
---
# Purpose
Bring useful memory into the current run before acting.

# Use This Skill When
- When prior knowledge may improve decisions
- When memory should be hydrated into context

# Do Not Use This Skill When
- When no useful memory is expected
- When the task is purely local and trivial

# Triggers
- memory search
- prior-knowledge lookup

# Anti-Triggers
- no goal or failure context

# Prerequisites
- goal_context is available
- failure_context is available

# Inputs Expected
- goal
- failure state

# Procedure
1. Confirm the skill applies.
2. Search knowledge
3. Load episodic memory
4. Load semantic memory
5. Load personal preferences

# Decision Points
- which memory sources matter

# Fallback Strategy
- Use the most relevant available memory source first

# Tool Contract
## Available Tools
- search_knowledge
- read
- glob
- grep

## Required Input Contexts
- goal_context
- failure_context

## Produced Output Contexts
- memory_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- memory context updates

# Completion Criteria
- Relevant memory has been surfaced

# Escalation Rules
- Escalate when available memory is too weak to inform the task

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
