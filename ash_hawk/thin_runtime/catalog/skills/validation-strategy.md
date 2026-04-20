---
id: skill.validation_strategy
name: validation-strategy
kind: skill
version: 1.0.0
status: active
summary: Chooses targeted or integrity validation paths.
goal: Apply the right validation depth based on current evaluation and workspace state.
file: skills/validation-strategy.md
category: verification
scope: narrow
when_to_use:
- When validation is needed after execution
- When evaluation context exists
when_not_to_use:
- When no evaluation path is available
- When only baseline establishment is needed
triggers:
- post-execution validation
- verification depth choice
anti_triggers:
- missing workspace context
prerequisites:
- evaluation_context is available
- workspace_context is available
inputs_expected:
- evaluation state
- workspace scope
procedure:
- Choose targeted or integrity validation
- Run the chosen validation path
- Record results
decision_points:
- targeted vs integrity validation
fallback_strategy:
- Prefer targeted validation when scope is known
- Escalate when neither path is credible
outputs:
  required_elements:
  - validation result
  - updated evaluation context
completion_criteria:
- Validation path is chosen explicitly
- Evaluation context is updated
escalation_rules:
- Escalate when validation cannot be completed credibly
dependencies:
  tools: &id001
  - run_targeted_validation
  - run_integrity_validation
  related_skills:
  - verification
  related_agents:
  - verifier
examples:
- description: Validate after execution
  input: 'The runtime needs to decide whether to run targeted or integrity validation.

    '
  expected_behavior:
  - chooses the correct validation path
  - updates evaluation context
tool_names: *id001
input_contexts:
- evaluation_context
- workspace_context
output_contexts:
- evaluation_context
memory_read_scopes: []
memory_write_scopes: []
description: Chooses targeted or integrity validation paths.
---
# Purpose
Apply the right validation depth based on current evaluation and workspace state.

# Use This Skill When
- When validation is needed after execution
- When evaluation context exists

# Do Not Use This Skill When
- When no evaluation path is available
- When only baseline establishment is needed

# Triggers
- post-execution validation
- verification depth choice

# Anti-Triggers
- missing workspace context

# Prerequisites
- evaluation_context is available
- workspace_context is available

# Inputs Expected
- evaluation state
- workspace scope

# Procedure
1. Confirm the skill applies.
2. Choose targeted or integrity validation
3. Run the chosen validation path
4. Record results

# Decision Points
- targeted vs integrity validation

# Fallback Strategy
- Prefer targeted validation when scope is known
- Escalate when neither path is credible

# Tool Contract
## Available Tools
- run_targeted_validation
- run_integrity_validation

## Required Input Contexts
- evaluation_context
- workspace_context

## Produced Output Contexts
- evaluation_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- validation result
- updated evaluation context

# Completion Criteria
- Validation path is chosen explicitly
- Evaluation context is updated

# Escalation Rules
- Escalate when validation cannot be completed credibly

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
