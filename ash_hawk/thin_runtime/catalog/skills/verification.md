---
id: skill.verification
name: verification
kind: skill
version: 1.0.0
status: active
summary: Verifies outcomes and aligns claims with evidence.
goal: Prove that observed results and reported claims are consistent with evidence.
file: skills/verification.md
category: verification
scope: narrow
when_to_use:
- When claims must be checked
- When evaluation and audit evidence exist
when_not_to_use:
- When no evidence exists yet
- When only execution is needed
triggers:
- claim verification
- outcome confirmation
anti_triggers:
- missing evaluation context
prerequisites:
- evaluation_context is available
- audit_context is available
inputs_expected:
- evaluation evidence
- audit trail
procedure:
- Verify the outcome
- Audit the claims
- Update evaluation and audit context
decision_points:
- sufficient evidence vs escalation
fallback_strategy:
- Escalate when evidence is insufficient
outputs:
  required_elements:
  - verification result
  - audited claims
completion_criteria:
- Verification is explicit
- Claim alignment is explicit
escalation_rules:
- Escalate when claims cannot be supported
dependencies:
  tools: &id001
  - verify_outcome
  - audit_claims
  related_skills: []
  related_agents:
  - verifier
  - reviewer
examples:
- description: Verify final claims
  input: 'The runtime has evaluation and audit context and needs to verify the final
    result.

    '
  expected_behavior:
  - verifies outcome
  - audits claims
tool_names: *id001
input_contexts:
- evaluation_context
- audit_context
output_contexts:
- evaluation_context
- audit_context
memory_read_scopes: []
memory_write_scopes: []
description: Verifies outcomes and aligns claims with evidence.
---
# Purpose
Prove that observed results and reported claims are consistent with evidence.

# Use This Skill When
- When claims must be checked
- When evaluation and audit evidence exist

# Do Not Use This Skill When
- When no evidence exists yet
- When only execution is needed

# Triggers
- claim verification
- outcome confirmation

# Anti-Triggers
- missing evaluation context

# Prerequisites
- evaluation_context is available
- audit_context is available

# Inputs Expected
- evaluation evidence
- audit trail

# Procedure
1. Confirm the skill applies.
2. Verify the outcome
3. Audit the claims
4. Update evaluation and audit context

# Decision Points
- sufficient evidence vs escalation

# Fallback Strategy
- Escalate when evidence is insufficient

# Tool Contract
## Available Tools
- verify_outcome
- audit_claims

## Required Input Contexts
- evaluation_context
- audit_context

## Produced Output Contexts
- evaluation_context
- audit_context

# Memory Contract
## Memory Read Scopes
- None

## Memory Write Scopes
- None

# Output Contract
## Required Elements
- verification result
- audited claims

# Completion Criteria
- Verification is explicit
- Claim alignment is explicit

# Escalation Rules
- Escalate when claims cannot be supported

# Guardrails
- Do not use this skill outside its declared scope.
- Prefer a narrower skill when one is more applicable.
- Do not claim completion unless the completion criteria are satisfied.
