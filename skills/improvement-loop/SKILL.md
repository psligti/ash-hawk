---
name: improvement-loop
description: improvement-loop
version: 1.0.0
metadata:
  id: skill.improvement_loop
  name: improvement-loop
  kind: skill
  version: 1.0.0
  status: active
  file: skills/improvement-loop.md
  category: reasoning
  scope: narrow
  tool_names:
  - run_baseline_eval
  - run_eval_repeated
  - sync_workspace_changes
  - delegate_task
  - diff_workspace_changes
  - call_llm_structured
  input_contexts:
  - goal_context
  - runtime_context
  - workspace_context
  output_contexts:
  - runtime_context
  - evaluation_context
  - failure_context
  - audit_context
  memory_read_scopes: []
  memory_write_scopes: []
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/improvement-loop.md
allowed-tools:
- run_baseline_eval
- run_eval_repeated
- delegate_task
- diff_workspace_changes
- call_llm_structured
---

# Purpose
Own one complete improvement loop from diagnosis through re-evaluation.

# Loop Definition
One loop means:
1. Diagnose the most likely blocker from eval evidence.
2. Form one concrete hypothesis.
3. Apply one mutation.
4. Re-evaluate with `run_eval_repeated`.

Only the completed re-evaluation step counts against the iteration budget.

# Use This Skill When
- The goal is to improve a measured eval result.
- A baseline exists or can be created.
- The runtime needs to move from evidence to action.

# Do Not Use This Skill When
- The task is only reporting.
- No eval target is available.
- The workspace is unavailable for mutation.

# Required Process
## Baseline
- If no baseline exists, run `run_baseline_eval` once before the first loop.
- If the baseline is already perfect, first check whether the scenario or pack is teaching a reusable capability that still lives only in scenario-local instructions.

## Diagnosis
- Read the latest eval evidence and failure signals.
- Focus on one real blocker at a time.

## Hypothesis
- Produce one hypothesis narrow enough to test in a single mutation.
- Use `call_llm_structured` to spell out the diagnosis and multiple concrete ways to improve the result before mutating.
- The diagnosis output must identify targetable durable files; if target files are still unknown, the loop is not ready for mutation or completion.

## Mutation
- Use `delegate_task` for the actual mutation step, delegating to `executor` with explicit requested tools and skills.
- Prefer minimal, high-leverage changes.
- The mutation prompt must already contain the diagnosis, what went wrong, what is needed, and the ideal condition before the coding agent starts exploring.
- Do not use `delegate_task` for exploratory file reading before the baseline exists.
- Do not delegate until the loop has both baseline evidence and at least one mutation-ready target file.

## Re-evaluation
- Use `run_eval_repeated` after mutation.
- Do not use it as a duplicate baseline tool.
- After a successful isolated candidate validation, use `sync_workspace_changes` exactly once to promote the candidate back to the primary workspace.

## Perfect Baseline Rule
- Historical memory can explain prior failures, but it must not force a mutation when the current run is already passing.
- Current eval evidence outranks historical memory for deciding whether another loop is needed.
- A perfect baseline on a capability-building pack is not enough by itself if the capability is still absent from the coding agent's durable prompt/code surface.

## Verification
- Use the result of `run_eval_repeated` to decide whether the loop actually improved the score.
- A loop that ends before mutation or re-evaluation is incomplete and must not be treated as success.

# Procedure
1. Establish the baseline if it does not already exist.
2. Diagnose one blocker from the eval evidence.
3. Form one concrete hypothesis.
4. Apply one mutation.
5. Use `run_eval_repeated` to close the loop.
6. Sync validated candidate changes back to the primary workspace.
7. Verify whether the loop improved the result.

# Available Tools
## run_baseline_eval
Starting measurement before the loop.

## run_eval_repeated
Loop-closing re-evaluation to confirm stability after a change.

## sync_workspace_changes
Use this only after a validated isolated candidate is ready to be promoted back to the primary workspace.

## delegate_task
Delegates bounded mutation work to the right specialist agent.

## diff_workspace_changes
Shows what actually changed before or after a mutation.

## call_llm_structured
Use this to make the diagnosis explicit and enumerate multiple concrete hypotheses before choosing a mutation.

# Output Contract
## Required Elements
- current blocker
- current hypothesis
- mutation status
- re-evaluation result
- next step or stop decision

# Completion Criteria
- A full improvement loop completed with a re-eval.
- The next action or stop decision is explicit.

# Escalation Rules
- Escalate when no plausible mutation remains.
- Escalate when repeated loops stop producing signal.
