---
name: signal-driven-workspace
description: signal-driven-workspace
version: 1.0.0
metadata:
  id: skill.signal_driven_workspace
  name: signal-driven-workspace
  kind: skill
  version: 1.0.0
  status: active
  file: skills/signal-driven-workspace.md
  category: execution
  scope: narrow
  tool_names:
  - load_workspace_state
  - detect_agent_config
  - scope_workspace
  - diff_workspace_changes
  - delegate_task
  input_contexts:
  - workspace_context
  output_contexts:
  - workspace_context
  - audit_context
  memory_read_scopes: []
  memory_write_scopes: []
  catalog_source: thin_runtime
  legacy_catalog_file: ash_hawk/thin_runtime/catalog/skills/signal-driven-workspace.md
allowed-tools:
- load_workspace_state
- detect_agent_config
- scope_workspace
- diff_workspace_changes
- delegate_task
---

# Purpose
Keep workspace work tightly focused on evidence that can improve the eval.

# Use This Skill When
- The run needs to inspect the workspace to understand a failing eval.
- The run needs to mutate a file after a concrete hypothesis exists.

# Do Not Use This Skill When
- The only available action is broad repo exploration with no eval question.
- The next best step is verification or re-evaluation, not more workspace inspection.

# High-Quality Signal
- The exact eval artifact or grader output that explains the current failure.
- A specific source/config file tied to the failing behavior.
- A diff from the previous mutation.
- A narrow grep query derived from an actual hypothesis.

# Weak Signal
- Broad file listings with no target question.
- Generic shell commands like `pwd` that do not inspect the failing evidence.
- Re-reading top-level docs because they are easy to access.
- Mutating the first available file without a file-specific reason.

# Procedure
1. Load workspace state once.
2. Detect the active agent/config path.
3. Use diagnosis or hypothesis output to name candidate files before any optional scoping step.
4. Delegate focused file inspection once a concrete hypothesis exists.
5. Delegate targeted search only when tied to the failing evidence.
6. Use diff output to confirm what changed.
7. Use delegate_task only after the target file is justified by evidence.

If `scope_workspace` returns zero targets, that is not completion. Diagnosis must derive the next file candidates from failure evidence or active agent/config files.

# Available Tools
## load_workspace_state
Use this to bootstrap the workspace snapshot.

## detect_agent_config
Use this to find the actual agent/config location.

## scope_workspace
Use this only to materialize diagnosis-derived targets into workspace context. Do not use it as the primary way to discover what to change.

## diff_workspace_changes
Use this to inspect the last mutation's actual effect.

## delegate_task
Use this to route deep inspection or mutation to a specialist with explicit requested skills and tools.
For mutation work, keep the candidate inside an isolated workspace until re-evaluation passes.

# Output Contract
## Required Elements
- the high-signal evidence consulted
- the target file or explicit reason no target is justified yet
- the next recommended workspace action

# Completion Criteria
- The workspace evidence is specific enough to support the next hypothesis or mutation.
- Any mutation step is tied to a justified target file.
