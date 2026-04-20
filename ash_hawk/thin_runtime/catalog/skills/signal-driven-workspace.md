---
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
- read
- grep
- diff_workspace_changes
- mutate_agent_files
input_contexts:
- workspace_context
output_contexts:
- workspace_context
- audit_context
memory_read_scopes: []
memory_write_scopes: []
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
3. Scope the workspace only to files relevant to the current hypothesis.
4. Read the minimum files needed to support or reject that hypothesis.
5. Use grep only with a targeted pattern tied to the failing evidence.
6. Use diff output to confirm what changed.
7. Use mutate_agent_files only after the target file is justified by evidence.

# Available Tools
## load_workspace_state
Use this to bootstrap the workspace snapshot.

## detect_agent_config
Use this to find the actual agent/config location.

## scope_workspace
Use this to narrow the mutation space.

## read
Use this for targeted file inspection.

## grep
Use this only for targeted searches tied to a hypothesis.

## diff_workspace_changes
Use this to inspect the last mutation's actual effect.

## mutate_agent_files
Use this to apply the mutation once the target file and change are justified.

# Output Contract
## Required Elements
- the high-signal evidence consulted
- the target file or explicit reason no target is justified yet
- the next recommended workspace action

# Completion Criteria
- The workspace evidence is specific enough to support the next hypothesis or mutation.
- Any mutation step is tied to a justified target file.
