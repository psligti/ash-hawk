---
name: "direct-task-execution"
description: "Prioritizes immediate execution of user-provided tasks over formal tracking initialization, eliminating redundant markdown lists and blocking steps."
---

## What I do

I bypass the "planning" phase when the user has already defined the tasks. Instead of verifying tool availability or outputting redundant task lists, I immediately invoke file operation tools (read, edit) to begin solving the stated problems. I treat the user's prompt as the source of truth for the task scope.

## When to use me

Use me whenever the user provides an explicit list of tasks, bugs, features, or requirements to be completed:
- Numbered lists of fixes, features, or documentation updates.
- Requests involving multiple specific files (e.g., "Fix auth.py and update db.py").
- Scenarios where `todo_create` is unavailable, to avoid stalling on markdown fallbacks.

## Guidelines

### Immediate Action Principle
- **No STOP steps:** Do not pause to verify tool availability or "initialize" tracking.
- **No Markdown Fallbacks:** If `todo_create` is unavailable, do **NOT** generate a markdown TODO list. Proceed directly to file operations.
- **User List is Absolute:** The user's provided list is the task tracker; reiterating it as markdown is redundant and delays progress.

### Tool Usage Strategy
- **Start with Discovery:** Your first tool calls must be `read` requests for the files mentioned in the user's tasks (e.g., `auth.py`, `README.md`).
- **Unavailable Tools:** If a user asks to use `todo_create` but it is unavailable, ignore the request and focus on the actual file work. Do not report the tool availability as a blocking status.
- **Parallel Execution (If Applicable):** If `todo_create` *is* available, it may be called in the same block as file reads, but never as a standalone preliminary step.

### Verification
- Success is defined by immediately showing file contents or edit results, not by the presence of a checklist in the chat history.
- Do not output "STOP 0", "STOP 1", or similar workflow markers.