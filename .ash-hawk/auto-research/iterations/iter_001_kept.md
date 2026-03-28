---
name: "blocking-task-fallback-protocol"
description: "Enforces a hard stop after task list initialization, strictly isolating the setup phase from any file operations to prevent early execution."
---

## What I do

I enforce a strict isolation of the task initialization phase. When responding to multi-task requests, I ensure that creating the task list (whether via `todo_create` tool or Markdown fallback) is the **only** action taken in that turn. I never combine task creation with file discovery, reading, or editing.

## When to use me

Use me for all multi-task requests, particularly when:
- Users provide a list of tasks requiring tracking.
- There is ambiguity about tool availability (`todo_create`).
- The agent attempts to "streamline" by doing setup and file reading simultaneously.

## Guidelines

### The Blocking Principle
- **Turn Isolation**: Task creation and Task Execution must happen in separate turns.
- **Step 1 (Setup)**: Check tools. Create task list (via `todo_create` OR Markdown). **STOP**.
- **Step 2 (Execution)**: Only begin file discovery (`read`, `ls`) *after* the task list has been successfully created and the user has responded.

### Tool Unavailability Protocol
- If `todo_create` is unavailable, output a Markdown TODO list.
- **CRITICAL HALT**: Do not generate any tool calls (e.g., `read`, `edit`) in the same response as the Markdown list.
- Do not assume user intent to proceed immediately; wait for a new user prompt to begin execution.
- Example of **FORBIDDEN** behavior:
  - Assistant: "Here is the list: [Markdown]... I will now read the files." (implied `read` tool call)
- Example of **CORRECT** behavior:
  - Assistant: "Here is the list: [Markdown]" (End of response. No tool calls)

### Verification
- Before invoking `read` or `edit`, verify that a task list exists in the conversation history AND that the current turn is not the turn where the list was created.
- If you catch yourself about to output a Markdown list and a file read in the same thought process, remove the file read and output only the list.