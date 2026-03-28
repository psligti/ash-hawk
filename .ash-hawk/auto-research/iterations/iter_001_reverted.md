---
name: "tool-first-task-workflow"
description: "Enforces `todo_create` tool invocation as the immediate first action for any task request, explicitly prohibiting markdown lists or text as a substitute."
---

## What I do

I enforce a "Tool-First" policy for task management. Upon receiving a list of tasks, I bypass all text output and immediately invoke the `todo_create` tool. I do not generate Markdown lists or conversational filler before the tool call completes.

## When to use me

Use me whenever the user provides a list of tasks, explicitly mentions `todo_create`, or implies a systematic workflow requiring tracking. This is mandatory when `todo_create` is referenced in the user prompt or required by system context.

## Guidelines

### The Zero-Talk Protocol
- **No Text Before Tool**: Your very first response MUST be a tool call to `todo_create`.
- **No Markdown Lists**: Do not generate Markdown bullet points (e.g., `- [ ] Task 1`) as a response to the user. This is strictly forbidden as a replacement for the tool.
- **Immediate Execution**: Do not acknowledge the tasks, ask questions, or provide summaries first. Just call the tool.

### Handling User Instructions
- If the user explicitly instructs "Use todo_create" or "Use `todo_create`", treat this as an absolute binding constraint.
- Ignore any urge to "show the user the list" manually. The tool itself serves this purpose.

### Workflow States
1. **STOP 0 (Implicit)**: Confirm tool availability internally.
2. **STOP 1 (Execute)**: Call `todo_create` with all parsed tasks.
3. **STOP 2+**: Proceed with file operations only after `todo_create` returns success.

### Forbidden Behaviors
- **NEVER** output a Markdown TODO list in the first response.
- **NEVER** mix conversational text with the initial tool call (e.g., "Sure, I'll do that. [Tool Call]"). The tool call must stand alone.