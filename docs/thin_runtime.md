# Thin Runtime

`ash_hawk.thin_runtime` is the agentic execution model for Ash Hawk's in-repo thin harness.

CLI entrypoint:

```bash
ash-hawk thin-runtime run "<goal>" --workdir <path>
```

## Scope

This runtime is intentionally thin:

- one active agent loop
- a thin harness façade that only wires registries, context, memory, hooks, and persistence
- agents loaded from markdown with front matter
- skills loaded from markdown with front matter
- skills as capability metadata
- deterministic tools
- agent text as a first-class execution input
- live context gating and unlocking
- scoped memory reads and writes
- hook emission for auditability
- durable run and memory persistence

It does **not** modify `ash_hawk.improve` behavior.

## Execution model

1. Resolve the active agent.
2. Resolve active skills from that agent.
3. Build agent text from the goal, agent intent, and active skills.
4. Resolve the allowed tool surface from the active skills and policy.
5. Assemble the initial context snapshot.
6. Read agent and skill memory scopes.
7. Select the next eligible tool dynamically.
8. Execute the tool and merge its payload back into context.
9. Unlock additional contexts through tool outputs and skill outputs.
10. Persist run state and memory snapshots.
11. Stop on error, iteration limits, or no more eligible tools.

## Agentic behavior

The runtime is agentic in three places:

- **agent text**: the runner composes explicit agent text from the goal, agent, and active skill set before execution
- **tool choice**: the loop prefers tool decisions through runtime state and tool outputs rather than static orchestration phases
- **delegation**: `delegate_task` can spawn delegated agent executions and capture their summaries as delegation records
- **context progression**: tools mutate runtime, evaluation, failure, memory, tool, workspace, and audit context, which changes future eligibility

## Catalog source of truth

Agents and skills are defined in markdown under `ash_hawk/thin_runtime/catalog/`.

- agent front matter defines identity, hooks, memory scopes, and attached skills
- skill front matter defines tools, context requirements, and memory scopes
- markdown bodies provide instructions that become part of runtime agent text

Tools are defined as contract-backed `ToolSpec` objects and implemented under `ash_hawk/thin_runtime/tool_impl/`.

- each tool has its own Python module with a stable `run` entrypoint
- tool metadata declares inputs, outputs, permissions, timeout, idempotence, observability, and escalation guidance
- tool contracts, calls, and results are enforced with pydantic models (`ToolSpec`, `ToolCall`, `ToolResult`, and `ToolExecutionPayload`)
- the runner constructs typed `ToolCall` objects and merges typed payload updates back into context
- the registry loads handlers from tool metadata rather than from a single monolithic dispatcher

## Typed tool framework

The thin runtime tool loop is pydantic-typed end-to-end:

- **Contract typing**: `ToolSpec.inputs` and `ToolSpec.outputs` use `ToolSchemaSpec`
- **Call typing**: tools receive `ToolCall` with typed runtime/workspace/eval/memory/tool/audit context
- **Result typing**: tools return `ToolResult` with `ToolExecutionPayload` for all context updates, delegation, stop flags, and observability
- **Shared command wrapper**: tool modules use a common `ToolCommand` runner to produce typed `ToolResult` output consistently
- **Schema authoring**: tool modules define inputs/outputs with typed schema builders (`basic_input_schema`, `context_input_schema`, `delegation_input_schema`, `standard_output_schema`) instead of inline dict JSON-schema payloads

This keeps the command-per-file pattern while removing untyped dict contracts at the framework boundary.

## Persistence

The runtime persists under `.ash-hawk/thin_runtime/` by default:

- `memory/snapshot.json`
- `runs/<run_id>/execution.json`
- `runs/<run_id>/summary.json`

## Realistic integrations

The runtime now includes lightweight integrations for:

- workspace inspection and git-repo detection
- agent config path detection
- optional scenario grading via `ThinScenarioRunner` when a scenario path is present in workspace context
- durable memory snapshots and run summaries

## Current caveats

- most workspace-management tools are deterministic and safe by default; eval tools now fail explicitly when no live scenario path is available
- delegation is recursive and auditable, but still intentionally bounded
- git commit behavior is detected but not executed automatically
- workspace scoping is driven by explicit scenario-required files and observed changed files, not repo-specific hidden target boosts
