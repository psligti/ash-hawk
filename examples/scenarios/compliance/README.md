# Skills/Tools/MCP Compliance Scenarios

This scenario pack focuses on validating whether an agent follows skill/tool/MCP constraints.

## What it checks

- Required trace event types are emitted (`ToolCallEvent`, etc.)
- MCP usage constraints are met via tool-name prefixes (for example `note-lark_`)
- Forbidden tools are not used (for example `bash`)
- Output contains required skill markers (for example `note-lark-memory`)

## Included scenarios

- `skills_tools_mcp_compliance.scenario.yaml`

## Validate

```bash
uv run ash-hawk scenario validate examples/scenarios/compliance/
```

## Run

```bash
uv run ash-hawk scenario run examples/scenarios/compliance/skills_tools_mcp_compliance.scenario.yaml --sut mock_adapter
```
