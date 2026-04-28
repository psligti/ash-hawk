---
name: transcript-tool-gap
description: 'Read the latest thin_runtime transcript artifact, identify what capability
  was

  missing, and name the specific tool needed. Produces a self-heal action plan

  with concrete evidence from execution.json.

  '
version: 1.0.0
metadata:
  author: Hephaestus
  date: 2026-04-23
  catalog_source: opencode
  legacy_skill_file: .opencode/skills/transcript-tool-gap/SKILL.md
---

# Transcript Tool Gap

Use this skill when a run fails and you need to quickly answer:

1. What was missing?
2. Which tool was actually needed?
3. What is the smallest self-heal patch to close the gap?

## What this skill does

This skill analyzes the latest thin runtime run artifact:

- `.ash-hawk/thin_runtime/runs/<run-id>/execution.json`

and outputs:

- Primary missing-tool gap (with confidence)
- Supporting evidence lines
- Fix hint (policy, handler, context producer, or delegation surface)
- Ranked gap candidates

## Run it

```bash
uv run python "scripts/analyze_latest_transcript.py"
```

Specific run:

```bash
uv run python "scripts/analyze_latest_transcript.py" --run-id <run-id>
```

JSON output for automation:

```bash
uv run python "scripts/analyze_latest_transcript.py" --format json
```

## Self-heal protocol

After diagnosis, do the smallest viable repair:

1. **missing_handler**
   - Implement the tool handler (`ash_hawk/thin_runtime/tool_impl/<tool>.py`)
   - Register in `ash_hawk/thin_runtime/defaults.py` if needed
   - Add targeted test

2. **policy_denied**
   - Add tool to the correct agent/skill surface, or
   - Delegate to an existing specialist that already has it

3. **missing_context**
   - Add producer tool before consumer tool
   - Typical producer examples: `load_workspace_state`, `run_baseline_eval`

4. **tool_surface_gap**
   - Expand active tool surface for the phase, or
   - Route with `delegate_task` + `requested_tools`

## Output contract

Always report:

- Run id
- Primary gap category + tool + confidence
- Exact evidence string(s)
- One concrete patch action
- Re-run command for verification

## Verification loop

After patching:

```bash
uv run pytest tests/thin_runtime/test_harness.py
uv run python "scripts/analyze_latest_transcript.py"
```

Success signal: primary gap confidence drops or category changes from hard gap
(`missing_handler` / `policy_denied`) to argument-level issue.
