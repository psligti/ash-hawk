# Ash Hawk

Python evaluation harness for AI agents.

## Quick Start

```bash
# Install
uv sync

# Run an evaluation suite
ash-hawk run path/to/suite.yaml --agent build

# List runs
ash-hawk list

# Generate report
ash-hawk report --run <run-id>
```

You can also set the default agent in suite YAML:

```yaml
agent:
  name: build

  # Optional: configure MCP servers for the runtime
  mcp_servers:
    - name: note-lark
      command: note-lark-mcp-stdio
```

Then run without CLI agent override:

```bash
ash-hawk run path/to/suite.yaml
```

## Documentation

- **[SKILL.md](SKILL.md)** - Complete usage guide for coding agents
- **[docs/skills/evals/README.md](docs/skills/evals/README.md)** - Skill graph for authoring evals (suites + scenarios)
- **[docs/PHASE2.md](docs/PHASE2.md)** - Phase 2 roadmap (pytest-like features)

## Examples

See `examples/complete-eval/` for a full example demonstrating:
- All three grader layers (deterministic, LLM, composite)
- Fixture creation and resolution
- Test file structure

## Improve Loop

Use the live improvement loop via the CLI:

```bash
ash-hawk improve evals/scenarios/ --agent build --max-iterations 5
```

Notes:
- `ash-hawk improve` is the supported entrypoint for iterative improvement in this repo.
- The legacy `auto_research` scripts and module paths have been retired from the live source tree.
- Pass a scenario file, glob, or directory. Directories are expanded recursively to `*.scenario.yaml` files.

## Related

- **dawn-kestrel** - Agent runtime SDK (dependency)
