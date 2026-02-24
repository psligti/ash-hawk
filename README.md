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
```

Then run without CLI agent override:

```bash
ash-hawk run path/to/suite.yaml
```

## Documentation

- **[SKILL.md](SKILL.md)** - Complete usage guide for coding agents
- **[docs/PHASE2.md](docs/PHASE2.md)** - Phase 2 roadmap (pytest-like features)

## Examples

See `examples/complete-eval/` for a full example demonstrating:
- All three grader layers (deterministic, LLM, composite)
- Fixture creation and resolution
- Test file structure

## Related

- **dawn-kestrel** - Agent runtime SDK (dependency)
