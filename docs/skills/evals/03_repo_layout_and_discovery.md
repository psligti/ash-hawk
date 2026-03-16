# A2. File layout + discoverability

Treat eval authoring as a content system with stable locations.

## Recommended layout

- `evals/` for your suite content
- `examples/` for runnable references
- `docs/skills/evals/` for authoring knowledge graph

From `docs/PHASE2.md`, the intended pytest-like direction includes:
- discovery patterns,
- directory-level defaults (`conftest.yaml`),
- project-level defaults (`pyproject.toml`).

## Scenario discovery rules (today)

`ash_hawk/scenario/loader.py` discovers:
- `*.scenario.yaml`
- `*.scenario.yml`

from a file or directory root.

## Why this matters

Good layout creates deep veins of reusable knowledge:
- one place for schema contracts,
- one place for examples,
- one place for policy/protocol packs.
