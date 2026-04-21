# Improve 4-Layer Memory Spec

## Objective

Reduce wasted improve-run cycles and token spend by turning lesson memory into a layered memory system that captures the full mutation funnel, not just terminal kept/reverted outcomes.

## Scope

This spec applies to `ash_hawk/improve/*` and improve-run artifacts.

## Memory Layers

1. **Working context**
   - Purpose: volatile run-local execution state
   - Storage: `.ash-hawk/memory/working/<run_id>.json`
   - Contents: active trial ids, current iteration, hypothesis count, stop context
   - Retention: overwrite during run, archive by run id

2. **Episodic memory**
   - Purpose: append-only event stream for every hypothesis attempt outcome
   - Storage: `.ash-hawk/memory/episodic/<run_id>.jsonl`
   - Contents: trial/family/targets, outcome (`no_file_changes`, `mutation_cli_timeout`, `targeted_regression`, `kept`, `reverted`, etc.), confidence, retry count, cost metrics
   - Retention: long-lived, append-only

3. **Semantic memory**
   - Purpose: distilled recurring rules from episodic patterns
   - Storage: `.ash-hawk/memory/semantic/rules.json`
   - Rule categories:
     - `friction_no_change`
     - `friction_timeout`
     - `success_pattern`
   - Rule payload: family, target signature, penalty/boost, evidence count
   - Retention: durable; updated by consolidation

4. **Personal memory**
   - Purpose: stable user/repo preferences and constraints
   - Storage: `.ash-hawk/memory/personal/preferences.json`
   - Retention: durable, manually or system updated

## Runtime Behavior

### Recording

- Persist working snapshots per iteration.
- Persist episodic entries for all branch outcomes, including early non-converting outcomes.

### Retrieval and gating

- Before speculative mutation execution, consult memory for conversion risk:
  - Compute historical success/timeout/no-change rates by `(diagnosis_family, target_signature)`.
  - Apply semantic penalties from friction rules.
  - Skip hypothesis when conversion probability is below threshold and friction is high.

### Consolidation

At end of each improve run:

- Group episodic entries by `(diagnosis_family, target_signature)`.
- Promote recurring patterns to semantic rules:
  - high no-change rate -> `friction_no_change`
  - high timeout rate -> `friction_timeout`
  - high keep rate -> `success_pattern`
- Emit consolidation summary artifact.

## APIs

`ash_hawk/improve/memory_store.py`

- `save_working(snapshot)`
- `append_episode(episode)`
- `load_episodes(run_id=None)`
- `load_semantic_rules()` / `save_semantic_rules(rules)`
- `load_personal_preferences()`
- `should_skip_hypothesis(agent_name, diagnosis_family, target_files, ...)`
- `calibration_factor(diagnosis_family, ...)`
- `consolidate_run(run_id)`

## Improve-loop integration

- Instantiate `MemoryStore` once per run.
- Record working snapshots at iteration boundaries.
- Record episodic outcomes on:
  - hypothesis exception
  - low-conversion memory skip
  - non-ready mutation outcomes
  - targeted regression
  - kept/reverted accepted candidates
- Add memory summary to:
  - `run.json`
  - `summary.md`
  - `memory_summary.json`

## Metrics and success criteria

Track and compare rolling windows:

- conversion rate: kept / attempted
- friction rates: no-change, timeout
- skip precision: skipped candidates that historically under-convert
- median attempts-to-first-kept
- token and wall-clock cost per kept mutation

## Non-goals

- No hard bans on code/prompt/skill/tool surfaces.
- No external DB dependency in this phase.

## Rollout

### Phase A (implemented)
- File-backed 4-layer model, attempt-level episodic logging, semantic consolidation, low-conversion skip gating.

### Phase B
- Incorporate calibration factor into ranking impact scoring.

### Phase C
- Add periodic decay/archival for episodic entries and semantic confidence aging.
