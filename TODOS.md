# TODOS

## Provenance & Causality

- [x] Add `RunManifest` dataclass to `ash_hawk/bridge/__init__.py` for run attribution
- [x] Add `DiffReport` dataclass to `ash_hawk/bridge/__init__.py` for structured diff output
- [x] Implement structured storage layout at `.ash-hawk/thin/{scenario_stem}/{run_id}/`
- [x] Add `--variant <name>` free-form string flag to `thin run` command
- [x] Expand hash scan from `.md`-only to all text file types (`.md`, `.py`, `.yaml`, `.json`, `.txt`, `.toml`, `.cfg`, `.ini`)
- [x] Implement `ash-hawk thin diff` command with auto-discover and explicit run ID modes
- [ ] Unify `_save_run_artifacts` to replace `_save_transcript_json` (two paths exist: `_save_run_artifacts_legacy()` in `cli/thin.py` and `_persist_run_artifacts()` in `scenario/thin_runner.py`)

## Pre-existing Bugs

- [x] Fix `GuardrailChecker` bug — `record_iteration()` never called with `reverted=True` when iterations are reverted (fixed: added revert tracking in cycle_runner.py)
- [x] `LeverMatrixSearch` has no `optimize()` method — moot after v0.2.0 evolvable phase replaced the optimization path

## Improver Agent Hardening

- [x] ~~Sanitize `grader_name` in `ImproverAgent._identify_target_files`~~ — `ImproverAgent` class does not exist in thev0.2.0` codebase (removed in legacy gut)
- [x] ~~Replace markdown code block wrapping with XML tags in `_build_improvement_prompt`~~ — method does not exist (closest equivalent is `generate_improvement()` in `llm.py`)
- [x] Fix `--backup/--no-backup` Click flag pattern in `improve_thin` — removed dead flag from deprecated command
- [x] ~~Wire real LLM client into `ImproverAgent._call_llm`~~ — deduplicated `_call_llm` across `llm.py` and `intent_analyzer.py` into single shared implementation

## ImprovementTarget Consolidation

- [x] Consolidate ImprovementTarget into `types.py` with merged fields (`dependencies`, `priority`) and methods (`delete_content`)
- [x] Remove duplicate class from `cycle_runner.py` and `target_discovery.py`
- [x] Update imports in test files to reference `types.py`
