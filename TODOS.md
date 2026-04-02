# TODOS

## Provenance & Causality (Follow-up PR)

- [ ] Add `RunManifest` dataclass to `ash_hawk/bridge/__init__.py` for run attribution
- [ ] Add `DiffReport` dataclass to `ash_hawk/bridge/__init__.py` for structured diff output
- [ ] Implement structured storage layout at `.ash-hawk/thin/{scenario_stem}/{run_id}/`
- [ ] Add `--variant <name>` free-form string flag to `thin run` command
- [ ] Expand hash scan from `.md`-only to all text file types (`.md`, `.py`, `.yaml`, `.json`, `.txt`)
- [ ] Implement `ash-hawk thin diff` command with auto-discover and explicit run ID modes
- [ ] Unify `_save_run_artifacts` to replace `_save_transcript_json`

## Pre-existing Bugs (Separate PR)

- [ ] Fix pre-existing `GuardrailChecker` bug (`record_iteration()` never calls `reverted=True`)
- [ ] `LeverMatrixSearch` has no `optimize()` method — enhanced runner silently skips

## Improver Agent Hardening

- [ ] Sanitize `grader_name` in `ImproverAgent._identify_target_files` to prevent path traversal
- [ ] Replace markdown code block wrapping with XML tags in `_build_improvement_prompt`
- [ ] Fix `--backup/--no-backup` Click flag pattern in `improve_thin`
- [ ] Wire real LLM client into `ImproverAgent._call_llm` (currently returns `None`)

## ImprovementTarget Consolidation

- [ ] Consolidate ImprovementTarget (move from `target_discovery.py` to `types.py`)
  - Both versions exist in `cycle_runner.py` and `target_discovery.py`
  - Fix `delete_content` path cleanup in `cycle_runner.py`
  - Add ImprovementTarget import to `types.py` from `target_discovery.py`
