# TODOS

## Provenance & Causality (Follow-up PR)

- [ ] Add `RunManifest` dataclass to `ash_hawk/bridge/__init__.py` for run attribution
- [ ] Add `DiffReport` dataclass to `ash_hawk/bridge/__init__.py` for structured diff output
- [ ] Implement structured storage layout at `.ash-hawk/thin/{scenario_stem}/{run_id}/`
- [ ] Add `--variant <name>` free-form string flag to `thin run` command
- [ ] Expand hash scan from `.md`-only to all text file types (`.md`, `.py`, `.yaml`, `.json`, `.txt`)
- [ ] Implement `ash-hawk thin diff` command with auto-discover and explicit run ID modes
- [ ] Unify `_save_run_artifacts` to replace `_save_transcript_json`

## Improvement Module Hardening (Follow-up PR)

- [ ] Add context verification to `DiffApplier._apply_patch` (verify removed lines match hunk context)
- [ ] Sanitize `grader_name` in `ImproverAgent._identify_target_files` to prevent path traversal
- [ ] Replace markdown code block wrapping with XML tags in `_build_improvement_prompt`
- [ ] Fix `--backup/--no-backup` Click flag pattern in `improve_thin`
- [ ] Wire real LLM client into `ImproverAgent._call_llm` (currently returns `None`)
