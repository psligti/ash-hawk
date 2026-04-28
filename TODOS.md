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

## CEO Review Deferred Platform Work

- [ ] Build challenger-set / adversarial scenario growth engine after the V1 thin_runtime improver is stable. Why: stop long-term overfitting by turning recurring failures into new stress scenarios. Pros: stronger generalization and future rigor. Cons: large scope and more eval curation complexity. Priority: P2. Blocked by: V1 route fix, frozen eval manifests, regression memory, and stable operator reporting.
- [ ] Build a full multi-agent improvement bus after the single-agent loop is proven. Why: avoid duplicate improvement systems and turn the feature into shared agent infrastructure. Pros: long-term platform leverage and shared lessons across agents. Cons: highest coupling risk on the biggest churn area. Priority: P2. Blocked by: proven single-agent loop, stable mutation path, shared stop/acceptance primitives, and clearer ownership between `thin_runtime` and `improve/loop.py`.
- [ ] Build async experiment orchestration beyond the single-run operator flow after V1 loop semantics are trustworthy. Why: scale from foreground manual runs to managed improvement workloads. Pros: throughput, resumability, future automation. Cons: more state-management complexity and more failure modes. Priority: P3. Blocked by: stable V1 loop semantics, frozen eval artifacts, regression memory, and trustworthy stop/revert behavior.

## Eng Review Deferred V1 Work

- [ ] Add regression memory + golden-set guard after the reduced thin_runtime V1 trust foundation is proven. Why: turn one-off fixes into cumulative learning without adding a second truth surface too early. Pros: prevents relearning the same failures and strengthens future generalization. Cons: premature introduction would increase complexity before the safe mutation + manifest foundation is trustworthy. Priority: P2. Blocked by: isolated-workspace mutation, explicit planner demotion, minimal frozen eval manifest, and full harness/eval test coverage.
- [ ] Add operator-facing improvement timeline/reporting after the reduced thin_runtime V1 core is stable. Why: human visibility matters, but it should describe trustworthy loop semantics instead of unreliable pre-foundation behavior. Pros: better debuggability and operator trust once the core loop is real. Cons: if added too early, it becomes dashboard theater and a duplicate truth surface. Priority: P2. Blocked by: isolated-workspace mutation, explicit planner demotion, minimal frozen eval manifest, and one trustworthy end-to-end harness test.
