# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-03-31

### Added

- Provenance manifest (`ProvenanceManifest`) for tracking agent, scenario, and tooling metadata across runs
- Structured storage layer with scenario artifacts auto-saved to `.ash-hawk/thin/`
- Thin diff comparison between baseline and candidate runs with per-grader delta analysis
- Bolt-merlin specific graders (`BoltMerlinGraderPresets`) wired into eval pack system
- Dawn Kestrel bridge adapter for running dawn-kestrel agents through the thin bridge
- Experimental graders support in cycle runner (`experimental_graders` param on `_run_evaluation`)
- Additional fields on auto-research types for adversarial co-evolution integration points
- Scenario label helper for progress indicators in cycle runner
- Error recovery in cycle runner: sets final_score on error instead of re-raising

### Changed

- Refactored auto-research architecture to thin-wrapper pattern, removing legacy modules
- Removed legacy event system, template system, and curation store (18K+ lines of dead code)
- Simplified scenario runner to focus on thin scenario execution model
- Updated `ScenarioAdapterResult` to Pydantic model with 6 fields (was 4-tuple)

### Fixed

- Pre-existing test failure in `test_mock_adapter` — unpack `ScenarioAdapterResult` as model, not tuple
- Error message display in CLI when cycle run fails

## [0.1.1] - 2026-03-30

### Added

- Thin telemetry bridge for agent evaluation (`ash_hawk/bridge/`) with `TelemetrySink`, `TranscriptData`, `OutcomeData`, and `run_real_agent` entry point
- Dawn Kestrel bridge adapter (`ash_hawk/bridge/dawn_kestrel.py`) for running dawn-kestrel agents through the thin bridge
- Thin scenario runner (`ash_hawk/scenario/thin_runner.py`) with `ThinScenarioRunner`, `ScenarioTelemetrySink`, and `ThinGradedResult`
- Thin CLI commands (`ash_hawk/cli/thin.py`): `ash-hawk thin run` and `ash-hawk thin improve`
- Improvement module (`ash_hawk/improvement/`) with `ImproverAgent`, `DiffGenerator`, and `DiffApplier` for automated agent improvement
- LLM boolean grading support in thin run command
- Auto-research cycle integration with thin bridge (`ash_hawk/auto_research/cycle_runner.py`)
- Deprecation warnings on legacy types and curation store

### Fixed

- Run thin bridge scenarios concurrently and thread `trial_max_workers`
- Preserve registered LLM queue in `EvalRunner` and thread `--max-concurrent` to scenario parallelism

## [0.1.0] - 2026-03-30

### Added

- Initial release of Ash Hawk evaluation harness
- Structured task execution with deterministic, LLM-based, and composite graders
- CLI commands: `run`, `list`, `report`, `validate`
- Storage backends: file, SQLite, PostgreSQL, S3
- Agent adapters and policy enforcement
- Calibration analysis (ECE, Brier)
- HTML/JSON reporting
