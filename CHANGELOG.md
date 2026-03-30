# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
