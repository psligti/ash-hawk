from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from ash_hawk.improve.memory_store import (
    EpisodeRecord,
    MemoryStore,
    PersonalPreference,
    WorkingSnapshot,
)


class TestMemoryStore:
    def test_writes_working_and_episodic_layers(self, tmp_path: Path) -> None:
        store = MemoryStore(base_dir=tmp_path / "memory")

        working_path = store.save_working(
            WorkingSnapshot(
                run_id="run-1",
                iteration=0,
                active_trial_ids=["trial-1"],
                active_families=["tool_use"],
                active_target_files=["prompt.md"],
                hypothesis_count=2,
                baseline_score=0.4,
                baseline_pass_rate=0.5,
                last_hypothesis_outcome="targeted_regression",
                memory_skip_count=1,
            )
        )
        episode_path = store.append_episode(
            EpisodeRecord(
                episode_id="ep-1",
                run_id="run-1",
                iteration=0,
                trial_id="trial-1",
                diagnosis_family="tool_use",
                target_files=["prompt.md"],
                outcome="no_file_changes",
                attempted=False,
            )
        )

        assert working_path.exists()
        assert '"active_families": [' in working_path.read_text(encoding="utf-8")
        assert episode_path.exists()
        episodes = store.load_episodes(run_id="run-1")
        assert len(episodes) == 1
        assert episodes[0].outcome == "no_file_changes"

    def test_should_skip_hypothesis_after_repeated_low_conversion(self, tmp_path: Path) -> None:
        store = MemoryStore(base_dir=tmp_path / "memory")

        for idx in range(6):
            store.append_episode(
                EpisodeRecord(
                    episode_id=f"ep-{idx}",
                    run_id="run-a",
                    agent_name="build",
                    iteration=0,
                    trial_id=f"trial-{idx}",
                    diagnosis_family="delegation",
                    target_files=["orchestrator.py"],
                    outcome="mutation_cli_timeout" if idx % 2 == 0 else "no_file_changes",
                    attempted=True,
                    kept=False,
                )
            )

        should_skip, reason, metrics = store.should_skip_hypothesis(
            agent_name="build",
            diagnosis_family="delegation",
            target_files=["orchestrator.py"],
        )

        assert should_skip is True
        assert reason == "low_conversion_from_memory"
        assert int(metrics["attempts"]) == 6

    def test_consolidates_semantic_rules(self, tmp_path: Path) -> None:
        store = MemoryStore(base_dir=tmp_path / "memory")

        for idx in range(5):
            store.append_episode(
                EpisodeRecord(
                    episode_id=f"ep-{idx}",
                    run_id="run-z",
                    agent_name="build",
                    iteration=idx,
                    trial_id=f"trial-{idx}",
                    diagnosis_family="verification",
                    target_files=["execute.py"],
                    outcome="no_file_changes",
                    attempted=False,
                    confidence=0.8,
                )
            )

        summary = store.consolidate_run("run-z")
        rules = store.load_semantic_rules()

        assert summary.episodes_processed == 5
        assert summary.semantic_rules_added >= 1
        assert any(rule.category == "friction_no_change" for rule in rules)

    def test_consolidates_repeated_patterns_without_confidence_when_evidence_is_strong(
        self, tmp_path: Path
    ) -> None:
        store = MemoryStore(base_dir=tmp_path / "memory")

        for idx in range(5):
            store.append_episode(
                EpisodeRecord(
                    episode_id=f"ep-nc-{idx}",
                    run_id="run-strong",
                    agent_name="bolt_merlin",
                    iteration=idx,
                    trial_id=f"trial-{idx}",
                    diagnosis_family="workspace_path_resolution",
                    target_files=["execute.py"],
                    outcome="no_file_changes",
                    attempted=False,
                )
            )

        summary = store.consolidate_run("run-strong")

        assert summary.semantic_rules_added >= 1
        assert any(rule.target_files == ["execute.py"] for rule in store.load_semantic_rules())

    def test_consolidate_all_promotes_cross_run_patterns(self, tmp_path: Path) -> None:
        store = MemoryStore(base_dir=tmp_path / "memory")

        for idx in range(3):
            store.append_episode(
                EpisodeRecord(
                    episode_id=f"ep-a-{idx}",
                    run_id="run-a",
                    agent_name="bolt_merlin",
                    iteration=idx,
                    trial_id=f"trial-a-{idx}",
                    diagnosis_family="workspace_path_resolution",
                    target_files=["execute.py"],
                    outcome="no_file_changes",
                    attempted=False,
                )
            )
        for idx in range(2):
            store.append_episode(
                EpisodeRecord(
                    episode_id=f"ep-b-{idx}",
                    run_id="run-b",
                    agent_name="bolt_merlin",
                    iteration=idx,
                    trial_id=f"trial-b-{idx}",
                    diagnosis_family="workspace_path_resolution",
                    target_files=["execute.py"],
                    outcome="no_file_changes",
                    attempted=False,
                )
            )

        summary = store.consolidate_all()

        assert summary.semantic_rules_added >= 1
        assert any(rule.category == "friction_no_change" for rule in store.load_semantic_rules())

    def test_calibration_factor_uses_episode_outcomes(self, tmp_path: Path) -> None:
        store = MemoryStore(base_dir=tmp_path / "memory")

        for idx in range(6):
            store.append_episode(
                EpisodeRecord(
                    episode_id=f"ep-{idx}",
                    run_id="run-c",
                    agent_name="build",
                    iteration=idx,
                    trial_id=f"trial-{idx}",
                    diagnosis_family="delegation",
                    target_files=["task.py"],
                    outcome="kept" if idx % 2 == 0 else "reverted",
                    attempted=True,
                    kept=idx % 2 == 0,
                    confidence=0.8,
                    score_delta=0.2 if idx % 2 == 0 else -0.1,
                )
            )

        factor = store.calibration_factor("delegation", agent_name="build")
        assert 0.25 <= factor <= 1.5

    def test_save_and_load_personal_preferences(self, tmp_path: Path) -> None:
        store = MemoryStore(base_dir=tmp_path / "memory")
        store.save_personal_preferences(
            [PersonalPreference(key="prefer_small_changes", value="true")]
        )

        loaded = store.load_personal_preferences()
        assert loaded[0].key == "prefer_small_changes"
        assert "Personal Memory" in store.format_personal_preferences_for_prompt()

    def test_archive_old_episodes_moves_low_signal_entries(self, tmp_path: Path) -> None:
        store = MemoryStore(base_dir=tmp_path / "memory")
        stale_timestamp = (datetime.now(UTC) - timedelta(days=45)).isoformat()
        store.append_episode(
            EpisodeRecord(
                episode_id="ep-old",
                timestamp=stale_timestamp,
                run_id="run-old",
                agent_name="build",
                iteration=0,
                trial_id="trial-old",
                diagnosis_family="tool_use",
                target_files=["prompt.md"],
                outcome="no_file_changes",
                attempted=False,
                kept=False,
                score_delta=0.0,
            )
        )

        archived = store.decay_and_archive_episodes(cutoff_days=30)

        assert archived == 1
        assert not store.load_episodes(run_id="run-old")

    def test_backfill_from_run_artifacts_imports_mutation_history(self, tmp_path: Path) -> None:
        store = MemoryStore(base_dir=tmp_path / "memory")
        runs_dir = tmp_path / "improve-runs"
        run_dir = runs_dir / "improve-abc"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text('{"agent_name":"bolt_merlin"}')
        (run_dir / "run.json").write_text(
            '{"mutation_history":[{"iteration":0,"hypothesis_rank":1,"trial_id":"trial-1","diagnosis_family":"delegation","applied_files":["task.py"],"kept":true,"improvement":0.2,"mutation_wall_seconds":12.0,"mutation_llm_calls":3}]}'
        )

        imported = store.backfill_from_run_artifacts(runs_dir)

        assert imported == 1
        episodes = store.load_episodes(run_id="improve-abc")
        assert episodes[0].outcome == "kept"
        assert episodes[0].agent_name == "bolt_merlin"

    def test_backfill_from_partial_run_imports_mutation_artifacts(self, tmp_path: Path) -> None:
        store = MemoryStore(base_dir=tmp_path / "memory")
        runs_dir = tmp_path / "improve-runs"
        run_dir = runs_dir / "improve-partial"
        (run_dir / "mutations" / "iter-000").mkdir(parents=True)
        (run_dir / "iteration_logs").mkdir(parents=True)
        (run_dir / "config.json").write_text('{"agent_name":"bolt_merlin"}')
        (run_dir / "iteration_logs" / "iter-000.json").write_text(
            '{"iteration":0,"hypothesis_attempted":"trial-1","hypothesis_outcome":"targeted_regression","delta":-0.1}'
        )
        (run_dir / "mutations" / "iter-000" / "rank-1.json").write_text(
            '{"trial_id":"trial-1","target_files":["execute.py"],"execution_metrics":{"llm_completion_count":5,"mean_llm_completion_seconds":7.5},"patch":{"diagnosis":{"family":"verification_honesty","trial_id":"trial-1","target_files":["execute.py"],"confidence":0.8}}}'
        )

        imported = store.backfill_from_run_artifacts(runs_dir)

        assert imported == 1
        episodes = store.load_episodes(run_id="improve-partial")
        assert episodes[0].outcome == "targeted_regression"
        assert episodes[0].agent_name == "bolt_merlin"
        assert episodes[0].metadata["source_kind"] == "partial_run"

    def test_backfill_from_improve_cycle_ingests_flat_runs(self, tmp_path: Path) -> None:
        store = MemoryStore(base_dir=tmp_path / "memory")
        root = tmp_path / ".ash-hawk"
        runs_dir = root / "improve-runs"
        runs_dir.mkdir(parents=True)
        cycle_dir = root / "improve-cycle"
        cycle_dir.mkdir(parents=True)
        (cycle_dir / "runs.json").write_text(
            '[{"run_id":"run-1","experiment_id":"exp-1","agent_id":"bolt-merlin","metrics":[{"name":"score","value":0.4,"delta":0.1}]}]'
        )

        imported = store.backfill_from_run_artifacts(
            runs_dir, force=True, include_improve_cycle=True
        )

        assert imported == 1
        episodes = store.load_episodes(run_id="cycle-run-1")
        assert episodes[0].agent_name == "bolt_merlin"
        assert episodes[0].metadata["source_kind"] == "improve_cycle"

    def test_repair_agent_scoping_updates_default_episodes(self, tmp_path: Path) -> None:
        store = MemoryStore(base_dir=tmp_path / "memory")
        runs_dir = tmp_path / "improve-runs"
        run_dir = runs_dir / "improve-xyz"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text('{"agent_name":"bolt_merlin"}')
        store.append_episode(
            EpisodeRecord(
                episode_id="ep-1",
                run_id="improve-xyz",
                agent_name="default",
                iteration=0,
                trial_id="trial-1",
                diagnosis_family="workspace_path_resolution",
                target_files=["execute.py"],
                outcome="targeted_regression",
                attempted=True,
                kept=False,
            )
        )

        repaired = store.repair_agent_scoping(runs_dir)

        assert repaired == 1
        assert store.load_episodes(run_id="improve-xyz")[0].agent_name == "bolt_merlin"

    def test_observability_metrics_reflect_semantic_usage(self, tmp_path: Path) -> None:
        store = MemoryStore(base_dir=tmp_path / "memory")
        store.append_episode(
            EpisodeRecord(
                episode_id="ep-1",
                run_id="run-1",
                agent_name="build",
                iteration=0,
                trial_id="trial-1",
                diagnosis_family="delegation",
                target_files=["task.py"],
                outcome="kept",
                attempted=True,
                kept=True,
                score_delta=0.2,
                metadata={"semantic_penalty": 0.1},
            )
        )

        metrics = store.observability_metrics()
        assert metrics["semantic_rule_utilization_rate"] == 1.0
