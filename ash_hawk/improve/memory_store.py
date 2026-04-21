# type-hygiene: skip-file
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pydantic as pd


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _target_signature(target_files: list[str]) -> str:
    return "|".join(sorted(set(target_files)))


def _metadata_number(metadata: dict[str, object], key: str) -> float:
    value = metadata.get(key)
    return float(value) if isinstance(value, int | float) else 0.0


def _normalize_agent_name(raw: object) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "default"
    return raw.strip().replace("-", "_")


class WorkingSnapshot(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    run_id: str
    iteration: int
    active_trial_ids: list[str] = pd.Field(default_factory=list)
    active_families: list[str] = pd.Field(default_factory=list)
    active_target_files: list[str] = pd.Field(default_factory=list)
    hypothesis_count: int = pd.Field(ge=0, default=0)
    baseline_score: float | None = None
    baseline_pass_rate: float | None = None
    last_hypothesis_outcome: str | None = None
    memory_skip_count: int = pd.Field(ge=0, default=0)
    stop_reasons: list[str] = pd.Field(default_factory=list)
    updated_at: str = pd.Field(default_factory=_utc_now)


class EpisodeRecord(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    episode_id: str
    timestamp: str = pd.Field(default_factory=_utc_now)
    run_id: str
    agent_name: str = "default"
    iteration: int
    hypothesis_rank: int | None = None
    trial_id: str
    diagnosis_family: str = "unknown"
    target_files: list[str] = pd.Field(default_factory=list)
    outcome: str
    attempted: bool = False
    kept: bool | None = None
    confidence: float | None = pd.Field(default=None, ge=0.0, le=1.0)
    score_delta: float | None = None
    mutation_wall_seconds: float | None = pd.Field(default=None, ge=0.0)
    mutation_llm_calls: int | None = pd.Field(default=None, ge=0)
    retry_count: int = pd.Field(default=0, ge=0)
    metadata: dict[str, object] = pd.Field(default_factory=dict)


class SemanticRule(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    rule_id: str
    category: str
    diagnosis_family: str
    target_signature: str
    target_files: list[str] = pd.Field(default_factory=list)
    penalty: float = pd.Field(default=0.0, ge=0.0, le=1.0)
    boost: float = pd.Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: int = pd.Field(default=0, ge=0)
    source_run_id: str
    created_at: str = pd.Field(default_factory=_utc_now)
    updated_at: str = pd.Field(default_factory=_utc_now)
    notes: str | None = None


class PersonalPreference(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    key: str
    value: str
    updated_at: str = pd.Field(default_factory=_utc_now)


class MemoryConsolidation(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    run_id: str
    episodes_processed: int = pd.Field(default=0, ge=0)
    semantic_rules_added: int = pd.Field(default=0, ge=0)
    semantic_rules_updated: int = pd.Field(default=0, ge=0)
    groups_considered: int = pd.Field(default=0, ge=0)
    archived_episodes: int = pd.Field(default=0, ge=0)
    backfilled_episodes: int = pd.Field(default=0, ge=0)


class MemoryStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = (base_dir or Path(".ash-hawk/memory")).resolve()
        self._working_dir = self._base_dir / "working"
        self._episodic_dir = self._base_dir / "episodic"
        self._semantic_dir = self._base_dir / "semantic"
        self._personal_dir = self._base_dir / "personal"
        for directory in (
            self._working_dir,
            self._episodic_dir,
            self._semantic_dir,
            self._personal_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def save_working(self, snapshot: WorkingSnapshot) -> Path:
        path = self._working_dir / f"{snapshot.run_id}.json"
        path.write_text(
            json.dumps(snapshot.model_dump(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return path

    def append_episode(self, episode: EpisodeRecord) -> Path:
        path = self._episodic_dir / f"{episode.run_id}.jsonl"
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(episode.model_dump(), sort_keys=True) + "\n")
        return path

    def load_episodes(self, *, run_id: str | None = None) -> list[EpisodeRecord]:
        files = (
            [self._episodic_dir / f"{run_id}.jsonl"]
            if run_id
            else sorted(self._episodic_dir.glob("*.jsonl"))
        )
        episodes: list[EpisodeRecord] = []
        for file in files:
            if not file.exists():
                continue
            for line in file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                episodes.append(EpisodeRecord.model_validate_json(line))
        return episodes

    def repair_agent_scoping(self, improve_runs_root: Path) -> int:
        repaired = 0
        for episode_file in sorted(self._episodic_dir.glob("*.jsonl")):
            run_id = episode_file.stem
            config_path = improve_runs_root / run_id / "config.json"
            if not config_path.exists():
                continue
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            agent_name = _normalize_agent_name(config.get("agent_name"))
            if agent_name == "default":
                continue
            updated_lines: list[str] = []
            file_repaired = False
            for line in episode_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                episode = EpisodeRecord.model_validate_json(line)
                if episode.agent_name in {"default", "<missing>"}:
                    episode.agent_name = agent_name
                    repaired += 1
                    file_repaired = True
                updated_lines.append(json.dumps(episode.model_dump(), sort_keys=True))
            if file_repaired:
                episode_file.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
        return repaired

    def _rules_path(self) -> Path:
        return self._semantic_dir / "rules.json"

    def load_semantic_rules(self) -> list[SemanticRule]:
        path = self._rules_path()
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
        return [SemanticRule.model_validate(item) for item in raw]

    def save_semantic_rules(self, rules: list[SemanticRule]) -> Path:
        path = self._rules_path()
        path.write_text(
            json.dumps([rule.model_dump() for rule in rules], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def load_personal_preferences(self) -> list[PersonalPreference]:
        path = self._personal_dir / "preferences.json"
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
        return [PersonalPreference.model_validate(item) for item in raw]

    def save_personal_preferences(self, preferences: list[PersonalPreference]) -> Path:
        path = self._personal_dir / "preferences.json"
        path.write_text(
            json.dumps(
                [preference.model_dump() for preference in preferences], indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )
        return path

    def format_personal_preferences_for_prompt(self) -> str:
        preferences = self.load_personal_preferences()
        if not preferences:
            return ""
        lines = ["## Personal Memory"]
        for preference in preferences[:10]:
            lines.append(f"- {preference.key}: {preference.value}")
        return "\n".join(lines)

    def semantic_adjustment(
        self, diagnosis_family: str, target_files: list[str]
    ) -> tuple[float, float]:
        signature = _target_signature(target_files)
        penalty = 0.0
        boost = 0.0
        for rule in self.load_semantic_rules():
            if rule.diagnosis_family != diagnosis_family or rule.target_signature != signature:
                continue
            penalty += rule.penalty
            boost += rule.boost
        return min(0.5, penalty), min(0.4, boost)

    def _load_agent_name(self, run_dir: Path) -> str:
        config_path = run_dir / "config.json"
        if not config_path.exists():
            return "default"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "default"
        return _normalize_agent_name(config.get("agent_name"))

    def _import_completed_run(self, run_dir: Path, episodic_path: Path) -> tuple[int, bool]:
        run_json_path = run_dir / "run.json"
        if episodic_path.exists() or not run_json_path.exists():
            return (0, False)
        try:
            raw = json.loads(run_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return (0, False)
        mutation_history = raw.get("mutation_history", [])
        if not isinstance(mutation_history, list):
            return (0, False)
        agent_name = self._load_agent_name(run_dir)
        imported = 0
        for idx, entry in enumerate(mutation_history):
            if not isinstance(entry, dict):
                continue
            self.append_episode(
                EpisodeRecord(
                    episode_id=f"bf-{run_dir.name}-{idx}",
                    run_id=run_dir.name,
                    agent_name=agent_name,
                    iteration=int(entry.get("iteration", 0) or 0),
                    hypothesis_rank=int(entry.get("hypothesis_rank", 0) or 0) or None,
                    trial_id=str(entry.get("trial_id", f"bf-{idx}")),
                    diagnosis_family=str(entry.get("diagnosis_family", "unknown")),
                    target_files=list(entry.get("applied_files", []))
                    or list(entry.get("targeted_paths", [])),
                    outcome="kept"
                    if bool(entry.get("kept"))
                    else str(entry.get("rejection_reason") or "reverted"),
                    attempted=True,
                    kept=bool(entry.get("kept")),
                    confidence=None,
                    score_delta=float(entry.get("improvement", 0.0) or 0.0),
                    mutation_wall_seconds=(
                        float(entry.get("mutation_wall_seconds", 0.0))
                        if isinstance(entry.get("mutation_wall_seconds"), int | float)
                        else None
                    ),
                    mutation_llm_calls=(
                        int(entry.get("mutation_llm_calls", 0))
                        if isinstance(entry.get("mutation_llm_calls"), int | float)
                        else None
                    ),
                    metadata={"backfilled": True, "source_kind": "completed_run"},
                )
            )
            imported += 1
        return (imported, imported > 0)

    def _import_partial_run(self, run_dir: Path, episodic_path: Path) -> tuple[int, bool]:
        if episodic_path.exists():
            return (0, False)
        mutations_root = run_dir / "mutations"
        if not mutations_root.exists():
            return (0, False)
        agent_name = self._load_agent_name(run_dir)
        iteration_logs: dict[int, dict[str, object]] = {}
        logs_root = run_dir / "iteration_logs"
        if logs_root.exists():
            for log_path in sorted(logs_root.glob("iter-*.json")):
                try:
                    raw = json.loads(log_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(raw, dict):
                    iteration_logs[int(raw.get("iteration", 0) or 0)] = raw
        imported = 0
        seen: set[tuple[int, int, str]] = set()
        for mutation_path in sorted(mutations_root.glob("iter-*/rank-*.json")):
            iteration = int(mutation_path.parent.name.split("-")[-1])
            rank = int(mutation_path.stem.split("-")[-1])
            try:
                raw = json.loads(mutation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(raw, dict):
                continue
            patch = raw.get("patch", {}) if isinstance(raw.get("patch"), dict) else {}
            diagnosis = (
                patch.get("diagnosis", {}) if isinstance(patch.get("diagnosis"), dict) else {}
            )
            trial_id = str(
                raw.get("trial_id") or diagnosis.get("trial_id") or f"partial-{iteration}-{rank}"
            )
            key = (iteration, rank, trial_id)
            if key in seen:
                continue
            seen.add(key)
            iter_log = iteration_logs.get(iteration, {})
            outcome = (
                str(iter_log.get("hypothesis_outcome"))
                if str(iter_log.get("hypothesis_attempted") or "") == trial_id
                else "partial_backfill"
            )
            exec_metrics = (
                raw.get("execution_metrics", {})
                if isinstance(raw.get("execution_metrics"), dict)
                else {}
            )
            self.append_episode(
                EpisodeRecord(
                    episode_id=f"partial-{run_dir.name}-{iteration}-{rank}",
                    run_id=run_dir.name,
                    agent_name=agent_name,
                    iteration=iteration,
                    hypothesis_rank=rank,
                    trial_id=trial_id,
                    diagnosis_family=str(diagnosis.get("family", "unknown")),
                    target_files=list(raw.get("target_files", []))
                    or list(diagnosis.get("target_files", [])),
                    outcome=outcome,
                    attempted=True,
                    kept=outcome == "kept",
                    confidence=(
                        float(diagnosis.get("confidence", 0.0))
                        if isinstance(diagnosis.get("confidence"), int | float)
                        else None
                    ),
                    score_delta=_metadata_number(iter_log, "delta")
                    if isinstance(iter_log.get("delta"), int | float)
                    else None,
                    mutation_wall_seconds=(
                        float(exec_metrics.get("mean_llm_completion_seconds", 0.0))
                        if isinstance(exec_metrics.get("mean_llm_completion_seconds"), int | float)
                        else None
                    ),
                    mutation_llm_calls=(
                        int(exec_metrics.get("llm_completion_count", 0))
                        if isinstance(exec_metrics.get("llm_completion_count"), int | float)
                        else None
                    ),
                    metadata={
                        "backfilled": True,
                        "partial_backfill": True,
                        "source_kind": "partial_run",
                    },
                )
            )
            imported += 1
        return (imported, imported > 0)

    def _import_improve_cycle(self, root_dir: Path) -> tuple[int, bool]:
        cycle_path = root_dir / "improve-cycle" / "runs.json"
        if not cycle_path.exists():
            return (0, False)
        try:
            raw = json.loads(cycle_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return (0, False)
        if not isinstance(raw, list):
            return (0, False)
        imported = 0
        for idx, entry in enumerate(raw):
            if not isinstance(entry, dict):
                continue
            run_id = f"cycle-{entry.get('run_id', idx)}"
            episodic_path = self._episodic_dir / f"{run_id}.jsonl"
            if episodic_path.exists():
                continue
            score_metric = None
            metrics = entry.get("metrics", [])
            if isinstance(metrics, list):
                for metric in metrics:
                    if isinstance(metric, dict) and metric.get("name") == "score":
                        score_metric = metric
                        break
            self.append_episode(
                EpisodeRecord(
                    episode_id=f"cycle-{idx}",
                    run_id=run_id,
                    agent_name=_normalize_agent_name(entry.get("agent_id")),
                    iteration=0,
                    trial_id=str(entry.get("experiment_id", f"cycle-{idx}")),
                    diagnosis_family="improve_cycle_history",
                    target_files=[],
                    outcome="cycle_run",
                    attempted=False,
                    kept=None,
                    confidence=None,
                    score_delta=(
                        float(score_metric.get("delta", 0.0))
                        if isinstance(score_metric, dict)
                        and isinstance(score_metric.get("delta"), int | float)
                        else None
                    ),
                    metadata={"backfilled": True, "source_kind": "improve_cycle"},
                )
            )
            imported += 1
        return (imported, imported > 0)

    def backfill_from_run_artifacts(
        self,
        improve_runs_root: Path,
        *,
        force: bool = False,
        include_improve_cycle: bool = False,
    ) -> int:
        marker = self._base_dir / ".backfilled"
        if (marker.exists() and not force) or not improve_runs_root.exists():
            return 0
        imported = 0
        repaired = self.repair_agent_scoping(improve_runs_root)
        imported_runs: set[str] = set()
        for run_dir in sorted(p for p in improve_runs_root.iterdir() if p.is_dir()):
            episodic_path = self._episodic_dir / f"{run_dir.name}.jsonl"
            completed_imported, completed_used = self._import_completed_run(run_dir, episodic_path)
            imported += completed_imported
            if completed_used:
                imported_runs.add(run_dir.name)
                continue

            partial_imported, partial_used = self._import_partial_run(run_dir, episodic_path)
            imported += partial_imported
            if partial_used:
                imported_runs.add(run_dir.name)
        for run_id in imported_runs:
            self.consolidate_run(run_id)

        cycle_imported = 0
        cycle_used = False
        if include_improve_cycle:
            cycle_imported, cycle_used = self._import_improve_cycle(improve_runs_root.parent)
            imported += cycle_imported

        if imported_runs or repaired or force or cycle_used:
            self.consolidate_all()
        marker.write_text(str(imported + repaired), encoding="utf-8")
        return imported + repaired

    def decay_and_archive_episodes(self, cutoff_days: int = 30) -> int:
        archive_dir = self._episodic_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived = 0
        cutoff = datetime.now(UTC)
        for episode_file in sorted(self._episodic_dir.glob("*.jsonl")):
            keep_lines: list[str] = []
            archive_lines: list[str] = []
            for line in episode_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                episode = EpisodeRecord.model_validate_json(line)
                age_days = (cutoff - datetime.fromisoformat(episode.timestamp)).days
                low_signal = (
                    episode.outcome
                    in {"no_file_changes", "mutation_cli_timeout", "low_conversion_from_memory"}
                    and not episode.kept
                    and (episode.score_delta is None or abs(episode.score_delta) < 0.01)
                )
                if age_days >= cutoff_days and low_signal:
                    archive_lines.append(line)
                    archived += 1
                else:
                    keep_lines.append(line)
            if archive_lines:
                archive_path = archive_dir / episode_file.name
                with archive_path.open("a", encoding="utf-8") as file:
                    for line in archive_lines:
                        file.write(line + "\n")
                if keep_lines:
                    episode_file.write_text("\n".join(keep_lines) + "\n", encoding="utf-8")
                else:
                    episode_file.unlink()
        return archived

    def observability_metrics(self) -> dict[str, float | int]:
        episodes = self.load_episodes()
        if not episodes:
            return {
                "skip_precision": 0.0,
                "promotion_hit_rate": 0.0,
                "semantic_rule_utilization_rate": 0.0,
            }
        skip_count = sum(
            1 for episode in episodes if episode.outcome == "low_conversion_from_memory"
        )
        attempted_count = sum(1 for episode in episodes if episode.attempted)
        influenced_attempts = sum(
            bool(
                episode.attempted
                and (
                    _metadata_number(episode.metadata, "semantic_penalty") > 0.0
                    or _metadata_number(episode.metadata, "semantic_boost") > 0.0
                )
            )
            for episode in episodes
        )
        promoted_attempts = sum(
            bool(
                episode.attempted
                and episode.kept
                and _metadata_number(episode.metadata, "semantic_boost") > 0.0
            )
            for episode in episodes
        )
        precise_skips = sum(
            bool(
                episode.outcome == "low_conversion_from_memory"
                and _metadata_number(episode.metadata, "adjusted_success_rate") < 0.15
            )
            for episode in episodes
        )
        return {
            "skip_precision": precise_skips / max(1, skip_count),
            "promotion_hit_rate": promoted_attempts / max(1, influenced_attempts),
            "semantic_rule_utilization_rate": influenced_attempts / max(1, attempted_count),
        }

    def should_skip_hypothesis(
        self,
        *,
        agent_name: str,
        diagnosis_family: str,
        target_files: list[str],
        min_attempts: int = 5,
        min_success_rate: float = 0.15,
    ) -> tuple[bool, str | None, dict[str, float | int]]:
        signature = _target_signature(target_files)
        episodes = [
            e
            for e in self.load_episodes()
            if e.agent_name == agent_name
            and e.diagnosis_family == diagnosis_family
            and _target_signature(e.target_files) == signature
            and e.outcome != "low_conversion_from_memory"
        ]
        attempt_episodes = [e for e in episodes if e.attempted]
        attempts = len(attempt_episodes)
        if attempts < min_attempts:
            return False, None, {"attempts": attempts, "success_rate": 0.0, "timeout_rate": 0.0}

        kept_count = sum(1 for e in attempt_episodes if e.kept)
        timeout_count = sum(1 for e in attempt_episodes if e.outcome == "mutation_cli_timeout")
        no_change_count = sum(1 for e in attempt_episodes if e.outcome == "no_file_changes")
        success_rate = kept_count / attempts
        timeout_rate = timeout_count / attempts
        no_change_rate = no_change_count / attempts

        semantic_penalty = sum(
            rule.penalty
            for rule in self.load_semantic_rules()
            if rule.diagnosis_family == diagnosis_family
            and rule.target_signature == signature
            and rule.category.startswith("friction")
        )
        adjusted_success_rate = max(0.0, success_rate - min(0.4, semantic_penalty))

        if adjusted_success_rate < min_success_rate and (
            timeout_rate >= 0.5 or no_change_rate >= 0.5
        ):
            reason = "low_conversion_from_memory"
            return (
                True,
                reason,
                {
                    "attempts": attempts,
                    "success_rate": success_rate,
                    "adjusted_success_rate": adjusted_success_rate,
                    "timeout_rate": timeout_rate,
                    "no_change_rate": no_change_rate,
                    "semantic_penalty": semantic_penalty,
                },
            )
        return (
            False,
            None,
            {
                "attempts": attempts,
                "success_rate": success_rate,
                "adjusted_success_rate": adjusted_success_rate,
                "timeout_rate": timeout_rate,
                "no_change_rate": no_change_rate,
                "semantic_penalty": semantic_penalty,
            },
        )

    def calibration_factor(
        self,
        diagnosis_family: str,
        *,
        agent_name: str = "default",
        min_samples: int = 5,
    ) -> float:
        samples = [
            episode
            for episode in self.load_episodes()
            if episode.agent_name == agent_name
            and episode.diagnosis_family == diagnosis_family
            and episode.confidence is not None
            and episode.score_delta is not None
        ]
        if len(samples) < min_samples:
            return 1.0
        mean_conf = sum(sample.confidence or 0.0 for sample in samples) / len(samples)
        mean_positive_delta = sum(max(0.0, sample.score_delta or 0.0) for sample in samples) / len(
            samples
        )
        if mean_conf <= 0.0:
            return 1.0
        return max(0.25, min(1.5, mean_positive_delta / mean_conf))

    def _consolidate_entries(
        self,
        entries: list[EpisodeRecord],
        *,
        source_run_id: str,
    ) -> MemoryConsolidation:
        grouped: dict[tuple[str, str], list[EpisodeRecord]] = {}
        for episode in entries:
            key = (episode.diagnosis_family, _target_signature(episode.target_files))
            grouped.setdefault(key, []).append(episode)

        rules = self.load_semantic_rules()
        by_key = {
            (rule.category, rule.diagnosis_family, rule.target_signature): rule for rule in rules
        }
        added = 0
        updated = 0
        now = _utc_now()

        for (family, signature), family_entries in grouped.items():
            if len(family_entries) < 3:
                continue
            if family == "unknown" or not signature:
                continue
            timeout_rate = sum(
                1 for entry in family_entries if entry.outcome == "mutation_cli_timeout"
            ) / len(family_entries)
            no_change_rate = sum(
                1 for entry in family_entries if entry.outcome == "no_file_changes"
            ) / len(family_entries)
            kept_rate = sum(1 for entry in family_entries if entry.kept) / len(family_entries)
            target_files = family_entries[0].target_files

            def upsert_rule(
                category: str, penalty: float = 0.0, boost: float = 0.0, notes: str = ""
            ) -> None:
                nonlocal added, updated
                rule_key = (category, family, signature)
                existing = by_key.get(rule_key)
                if existing is None:
                    rule = SemanticRule(
                        rule_id=f"{category}-{family}-{abs(hash(signature)) % 10_000_000}",
                        category=category,
                        diagnosis_family=family,
                        target_signature=signature,
                        target_files=target_files,
                        penalty=penalty,
                        boost=boost,
                        evidence_count=len(family_entries),
                        source_run_id=source_run_id,
                        notes=notes,
                    )
                    rules.append(rule)
                    by_key[rule_key] = rule
                    added += 1
                else:
                    existing.penalty = max(existing.penalty, penalty)
                    existing.boost = max(existing.boost, boost)
                    existing.evidence_count = max(existing.evidence_count, len(family_entries))
                    existing.source_run_id = source_run_id
                    existing.updated_at = now
                    existing.notes = notes
                    updated += 1

            confidence_samples = [
                entry.confidence for entry in family_entries if entry.confidence is not None
            ]
            confidence_mean = (
                sum(confidence_samples) / len(confidence_samples) if confidence_samples else 0.0
            )
            if no_change_rate >= 0.5 and (confidence_mean >= 0.4 or len(family_entries) >= 5):
                upsert_rule(
                    "friction_no_change",
                    penalty=min(0.5, 0.2 + no_change_rate * 0.4),
                    notes="High no_file_changes recurrence",
                )
            if timeout_rate >= 0.5 and (confidence_mean >= 0.4 or len(family_entries) >= 5):
                upsert_rule(
                    "friction_timeout",
                    penalty=min(0.5, 0.2 + timeout_rate * 0.4),
                    notes="High mutation timeout recurrence",
                )
            if kept_rate >= 0.4 and len(family_entries) >= 4:
                upsert_rule(
                    "success_pattern",
                    boost=min(0.3, kept_rate * 0.3),
                    notes="Pattern has recurring keeps",
                )

        self.save_semantic_rules(rules)
        return MemoryConsolidation(
            run_id=source_run_id,
            episodes_processed=len(entries),
            semantic_rules_added=added,
            semantic_rules_updated=updated,
            groups_considered=len(grouped),
        )

    def consolidate_all(self) -> MemoryConsolidation:
        consolidation = self._consolidate_entries(self.load_episodes(), source_run_id="global")
        consolidation.archived_episodes = self.decay_and_archive_episodes()
        return consolidation

    def consolidate_run(self, run_id: str) -> MemoryConsolidation:
        consolidation = self._consolidate_entries(
            self.load_episodes(run_id=run_id),
            source_run_id=run_id,
        )
        consolidation.archived_episodes = self.decay_and_archive_episodes()
        return consolidation
