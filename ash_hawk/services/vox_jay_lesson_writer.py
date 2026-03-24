from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ash_hawk.contracts import CuratedLesson
from ash_hawk.curation.experiment_store import ExperimentStore

logger = logging.getLogger(__name__)


@dataclass
class VoxJayLessonWriteResult:
    experiments_read: list[str]
    approved_lessons_seen: int
    lessons_written: int
    lessons_skipped_existing: int
    lessons_skipped_unmapped: int
    lessons_skipped_non_actionable: int
    writes_by_target: dict[str, int]


class VoxJayLessonWriter:
    _VOICE_TARGET = Path("src/vox_jay/assets/voice.md")
    _POLICY_TARGET = Path("src/vox_jay/assets/policy.md")
    _STRATEGY_TARGET = Path("src/vox_jay/assets/strategy.md")

    def __init__(
        self,
        experiments_root: Path = Path(".ash-hawk/experiments"),
        vox_jay_root: Path = Path("."),
    ) -> None:
        self._experiments_root = experiments_root
        self._vox_jay_root = vox_jay_root
        self._store = ExperimentStore(base_path=experiments_root.parent)

    def apply(
        self,
        *,
        experiment_id: str | None = None,
        dry_run: bool = False,
    ) -> VoxJayLessonWriteResult:
        experiments = self._resolve_experiments(experiment_id)
        lessons: list[CuratedLesson] = []
        for exp_id in experiments:
            lessons.extend(self._store.list_all(exp_id, status="approved"))

        lessons.sort(key=lambda lesson: lesson.created_at)

        writes_by_target: dict[str, int] = {}
        existing_markers: dict[str, set[str]] = {}
        pending_content: dict[str, list[str]] = {}

        lessons_written = 0
        lessons_skipped_existing = 0
        lessons_skipped_unmapped = 0
        lessons_skipped_non_actionable = 0

        for lesson in lessons:
            target = self._target_for_lesson(lesson)
            if target is None:
                lessons_skipped_unmapped += 1
                continue

            target_path = self._vox_jay_root / target
            target_key = str(target)
            marker = self._lesson_marker(lesson.lesson_id)

            if target_key not in existing_markers:
                existing_markers[target_key] = self._read_existing_markers(target_path)

            if marker in existing_markers[target_key]:
                lessons_skipped_existing += 1
                continue

            section = self._render_lesson_section(lesson)
            if section is None:
                lessons_skipped_non_actionable += 1
                continue
            pending_content.setdefault(target_key, []).append(section)
            existing_markers[target_key].add(marker)
            writes_by_target[target_key] = writes_by_target.get(target_key, 0) + 1
            lessons_written += 1

        if not dry_run:
            for target_key, sections in pending_content.items():
                target_path = self._vox_jay_root / Path(target_key)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                prefix = ""
                if target_path.exists() and target_path.read_text(encoding="utf-8").strip():
                    prefix = "\n\n"
                with open(target_path, "a", encoding="utf-8") as file_handle:
                    file_handle.write(prefix + "\n\n".join(sections))

        return VoxJayLessonWriteResult(
            experiments_read=experiments,
            approved_lessons_seen=len(lessons),
            lessons_written=lessons_written,
            lessons_skipped_existing=lessons_skipped_existing,
            lessons_skipped_unmapped=lessons_skipped_unmapped,
            lessons_skipped_non_actionable=lessons_skipped_non_actionable,
            writes_by_target=writes_by_target,
        )

    def _resolve_experiments(self, experiment_id: str | None) -> list[str]:
        if experiment_id:
            return [experiment_id]

        if not self._experiments_root.exists():
            return []

        candidates = [path for path in self._experiments_root.iterdir() if path.is_dir()]
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            return []
        return [candidates[0].name]

    def _target_for_lesson(self, lesson: CuratedLesson) -> Path | None:
        if lesson.lesson_type == "policy":
            return self._POLICY_TARGET
        if lesson.lesson_type != "skill":
            return None

        if self._matches_sub_strategy(lesson, "voice-tone"):
            return self._VOICE_TARGET
        if self._matches_sub_strategy(lesson, "playbook-adherence"):
            return self._STRATEGY_TARGET

        combined = " ".join(
            [lesson.title, lesson.description, json.dumps(lesson.lesson_payload, sort_keys=True)]
        ).lower()
        if "voice" in combined or "tone" in combined:
            return self._VOICE_TARGET
        if "playbook" in combined:
            return self._STRATEGY_TARGET

        return None

    def _matches_sub_strategy(self, lesson: CuratedLesson, expected: str) -> bool:
        for sub_strategy in lesson.sub_strategies:
            if str(sub_strategy) == expected:
                return True
        return False

    def _read_existing_markers(self, target_path: Path) -> set[str]:
        if not target_path.exists():
            return set()

        content = target_path.read_text(encoding="utf-8")
        markers: set[str] = set()
        for line in content.splitlines():
            if line.startswith("<!-- ash-hawk-lesson:") and line.endswith(" -->"):
                markers.add(line.strip())
        return markers

    def _lesson_marker(self, lesson_id: str) -> str:
        return f"<!-- ash-hawk-lesson:{lesson_id} -->"

    def _render_lesson_section(self, lesson: CuratedLesson) -> str | None:
        if lesson.lesson_type == "policy":
            return self._render_policy_section(lesson)

        lines = [
            self._lesson_marker(lesson.lesson_id),
            f"## {lesson.title}",
            f"- lesson_id: `{lesson.lesson_id}`",
            f"- source_proposal_id: `{lesson.source_proposal_id}`",
            f"- lesson_type: `{lesson.lesson_type}`",
        ]
        if lesson.strategy is not None:
            lines.append(f"- strategy: `{lesson.strategy}`")
        if lesson.sub_strategies:
            values = ", ".join(f"`{sub}`" for sub in lesson.sub_strategies)
            lines.append(f"- sub_strategies: {values}")

        lines.append("")
        lines.append(lesson.description)

        if lesson.lesson_payload:
            payload_text = json.dumps(lesson.lesson_payload, indent=2, sort_keys=True, default=str)
            lines.extend(["", "```json", payload_text, "```"])
        return "\n".join(lines)

    def _render_policy_section(self, lesson: CuratedLesson) -> str | None:
        fields = self._extract_policy_fields(lesson.lesson_payload)
        if not fields:
            logger.debug("Policy lesson %s has no actionable fields in payload", lesson.lesson_id)
            return None

        lines = [
            self._lesson_marker(lesson.lesson_id),
            "## Policy Rule Update",
            f"- lesson_id: `{lesson.lesson_id}`",
            f"- source_proposal_id: `{lesson.source_proposal_id}`",
        ]

        lines.append("")
        lines.append("Actionable policy fields:")
        for key, value in fields:
            lines.append(f"- {key}: `{value}`")
        return "\n".join(lines)

    def _render_policy_payload(
        self,
        lesson_id: str,
        lesson_payload: dict[str, object],
    ) -> list[str]:
        fields = self._extract_policy_fields(lesson_payload)
        if not fields:
            logger.debug("Policy lesson %s has no actionable fields in payload", lesson_id)
            return []

        rendered = ["", "Actionable policy fields:"]
        for key, value in fields:
            rendered.append(f"- {key}: `{value}`")
        return rendered

    def _extract_policy_fields(self, lesson_payload: dict[str, object]) -> list[tuple[str, str]]:
        fields: list[tuple[str, str]] = []
        for key in ["rule_name", "rule_type", "condition", "action", "priority"]:
            value = lesson_payload.get(key)
            if value is None:
                continue
            text_value = str(value).strip()
            if not text_value:
                continue
            fields.append((key, text_value))
        return fields
