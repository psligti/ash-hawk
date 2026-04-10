from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ash_hawk.improve.lesson_store import LessonStore
    from ash_hawk.types import EvalTrial

EXPLORER_DIAGNOSIS_PROMPT = """Investigate the failed evaluation trial using read-only exploration.

Do not modify files.

Start from these paths:
- agent root: {agent_root}
- trial bundle: {trial_bundle}
- lessons dir: {lessons_dir}

Read the bundle, inspect relevant code, prompts, skills, traces, and lessons, then return at least 5 distinct diagnosis ideas whenever the evidence supports it.

Return only JSON in this shape:
{{
  "ideas": [
    {{
      "failure_summary": "one-line summary",
      "root_cause": "detailed root cause analysis grounded in evidence",
      "suggested_fix": "concrete fix suggestion",
      "target_files": ["relative/path.py"],
      "confidence": 0.8
    }}
  ]
}}"""


def _repo_root_from_agent_path(agent_path: Path) -> Path:
    return agent_path.parent.parent


def _explorer_config_path(repo_root: Path) -> Path:
    return repo_root / ".dawn-kestrel" / "explorer_config.yaml"


def _lessons_dir(lesson_store: LessonStore | None, repo_root: Path) -> Path:
    if lesson_store is not None:
        lessons_dir = getattr(lesson_store, "_lessons_dir", None)
        if isinstance(lessons_dir, Path):
            return lessons_dir
    return repo_root / ".ash-hawk" / "lessons"


def _write_trial_bundle(temp_dir: Path, trial: EvalTrial) -> Path:
    bundle_path = temp_dir / "trial.json"
    bundle_path.write_text(json.dumps(trial.model_dump(mode="json"), indent=2), encoding="utf-8")
    return bundle_path


async def investigate_trial_with_explorer(
    trial: EvalTrial,
    agent_path: Path,
    lesson_store: LessonStore | None = None,
) -> str | None:
    from ash_hawk.improve.patch import run_agent_cli

    repo_root = _repo_root_from_agent_path(agent_path)
    config_path = _explorer_config_path(repo_root)
    if not config_path.exists():
        return None

    with tempfile.TemporaryDirectory(prefix=f"ash-hawk-diagnose-{trial.id}-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        bundle_path = _write_trial_bundle(temp_dir, trial)
        prompt = EXPLORER_DIAGNOSIS_PROMPT.format(
            agent_root=str(agent_path.relative_to(repo_root)),
            trial_bundle=str(bundle_path),
            lessons_dir=str(_lessons_dir(lesson_store, repo_root)),
        )
        stdout, error = await run_agent_cli(
            prompt=prompt,
            cwd=repo_root,
            config_path=config_path,
            agent_name="explorer",
        )
        if error is not None:
            return None
        return stdout


__all__ = ["investigate_trial_with_explorer"]
