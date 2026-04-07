# type-hygiene: skip-file
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ash_hawk.improve.diagnose import Diagnosis

logger = logging.getLogger(__name__)


@dataclass
class ProposedPatch:
    diagnosis: Diagnosis
    file_path: str
    description: str
    diff: str
    rationale: str


async def propose_patch(diagnosis: Diagnosis) -> ProposedPatch:
    from ash_hawk.improve.diagnose import _call_llm

    prompt = (
        f"Based on this diagnosis, generate a unified diff to fix the issue.\n\n"
        f"File: {diagnosis.target_files}\n"
        f"Root cause: {diagnosis.root_cause}\n"
        f"Suggested fix: {diagnosis.suggested_fix}\n\n"
        f"Output only the unified diff, no explanation."
    )

    try:
        response = await _call_llm(prompt)
        diff = response if response else ""
    except Exception:
        logger.warning("Patch proposal failed for %s", diagnosis.trial_id, exc_info=True)
        diff = ""

    return ProposedPatch(
        diagnosis=diagnosis,
        file_path=diagnosis.target_files[0] if diagnosis.target_files else "unknown",
        description=diagnosis.suggested_fix,
        diff=diff,
        rationale=diagnosis.root_cause,
    )


def _sanitize_path(name: str) -> str:
    return re.sub(r"[\W\./\\\x00]", "_", name)


def write_patch(patch: ProposedPatch, output_dir: Path | None = None) -> Path:
    dir_ = output_dir or Path(".ash-hawk/patches")
    dir_.mkdir(parents=True, exist_ok=True)

    safe_trial_id = _sanitize_path(patch.diagnosis.trial_id)
    safe_file_stem = _sanitize_path(Path(patch.file_path).stem)
    path = dir_ / f"{safe_trial_id}_{safe_file_stem}.patch"

    path.write_text(_format_patch(patch))
    return path


def _format_patch(patch: ProposedPatch) -> str:
    return (
        f"# Patch for {patch.file_path}\n"
        f"# Trial: {patch.diagnosis.trial_id}\n"
        f"# Confidence: {patch.diagnosis.confidence:.2f}\n"
        f"# Root cause: {patch.diagnosis.root_cause}\n"
        f"\n"
        f"## Description\n{patch.description}\n\n"
        f"## Rationale\n{patch.rationale}\n\n"
        f"## Diff\n{patch.diff}\n"
    )
