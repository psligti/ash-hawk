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
    agent_relative_path: str | None = None
    content: str | None = None


async def propose_patch(
    diagnosis: Diagnosis,
    agent_content: dict[str, str] | None = None,
) -> ProposedPatch:
    from ash_hawk.improve.diagnose import _call_llm

    agent_relative_path: str | None = None
    full_content: str | None = None
    diff = ""

    if agent_content:
        formatted_agent = ""
        for key in sorted(agent_content.keys()):
            formatted_agent += f"\n--- {key} ---\n{agent_content[key][:2000]}\n"

        target_files = ", ".join(diagnosis.target_files) if diagnosis.target_files else "unknown"
        prompt = (
            "Based on this diagnosis and the agent content below, output the COMPLETE FILE CONTENT "
            "for the file that needs to change. Do NOT output a diff. Output the full file.\n\n"
            f"Agent files:\n{formatted_agent}\n\n"
            f"Target files: {target_files}\n"
            f"Root cause: {diagnosis.root_cause}\n"
            f"Suggested fix: {diagnosis.suggested_fix}\n\n"
            "Output the complete file content, starting with the file path on the first line as: "
            "FILE: path/to/file"
        )

        try:
            response = await _call_llm(prompt)
            if response and response.strip().startswith("FILE:"):
                lines = response.strip().split("\n", 1)
                agent_relative_path = lines[0].replace("FILE:", "").strip()
                full_content = lines[1] if len(lines) > 1 else ""
            elif response:
                diff = response
        except Exception:
            logger.warning("Patch proposal failed for %s", diagnosis.trial_id, exc_info=True)
    else:
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

    return ProposedPatch(
        diagnosis=diagnosis,
        file_path=diagnosis.target_files[0] if diagnosis.target_files else "unknown",
        description=diagnosis.suggested_fix,
        diff=diff,
        rationale=diagnosis.root_cause,
        agent_relative_path=agent_relative_path,
        content=full_content,
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
    parts = [
        f"# Patch for {patch.file_path}",
        f"# Trial: {patch.diagnosis.trial_id}",
        f"# Confidence: {patch.diagnosis.confidence:.2f}",
        f"# Root cause: {patch.diagnosis.root_cause}",
        "",
        f"## Description\n{patch.description}\n",
        f"## Rationale\n{patch.rationale}\n",
    ]
    if patch.agent_relative_path and patch.content:
        parts.append(f"## Agent File: {patch.agent_relative_path}\n{patch.content}\n")
    else:
        parts.append(f"## Diff\n{patch.diff}\n")
    return "\n".join(parts)
