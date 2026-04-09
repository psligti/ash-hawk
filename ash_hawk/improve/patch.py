# type-hygiene: skip-file
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pydantic as pd

from ash_hawk.improve.diagnose import Diagnosis

logger = logging.getLogger(__name__)


class ProposedPatch(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    diagnosis: Any = pd.Field(description="The Diagnosis that generated this patch")
    file_path: str = pd.Field(description="Target file path")
    description: str = pd.Field(description="Human-readable description")
    diff: str = pd.Field(default="", description="Unified diff content")
    rationale: str = pd.Field(default="", description="Why this patch was proposed")
    agent_relative_path: str | None = pd.Field(
        default=None, description="Relative path within agent dir"
    )
    content: str | None = pd.Field(default=None, description="Full file content")


async def propose_patch(
    diagnosis: Diagnosis,
    agent_content: dict[str, str] | None = None,
    console: Any | None = None,
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
            if console is not None:
                console.print("    [dim]Generating patch via LLM...[/dim]")
            response = await _call_llm(prompt)
            if response and response.strip().startswith("FILE:"):
                lines = response.strip().split("\n", 1)
                agent_relative_path = lines[0].replace("FILE:", "").strip()
                full_content = lines[1] if len(lines) > 1 else ""
                if agent_relative_path and console is not None:
                    console.print(f"    [dim]Patch targets: {agent_relative_path}[/dim]")
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
            if console is not None:
                console.print("    [dim]Generating diff patch via LLM...[/dim]")
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


def _build_mutation_prompt(
    diagnosis: Diagnosis,
    grader_details: str = "",
    transcript_excerpt: str = "",
    agent_content: dict[str, str] | None = None,
    agent_source_path: Path | None = None,
    repo_root: Path | None = None,
) -> str:
    agent_files_section = ""
    if agent_content:
        lines: list[str] = []
        for key in sorted(agent_content.keys()):
            lines.append(f"\n### {key}\n```\n{agent_content[key][:3000]}\n```")
        agent_files_section = "\n## Current Source Files\n" + "\n".join(lines)

    target_files = ", ".join(diagnosis.target_files) if diagnosis.target_files else "unknown"

    grader_section = ""
    if grader_details:
        grader_section = f"\n## Grader Results (detailed)\n{grader_details}\n"

    transcript_section = ""
    if transcript_excerpt:
        transcript_section = f"\n## Agent Transcript (last actions)\n{transcript_excerpt}\n"

    scope_constraint = ""
    if agent_source_path is not None and repo_root is not None:
        rel_agent = (
            agent_source_path.relative_to(repo_root)
            if agent_source_path.is_relative_to(repo_root)
            else agent_source_path
        )
        scope_constraint = (
            f"\n## CRITICAL SCOPE CONSTRAINT\n"
            f"You MUST ONLY modify files under {rel_agent}/.\n"
            f"Do NOT modify session files, test files, eval infrastructure, or anything outside {rel_agent}/.\n"
            f"Your changes will be measured by diffing only files in {rel_agent}/ — "
            f"changes elsewhere are invisible and wasted effort.\n"
        )

    return (
        "You are a coding agent that has been running evaluation scenarios. "
        "The latest evaluation run FAILED. Your task is to fix your OWN source code "
        "to improve performance on future runs.\n\n"
        "## Failure Diagnosis\n"
        f"- Trial: {diagnosis.trial_id}\n"
        f"- Summary: {diagnosis.failure_summary}\n"
        f"- Root Cause: {diagnosis.root_cause}\n"
        f"- Suggested Fix: {diagnosis.suggested_fix}\n"
        f"- Target Files: {target_files}\n"
        f"{grader_section}"
        f"{transcript_section}"
        f"{scope_constraint}"
        f"{agent_files_section}\n\n"
        "## Instructions\n"
        "1. Read the relevant source files (prompts, config, tools)\n"
        "2. Identify what in YOUR code caused this failure\n"
        "3. Make minimal, targeted changes to fix the issue\n"
        "4. Do NOT change test files, session files, or evaluation infrastructure\n"
        f"5. Focus on the root cause: {diagnosis.root_cause}\n\n"
        "The files are in the current working directory. Use your tools to read and edit them."
    )


async def propose_patch_via_agent(
    diagnosis: Diagnosis,
    agent_source_path: Path,
    agent_content: dict[str, str] | None = None,
    grader_details: str = "",
    transcript_excerpt: str = "",
    console: Any | None = None,
    config_path: Path | None = None,
    repo_root: Path | None = None,
) -> ProposedPatch:
    prompt = _build_mutation_prompt(
        diagnosis,
        grader_details=grader_details,
        transcript_excerpt=transcript_excerpt,
        agent_content=agent_content,
        agent_source_path=agent_source_path,
        repo_root=repo_root,
    )

    if console is not None:
        console.print("    [dim]Running agent to apply fix...[/dim]")

    response_text, error = await _run_agent_cli(
        prompt=prompt,
        cwd=repo_root or agent_source_path.parent,
        config_path=config_path,
    )

    if error is not None:
        if console is not None:
            console.print(f"    [bold red]✗ Agent execution failed: {error}[/bold red]")
        logger.warning("Agent execution failed for %s: %s", diagnosis.trial_id, error)
        return ProposedPatch(
            diagnosis=diagnosis,
            file_path=diagnosis.target_files[0] if diagnosis.target_files else "unknown",
            description=f"Agent execution failed: {error}",
            diff="",
            rationale=diagnosis.root_cause,
            agent_relative_path=None,
            content=None,
        )

    return ProposedPatch(
        diagnosis=diagnosis,
        file_path=diagnosis.target_files[0] if diagnosis.target_files else "unknown",
        description=diagnosis.suggested_fix,
        diff="",
        rationale=diagnosis.root_cause,
        agent_relative_path="(agent-edited)",
        content=response_text or "",
    )


async def _run_agent_cli(
    prompt: str,
    cwd: Path,
    config_path: Path | None = None,
) -> tuple[str | None, str | None]:
    import asyncio
    import shutil

    bolt_merlin = shutil.which("bolt-merlin")
    if bolt_merlin is None:
        venv_bin = Path(".venv") / "bin" / "bolt-merlin"
        if venv_bin.exists():
            bolt_merlin = str(venv_bin)
        else:
            return None, "bolt-merlin CLI not found in PATH or .venv/bin"

    cmd = [bolt_merlin, "code", prompt]
    if config_path is not None and config_path.exists():
        cmd.extend(["--config", str(config_path)])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        err_msg = stderr.decode(errors="replace").strip() or f"exit code {proc.returncode}"
        return None, err_msg

    return stdout.decode(errors="replace").strip(), None


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
