# type-hygiene: skip-file
from __future__ import annotations

import logging
import re
import statistics
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
    execution_metrics: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Mutation subprocess execution metrics",
    )
    failure_reason: str | None = pd.Field(
        default=None,
        description="Structured failure reason when mutation generation did not yield a clean patch",
    )
    recovered_from_changed_files: bool = pd.Field(
        default=False,
        description="Whether the mutation was recovered from on-disk changes instead of clean stdout",
    )


class MutationExecutionMetrics(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    registered_tool_count: int = pd.Field(default=0, ge=0)
    llm_completion_count: int = pd.Field(default=0, ge=0)
    llm_completion_durations: list[float] = pd.Field(default_factory=list)
    max_llm_completion_seconds: float = pd.Field(default=0.0, ge=0.0)
    mean_llm_completion_seconds: float = pd.Field(default=0.0, ge=0.0)
    long_llm_completion_count: int = pd.Field(default=0, ge=0)
    created_virtualenv: bool = pd.Field(default=False)


def _extract_execution_metrics(stdout_text: str, stderr_text: str) -> MutationExecutionMetrics:
    durations = [
        float(match)
        for match in re.findall(
            r"_collect_stream_events_once completed in ([0-9]+(?:\.[0-9]+)?)s", stdout_text
        )
    ]
    return MutationExecutionMetrics(
        registered_tool_count=len(
            re.findall(r"^Registered tool:", stdout_text, flags=re.MULTILINE)
        ),
        llm_completion_count=len(durations),
        llm_completion_durations=durations,
        max_llm_completion_seconds=max(durations, default=0.0),
        mean_llm_completion_seconds=statistics.mean(durations) if durations else 0.0,
        long_llm_completion_count=sum(1 for duration in durations if duration >= 90.0),
        created_virtualenv="Creating virtual environment at:" in stderr_text,
    )


def _classify_cli_error(error: str) -> str:
    lowered = error.lower()
    if "timed out" in lowered:
        return "mutation_cli_timeout"
    if "registered zero tools" in lowered:
        return "mutation_zero_tools"
    return "mutation_cli_error"


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
        anchor_files = ", ".join(diagnosis.anchor_files) if diagnosis.anchor_files else "none"
        prompt = (
            "Based on this diagnosis and the agent content below, output the COMPLETE FILE CONTENT "
            "for the file that needs to change. Do NOT output a diff. Output the full file.\n\n"
            f"Agent files:\n{formatted_agent}\n\n"
            f"Target files: {target_files}\n"
            f"Anchor files: {anchor_files}\n"
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
    anchor_files = ", ".join(diagnosis.anchor_files) if diagnosis.anchor_files else "none"

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
        f"- Anchor Files: {anchor_files}\n"
        f"{grader_section}"
        f"{transcript_section}"
        f"{scope_constraint}"
        f"{agent_files_section}\n\n"
        "## Instructions\n"
        "1. Read the relevant source files (prompts, config, tools)\n"
        "2. Identify the smallest plausible change in YOUR code that caused this failure\n"
        "3. Make the smallest possible targeted change. Prefer one file, or at most two tightly-coupled files.\n"
        "4. Avoid broad prompt rewrites, sweeping refactors, new modules, or touching many tools unless the evidence clearly requires it.\n"
        "5. Do NOT change test files, session files, or evaluation infrastructure\n"
        f"6. Focus on the root cause: {diagnosis.root_cause}\n\n"
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
    timeout_seconds: float | None = None,
    audit_bundle: Any | None = None,
    audit_stem: str | None = None,
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

    response_text, error, execution_metrics = await run_agent_cli(
        prompt=prompt,
        cwd=repo_root or agent_source_path.parent,
        config_path=config_path,
        timeout_seconds=timeout_seconds,
        audit_bundle=audit_bundle,
        audit_stem=audit_stem,
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
            execution_metrics=execution_metrics,
            failure_reason=_classify_cli_error(error),
        )

    return ProposedPatch(
        diagnosis=diagnosis,
        file_path=diagnosis.target_files[0] if diagnosis.target_files else "unknown",
        description=diagnosis.suggested_fix,
        diff="",
        rationale=diagnosis.root_cause,
        agent_relative_path="(agent-edited)",
        content=response_text or "",
        execution_metrics=execution_metrics,
        failure_reason=None,
    )


async def run_agent_cli(
    prompt: str,
    cwd: Path,
    config_path: Path | None = None,
    agent_name: str | None = None,
    timeout_seconds: float | None = None,
    command_name: str = "code",
    json_output: bool = False,
    audit_bundle: Any | None = None,
    audit_stem: str | None = None,
) -> tuple[str | None, str | None, dict[str, Any]]:
    import asyncio
    import os
    import shutil

    uv_executable = shutil.which("uv")
    local_pyproject = cwd / "pyproject.toml"
    local_cli = cwd / ".venv" / "bin" / "bolt-merlin"

    if uv_executable is not None and local_pyproject.exists():
        cmd = [uv_executable, "run", "bolt-merlin", command_name]
    elif local_cli.exists():
        cmd = [str(local_cli), command_name]
    else:
        bolt_merlin = shutil.which("bolt-merlin")
        if bolt_merlin is None:
            return None, "bolt-merlin CLI not found in PATH, local .venv, or via uv run", {}
        cmd = [bolt_merlin, command_name]

    if agent_name is not None and command_name == "code":
        cmd.extend(["--agent", agent_name])
    if config_path is not None and config_path.exists():
        cmd.extend(["--config", str(config_path)])
    if json_output:
        cmd.append("--json")
    cmd.append(prompt)

    # Prevent the worktree's source tree (e.g. bolt_merlin/) from shadowing
    # the installed package on sys.path.  Without this, Python adds cwd to
    # sys.path and `from bolt_merlin.cli.main import cli` resolves to the
    # local copy instead of the installed one.
    env = os.environ.copy()
    env["PYTHONPATH"] = ""

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except TimeoutError:
        import signal

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.communicate()
        timeout_label = timeout_seconds if timeout_seconds is not None else "configured"
        if audit_bundle is not None and audit_stem is not None:
            audit_bundle.write_json(
                f"subprocess/{audit_stem}.json",
                {
                    "command": cmd,
                    "cwd": str(cwd),
                    "config_path": str(config_path) if config_path else None,
                    "command_name": command_name,
                    "agent_name": agent_name,
                    "json_output": json_output,
                    "timeout_seconds": timeout_seconds,
                    "stdout": None,
                    "stderr": None,
                    "error": f"bolt-merlin CLI timed out after {timeout_label}s",
                    "execution_metrics": {},
                },
            )
        return None, f"bolt-merlin CLI timed out after {timeout_label}s", {}

    stdout_text = stdout.decode(errors="replace").strip()
    stderr_text = stderr.decode(errors="replace").strip()
    execution_metrics = _extract_execution_metrics(stdout_text, stderr_text).model_dump()
    if audit_bundle is not None and audit_stem is not None:
        audit_bundle.write_json(
            f"subprocess/{audit_stem}.json",
            {
                "command": cmd,
                "cwd": str(cwd),
                "config_path": str(config_path) if config_path else None,
                "command_name": command_name,
                "agent_name": agent_name,
                "json_output": json_output,
                "timeout_seconds": timeout_seconds,
                "returncode": proc.returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "execution_metrics": execution_metrics,
                "error": None
                if proc.returncode == 0
                else (stderr_text or f"exit code {proc.returncode}"),
            },
        )

    if proc.returncode != 0:
        err_msg = stderr_text or f"exit code {proc.returncode}"
        return None, err_msg, execution_metrics

    if execution_metrics.get("registered_tool_count", 0) == 0:
        return None, "mutation subprocess registered zero tools", execution_metrics

    return stdout_text, None, execution_metrics


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
