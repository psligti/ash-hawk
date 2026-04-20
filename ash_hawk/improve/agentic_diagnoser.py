from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ash_hawk.improve.lesson_store import LessonStore
    from ash_hawk.improve.run_bundle import ImproveRunBundle
    from ash_hawk.types import EvalTrial

EXPLORER_DIAGNOSIS_PROMPT = """Investigate the failed evaluation trial using read-only exploration.

Do not modify files.

You have a strict investigation budget:
- Read the trial bundle first.
- Use at most 5 tool calls total.
- Inspect at most 3 repository files.
- Inspect lessons only if the bundle explicitly suggests they are relevant.
- Prefer narrow `grep`/`glob` queries over broad recursive searching.

Start from these paths:
- agent root: {agent_root}
- trial bundle: {trial_bundle}
- lessons dir: {lessons_dir}

Mutable agent files (relative to agent root):
{agent_file_manifest}

Follow this workflow exactly:
1. Read the trial bundle.
2. Identify the 2-3 strongest failure signals.
3. Start with the suggested inspection paths in the bundle. These are your primary clues.
4. Use `json_query` on the bundle before broad reads whenever you need specific grader or trace fields.
5. If the request gives exact file paths, use `grep` first to locate symbols or patterns, then `read_range` for the minimum chunk.
6. Use narrow `glob` only if an exact path is missing.
7. Use full `read` only as a last resort when `grep` and `read_range` are insufficient.
8. Use `lesson_query` to search lessons instead of broad lesson reads.
9. Return 1-3 distinct diagnosis ideas grounded in the evidence you actually inspected.

Do not search the entire repository. Do not inspect unrelated files. If the bundle already points to likely files, follow those clues first and stop once you have enough evidence.
Prefer `json_query`, `grep`, and `read_range` over broad `read` calls whenever possible.
If tool_call_count is 0 or required files were not changed, prioritize diagnosing why the agent produced prose/todo updates without real file edits.
If completion claims exceed actual completed work, prioritize truthfulness and execution-flow failures.

Prefer a single-file or tightly-coupled two-file fix whenever possible.
Avoid broad refactors, sweeping prompt rewrites, and cross-module changes unless the inspected
evidence clearly requires them.
Prefer executable code-path fixes first: `execute.py`, `coding_agent.py`, `prompt_builder.py`,
`tool_dispatcher.py`, or specific `tools/*` modules.
Treat `prompts/*` and `skills/*` as last-resort shared surfaces. Only target them when the
inspected evidence shows a narrower code-path fix is insufficient.
Do not mix a local code fix with prompt/skill cleanup in the same idea unless the evidence proves
both changes are required together.
Return fewer ideas if the evidence is narrow; do not invent diversity.
Return only JSON in this shape:
{{
  "ideas": [
    {{
      "failure_summary": "one-line summary",
      "root_cause": "detailed root cause analysis grounded in evidence",
      "suggested_fix": "concrete fix suggestion",
      "target_files": ["relative/path.py"],
      "anchor_files": ["existing/file.py"],
      "confidence": 0.8
    }}
  ]
}}

Rules for file targeting:
- `target_files` must be relative to the agent root.
- Prefer existing files from the mutable agent file manifest above.
- You may propose a NEW file only if it clearly fits under the existing architecture and you also provide 1-2 existing `anchor_files` from the manifest that will wire it in.
- Do not invent generic filenames like `agent.py` or `tools.py` unless they actually appear in the manifest.
- Shared files under `prompts/` or `skills/` should appear only when you can explain why a narrower code-path fix is insufficient.
"""


@dataclass(frozen=True)
class ExplorerDiagnosisResult:
    response: str
    raw_response: str = ""
    error: str | None = None
    tool_calls_used: int | None = None
    tool_calls_max: int | None = None
    file_reads_used: int | None = None
    file_reads_max: int | None = None
    search_calls_used: int | None = None
    search_calls_max: int | None = None


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


def _path_for_prompt(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _format_agent_file_manifest(agent_content: dict[str, str] | None) -> str:
    if not agent_content:
        return "none"
    files = sorted(agent_content.keys())
    if len(files) > 80:
        visible = files[:80]
        visible.append(f"... (+{len(files) - 80} more)")
        files = visible
    return "\n".join(f"- {path}" for path in files)


def _write_trial_bundle(bundle_dir: Path, trial: EvalTrial) -> Path:
    bundle = _build_trial_bundle(trial)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / f"{trial.id}.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle_path


def _extract_json_payload(stdout: str) -> dict[str, object] | None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _build_trial_bundle(trial: EvalTrial) -> dict[str, object]:
    transcript = trial.result.transcript if trial.result is not None else None
    grader_failures: list[dict[str, object]] = []
    candidate_repo_files: list[str] = []
    if trial.result is not None:
        for grader in trial.result.grader_results:
            if grader.passed:
                continue
            details = grader.details
            if isinstance(details, dict):
                required = details.get("missing_required")
                if isinstance(required, list):
                    for path in required:
                        if isinstance(path, str) and path not in candidate_repo_files:
                            candidate_repo_files.append(path)
                semantic = details.get("semantic_failures")
                if isinstance(semantic, list):
                    for failure in semantic:
                        if isinstance(failure, dict):
                            path = failure.get("path")
                            if isinstance(path, str) and path not in candidate_repo_files:
                                candidate_repo_files.append(path)
            grader_failures.append(
                {
                    "grader_type": grader.grader_type,
                    "score": grader.score,
                    "error_message": grader.error_message,
                    "details": details,
                }
            )

    tool_names: list[str] = []
    if transcript is not None:
        for call in transcript.tool_calls[:12]:
            name = call.get("name") or call.get("tool")
            if isinstance(name, str):
                tool_names.append(name)

    first_message = ""
    if transcript is not None and transcript.messages:
        content = transcript.messages[0].get("content")
        if isinstance(content, str):
            first_message = content[:2000]

    input_data = trial.input_snapshot if isinstance(trial.input_snapshot, dict) else {}
    scenario_path = input_data.get("scenario_path") if isinstance(input_data, dict) else None
    scenario_root = input_data.get("scenario_root") if isinstance(input_data, dict) else None
    agent_response_excerpt = ""
    if transcript is not None and isinstance(transcript.agent_response, str):
        agent_response_excerpt = transcript.agent_response[:2000]

    return {
        "trial_id": trial.id,
        "task_id": trial.task_id,
        "status": trial.status.value,
        "scenario_path": scenario_path,
        "scenario_root": scenario_root,
        "user_task": first_message,
        "failure_mode": None if trial.result is None else trial.result.outcome.failure_mode,
        "error_message": None if trial.result is None else trial.result.outcome.error_message,
        "aggregate_score": None if trial.result is None else trial.result.aggregate_score,
        "aggregate_passed": None if trial.result is None else trial.result.aggregate_passed,
        "agent_response_excerpt": agent_response_excerpt,
        "tool_call_count": 0 if transcript is None else len(transcript.tool_calls),
        "tool_names": tool_names,
        "candidate_repo_files": candidate_repo_files[:8],
        "investigation_focus": _derive_investigation_focus(trial, candidate_repo_files),
        "suggested_inspection_paths": _derive_suggested_inspection_paths(
            trial, candidate_repo_files
        ),
        "trace_event_types": []
        if transcript is None
        else [
            str(event.get("event_type", "unknown"))
            for event in transcript.trace_events[:12]
            if isinstance(event, dict)
        ],
        "failed_graders": grader_failures,
    }


def _derive_investigation_focus(trial: EvalTrial, candidate_repo_files: list[str]) -> str:
    transcript = trial.result.transcript if trial.result is not None else None
    if transcript is not None and len(transcript.tool_calls) == 0:
        return (
            "The agent produced natural-language progress updates but made zero tool calls. "
            "Focus on why the coding agent can claim work without actually using tools or changing files."
        )
    if candidate_repo_files:
        return (
            "The graders point to required file edits that did not happen. Focus on how the agent chooses files, "
            "invokes tools, and verifies edits."
        )
    return "Focus on the strongest failed grader signals and the agent execution path that could explain them."


def _derive_suggested_inspection_paths(
    trial: EvalTrial, candidate_repo_files: list[str]
) -> list[str]:
    transcript = trial.result.transcript if trial.result is not None else None
    suggestions: list[str] = []

    if transcript is not None and len(transcript.tool_calls) == 0:
        suggestions.extend(
            [
                "bolt_merlin/agent/execute.py",
                "bolt_merlin/agent/coding_agent.py",
                "bolt_merlin/agent/tool_dispatcher.py",
            ]
        )

    if candidate_repo_files:
        for path in [
            "bolt_merlin/agent/execute.py",
            "bolt_merlin/agent/coding_agent.py",
            "bolt_merlin/agent/tools/edit.py",
            "bolt_merlin/agent/tool_dispatcher.py",
            "bolt_merlin/agent/prompts/coding.md",
            "bolt_merlin/agent/skills/general-coding/SKILL.md",
        ]:
            if path not in suggestions:
                suggestions.append(path)

    return suggestions[:3]


async def investigate_trial_with_explorer(
    trial: EvalTrial,
    agent_path: Path,
    lesson_store: LessonStore | None = None,
    *,
    agent_content: dict[str, str] | None = None,
    audit_bundle: ImproveRunBundle | None = None,
    audit_stem: str | None = None,
) -> ExplorerDiagnosisResult | None:
    from ash_hawk.improve.patch import run_agent_cli

    repo_root = _repo_root_from_agent_path(agent_path)
    config_path = _explorer_config_path(repo_root)
    if not config_path.exists():
        return None

    bundle_dir = repo_root / ".ash-hawk" / "diagnosis-bundles"
    bundle_path = _write_trial_bundle(bundle_dir, trial)
    try:
        if audit_bundle is not None and audit_stem is not None:
            audit_bundle.write_json(
                f"diagnoses/{audit_stem}/bundle.json", _build_trial_bundle(trial)
            )
        prompt = EXPLORER_DIAGNOSIS_PROMPT.format(
            agent_root=_path_for_prompt(agent_path, repo_root),
            trial_bundle=_path_for_prompt(bundle_path, repo_root),
            lessons_dir=_path_for_prompt(_lessons_dir(lesson_store, repo_root), repo_root),
            agent_file_manifest=_format_agent_file_manifest(agent_content),
        )
        if audit_bundle is not None and audit_stem is not None:
            audit_bundle.write_text(f"diagnoses/{audit_stem}/prompt.txt", prompt)
        cli_result = await run_agent_cli(
            prompt=prompt,
            cwd=repo_root,
            config_path=config_path,
            command_name="explore",
            timeout_seconds=240.0,
            json_output=True,
        )
        if len(cli_result) == 3:
            stdout, error, _execution_metrics = cli_result
        else:
            stdout, error = cli_result
        if error is not None:
            return ExplorerDiagnosisResult(response="", error=error)
        if stdout is None:
            return ExplorerDiagnosisResult(response="", error="explorer returned no stdout")
        payload = _extract_json_payload(stdout)
        if payload is None:
            return ExplorerDiagnosisResult(response=stdout, raw_response=stdout)

        response = payload.get("response")
        metadata = payload.get("metadata")
        budget = metadata.get("explorer_budget") if isinstance(metadata, dict) else None
        return ExplorerDiagnosisResult(
            response=response if isinstance(response, str) else "",
            raw_response=stdout,
            error=None,
            tool_calls_used=budget.get("tool_calls_used") if isinstance(budget, dict) else None,
            tool_calls_max=budget.get("tool_calls_max") if isinstance(budget, dict) else None,
            file_reads_used=budget.get("file_reads_used") if isinstance(budget, dict) else None,
            file_reads_max=budget.get("file_reads_max") if isinstance(budget, dict) else None,
            search_calls_used=budget.get("search_calls_used") if isinstance(budget, dict) else None,
            search_calls_max=budget.get("search_calls_max") if isinstance(budget, dict) else None,
        )
    finally:
        bundle_path.unlink(missing_ok=True)


__all__ = ["ExplorerDiagnosisResult", "investigate_trial_with_explorer"]
