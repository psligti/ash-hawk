"""Analyze thin_runtime run artifacts for missing-tool/self-heal gaps.

Usage:
  uv run python scripts/analyze_latest_transcript.py
  uv run python scripts/analyze_latest_transcript.py --run-id <run-id>
  uv run python scripts/analyze_latest_transcript.py --format json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

_DEFAULT_NO_HANDLER_TEMPLATE = "No handler registered for tool: {tool_name}"
_DEFAULT_TOOL_DENIED_TEMPLATE = "Tool denied by policy: {tool_name}"
_DEFAULT_MISSING_CONTEXTS_TEMPLATE = "Missing required contexts: {contexts}"


def _load_error_templates() -> tuple[str, str, str]:
    module_path = (
        Path(__file__).resolve().parent.parent / "ash_hawk" / "thin_runtime" / "error_signatures.py"
    )
    if not module_path.exists():
        return (
            _DEFAULT_NO_HANDLER_TEMPLATE,
            _DEFAULT_TOOL_DENIED_TEMPLATE,
            _DEFAULT_MISSING_CONTEXTS_TEMPLATE,
        )

    try:
        source = module_path.read_text(encoding="utf-8")
        parsed = ast.parse(source)
    except (OSError, SyntaxError):
        return (
            _DEFAULT_NO_HANDLER_TEMPLATE,
            _DEFAULT_TOOL_DENIED_TEMPLATE,
            _DEFAULT_MISSING_CONTEXTS_TEMPLATE,
        )

    values: dict[str, str] = {}
    for node in parsed.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                "NO_HANDLER_TEMPLATE",
                "TOOL_DENIED_TEMPLATE",
                "MISSING_CONTEXTS_TEMPLATE",
            }:
                value = ast.literal_eval(node.value)
                if isinstance(value, str):
                    values[target.id] = value

    return (
        values.get("NO_HANDLER_TEMPLATE", _DEFAULT_NO_HANDLER_TEMPLATE),
        values.get("TOOL_DENIED_TEMPLATE", _DEFAULT_TOOL_DENIED_TEMPLATE),
        values.get("MISSING_CONTEXTS_TEMPLATE", _DEFAULT_MISSING_CONTEXTS_TEMPLATE),
    )


_NO_HANDLER_TEMPLATE, _TOOL_DENIED_TEMPLATE, _MISSING_CONTEXTS_TEMPLATE = _load_error_templates()

_NO_HANDLER_PREFIX = _NO_HANDLER_TEMPLATE.split("{tool_name}", maxsplit=1)[0]
_TOOL_DENIED_PREFIX = _TOOL_DENIED_TEMPLATE.split("{tool_name}", maxsplit=1)[0]
_MISSING_CONTEXTS_PREFIX = _MISSING_CONTEXTS_TEMPLATE.split("{contexts}", maxsplit=1)[0]

_RE_NO_HANDLER = re.compile(rf"{re.escape(_NO_HANDLER_PREFIX)}\s*([a-zA-Z0-9_\-]+)")
_RE_DENIED = re.compile(rf"{re.escape(_TOOL_DENIED_PREFIX)}\s*([a-zA-Z0-9_\-]+)")
_RE_MISSING_CONTEXTS = re.compile(rf"{re.escape(_MISSING_CONTEXTS_PREFIX)}\s*(.+)$")
_RE_MISSING_FIELD = re.compile(r"Missing required field:\s*([a-zA-Z0-9_\-]+)")


@dataclass
class GapFinding:
    category: str
    tool: str | None
    confidence: float
    evidence: str
    fix_hint: str


def _latest_run_dir(runs_root: Path) -> Path:
    candidates = [
        path for path in runs_root.iterdir() if path.is_dir() and (path / "execution.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No run directories with execution.json found under: {runs_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_execution(run_dir: Path) -> dict[str, object]:
    execution_path = run_dir / "execution.json"
    if not execution_path.exists():
        raise FileNotFoundError(f"Missing execution.json at: {execution_path}")
    loaded = json.loads(execution_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"execution.json must contain an object: {execution_path}")
    return loaded


def _extract_error_signals(execution: dict[str, object]) -> list[tuple[str, str, str]]:
    signals: list[tuple[str, str, str]] = []

    top_error = execution.get("error")
    if isinstance(top_error, str) and top_error.strip():
        signals.append(("run", "error", top_error.strip()))

    for tool_result in execution.get("tool_results", []):
        if not isinstance(tool_result, dict):
            continue
        tool_name = str(tool_result.get("tool_name") or "unknown")
        error = tool_result.get("error")
        if isinstance(error, str) and error.strip():
            signals.append((tool_name, "error", error.strip()))
        payload = tool_result.get("payload")
        if isinstance(payload, dict):
            payload_errors = payload.get("errors")
            if isinstance(payload_errors, list):
                for item in payload_errors:
                    if isinstance(item, str) and item.strip():
                        signals.append((tool_name, "payload_error", item.strip()))

    return signals


def _infer_context_producer_tool(context_name: str) -> str | None:
    mapping = {
        "evaluation_context": "run_baseline_eval",
        "failure_context": "run_baseline_eval",
        "workspace_context": "load_workspace_state",
        "audit_context": "load_workspace_state",
        "tool_context": "load_workspace_state",
    }
    return mapping.get(context_name.strip())


def _find_gaps(execution: dict[str, object]) -> list[GapFinding]:
    findings: list[GapFinding] = []
    signals = _extract_error_signals(execution)

    selected_tool_names = [
        name for name in execution.get("selected_tool_names", []) if isinstance(name, str)
    ]
    produced_tools = {
        item.get("tool_name")
        for item in execution.get("tool_results", [])
        if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
    }
    for selected in selected_tool_names:
        if selected not in produced_tools:
            findings.append(
                GapFinding(
                    category="planned_not_executed",
                    tool=selected,
                    confidence=0.68,
                    evidence=f"selected_tool_names contains '{selected}' but no matching ToolResult exists",
                    fix_hint="Ensure this tool is invokable in the current surface and receives required args.",
                )
            )

    preferred_tool = execution.get("context", {}).get("runtime", {}).get("preferred_tool")
    if (
        isinstance(preferred_tool, str)
        and preferred_tool.strip()
        and preferred_tool not in selected_tool_names
    ):
        findings.append(
            GapFinding(
                category="preferred_tool_not_selected",
                tool=preferred_tool,
                confidence=0.62,
                evidence=f"runtime.preferred_tool='{preferred_tool}' was never selected",
                fix_hint="Add this tool to the active agent/skill surface or adjust planner/tool order to allow it.",
            )
        )

    for source, signal_type, message in signals:
        no_handler = _RE_NO_HANDLER.search(message)
        if no_handler:
            tool = no_handler.group(1)
            findings.append(
                GapFinding(
                    category="missing_handler",
                    tool=tool,
                    confidence=0.99,
                    evidence=f"{source}:{signal_type} -> {message}",
                    fix_hint=(
                        f"Implement tool handler for '{tool}' and register it in defaults/registry so it can execute."
                    ),
                )
            )
            continue

        denied = _RE_DENIED.search(message)
        if denied:
            tool = denied.group(1)
            findings.append(
                GapFinding(
                    category="policy_denied",
                    tool=tool,
                    confidence=0.96,
                    evidence=f"{source}:{signal_type} -> {message}",
                    fix_hint=(
                        f"Grant '{tool}' in the right agent/skill surface or delegate to a specialist that already has it."
                    ),
                )
            )
            continue

        missing_contexts = _RE_MISSING_CONTEXTS.search(message)
        if missing_contexts:
            contexts = [
                item.strip() for item in missing_contexts.group(1).split(",") if item.strip()
            ]
            for context_name in contexts:
                producer = _infer_context_producer_tool(context_name)
                findings.append(
                    GapFinding(
                        category="missing_context",
                        tool=producer,
                        confidence=0.84 if producer else 0.55,
                        evidence=f"{source}:{signal_type} -> missing '{context_name}'",
                        fix_hint=(
                            f"Run a producer for '{context_name}' before the failing tool."
                            if producer
                            else f"Add a tool that produces '{context_name}' before this step."
                        ),
                    )
                )
            continue

        if "No eligible tools available" in message:
            findings.append(
                GapFinding(
                    category="tool_surface_gap",
                    tool=preferred_tool
                    if isinstance(preferred_tool, str) and preferred_tool
                    else None,
                    confidence=0.66,
                    evidence=f"{source}:{signal_type} -> {message}",
                    fix_hint=(
                        "Expand active tool surface for this phase or use delegate_task to route to a specialist."
                    ),
                )
            )
            continue

        if "cannot write memory scope" in message:
            findings.append(
                GapFinding(
                    category="memory_scope_permission",
                    tool=None,
                    confidence=0.72,
                    evidence=f"{source}:{signal_type} -> {message}",
                    fix_hint=(
                        "Align agent/skill memory_write_scopes with permitted writable_by policy for this scope."
                    ),
                )
            )
            continue

        missing_field = _RE_MISSING_FIELD.search(message)
        if missing_field:
            findings.append(
                GapFinding(
                    category="argument_gap",
                    tool=source if source != "run" else None,
                    confidence=0.35,
                    evidence=f"{source}:{signal_type} -> {message}",
                    fix_hint=(
                        "Tool likely exists but call shape is incomplete. Update prompt/schema examples and retry."
                    ),
                )
            )

    deduped: dict[tuple[str, str | None, str], GapFinding] = {}
    for finding in findings:
        key = (finding.category, finding.tool, finding.evidence)
        deduped[key] = finding
    ordered = sorted(deduped.values(), key=lambda item: item.confidence, reverse=True)
    if not ordered:
        run_error = execution.get("error")
        if isinstance(run_error, str) and run_error.strip():
            ordered.append(
                GapFinding(
                    category="unclassified_run_blocker",
                    tool=None,
                    confidence=0.2,
                    evidence=f"run:error -> {run_error.strip()}",
                    fix_hint="No explicit missing-tool signature found. Inspect execution.json context and tool payloads for root cause.",
                )
            )
    return ordered


def _summarize(
    run_dir: Path, execution: dict[str, object], findings: list[GapFinding]
) -> dict[str, object]:
    tool_counter = Counter(
        item.get("tool_name")
        for item in execution.get("tool_results", [])
        if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
    )
    top_finding = findings[0] if findings else None
    return {
        "run_dir": str(run_dir),
        "run_id": execution.get("run_id"),
        "success": execution.get("success"),
        "error": execution.get("error"),
        "selected_tool_names": execution.get("selected_tool_names", []),
        "tool_result_count": len(execution.get("tool_results", [])),
        "tool_frequency": dict(tool_counter),
        "primary_gap": asdict(top_finding) if top_finding else None,
        "gaps": [asdict(item) for item in findings],
    }


def _render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Latest Transcript Tool-Gap Diagnosis",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Directory: `{summary['run_dir']}`",
        f"- Success: `{summary['success']}`",
        f"- Error: `{summary['error']}`",
        "",
        "## Primary Gap",
    ]

    primary = summary.get("primary_gap")
    if primary is None:
        lines.append("No high-confidence tool gap found.")
    else:
        lines.extend(
            [
                f"- Category: `{primary['category']}`",
                f"- Tool: `{primary['tool']}`",
                f"- Confidence: `{primary['confidence']}`",
                f"- Evidence: {primary['evidence']}",
                f"- Fix hint: {primary['fix_hint']}",
            ]
        )

    lines.extend(["", "## Gap Candidates"])
    gaps = summary.get("gaps", [])
    if not gaps:
        lines.append("- None")
    else:
        for gap in gaps[:12]:
            lines.append(
                f"- `{gap['category']}` / tool=`{gap['tool']}` / conf={gap['confidence']}: {gap['evidence']}"
            )

    lines.extend(
        [
            "",
            "## Self-Heal Next Step",
            "1. Apply the primary fix hint.",
            "2. Re-run target eval.",
            "3. Re-run this analyzer and compare gap list reduction.",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze latest thin_runtime transcript for missing-tool gaps"
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(".ash-hawk/thin_runtime/runs"),
        help="Root directory containing thin_runtime run directories",
    )
    parser.add_argument("--run-id", type=str, default="", help="Specific run id to analyze")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    run_dir = runs_root / args.run_id if args.run_id else _latest_run_dir(runs_root)
    execution = _load_execution(run_dir)
    findings = _find_gaps(execution)
    summary = _summarize(run_dir, execution, findings)

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        print(_render_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
