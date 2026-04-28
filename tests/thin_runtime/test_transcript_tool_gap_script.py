from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "analyze_latest_transcript.py"
    module_name = "analyze_latest_transcript"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_script_detects_missing_handler_gap() -> None:
    module = _load_script_module()
    execution = {
        "error": "No handler registered for tool: synthesize_patch",
        "selected_tool_names": ["synthesize_patch"],
        "tool_results": [],
        "context": {"runtime": {}},
    }

    findings = module._find_gaps(execution)

    assert findings
    assert findings[0].category == "missing_handler"
    assert findings[0].tool == "synthesize_patch"


def test_script_detects_policy_denied_gap() -> None:
    module = _load_script_module()
    execution = {
        "error": "",
        "selected_tool_names": ["grep"],
        "tool_results": [
            {
                "tool_name": "grep",
                "error": "Tool denied by policy: grep",
                "payload": {"errors": []},
            }
        ],
        "context": {"runtime": {}},
    }

    findings = module._find_gaps(execution)

    assert any(item.category == "policy_denied" and item.tool == "grep" for item in findings)


def test_script_infers_context_producer_tool() -> None:
    module = _load_script_module()
    execution = {
        "error": "",
        "selected_tool_names": ["verify_outcome"],
        "tool_results": [
            {
                "tool_name": "verify_outcome",
                "error": "Missing required contexts: evaluation_context",
                "payload": {"errors": []},
            }
        ],
        "context": {"runtime": {}},
    }

    findings = module._find_gaps(execution)

    assert any(
        item.category == "missing_context" and item.tool == "run_baseline_eval" for item in findings
    )


def test_script_has_unclassified_fallback() -> None:
    module = _load_script_module()
    execution = {
        "error": "Some unexpected runtime blocker",
        "selected_tool_names": [],
        "tool_results": [],
        "context": {"runtime": {}},
    }

    findings = module._find_gaps(execution)

    assert findings
    assert findings[0].category == "unclassified_run_blocker"


def test_script_loads_error_templates_from_source_module(tmp_path: Path) -> None:
    module = _load_script_module()

    module_root = tmp_path / "workspace"
    script_dir = module_root / "scripts"
    signatures_dir = module_root / "ash_hawk" / "thin_runtime"
    script_dir.mkdir(parents=True)
    signatures_dir.mkdir(parents=True)

    fake_script_path = script_dir / "analyze_latest_transcript.py"
    fake_script_path.write_text("# placeholder", encoding="utf-8")

    signatures_path = signatures_dir / "error_signatures.py"
    signatures_path.write_text(
        textwrap.dedent(
            """
            NO_HANDLER_TEMPLATE = "NO HANDLER => {tool_name}"
            TOOL_DENIED_TEMPLATE = "DENIED => {tool_name}"
            MISSING_CONTEXTS_TEMPLATE = "NEED CONTEXTS => {contexts}"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    original_file = module.__file__
    module.__file__ = str(fake_script_path)
    try:
        no_handler, denied, missing_contexts = module._load_error_templates()
    finally:
        module.__file__ = original_file

    assert no_handler == "NO HANDLER => {tool_name}"
    assert denied == "DENIED => {tool_name}"
    assert missing_contexts == "NEED CONTEXTS => {contexts}"


def test_script_default_templates_match_shared_templates() -> None:
    module = _load_script_module()

    no_handler, denied, missing_contexts = module._load_error_templates()

    assert module._DEFAULT_NO_HANDLER_TEMPLATE == no_handler
    assert module._DEFAULT_TOOL_DENIED_TEMPLATE == denied
    assert module._DEFAULT_MISSING_CONTEXTS_TEMPLATE == missing_contexts
