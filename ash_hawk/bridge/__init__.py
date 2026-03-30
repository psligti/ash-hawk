"""Thin telemetry bridge for running real dawn-kestrel agents.

This module provides a thin wrapper around dawn-kestrel's agent_v2
that captures telemetry via RuntimeHook callbacks. Ash-hawk no longer
constructs execution contexts - it runs real agents and observes.

Key components:
- TelemetrySink: Protocol for receiving telemetry events
- TranscriptData: Captured transcript from agent run
- OutcomeData: Captured outcome from agent run
- RunManifest: Provenance identity for reproducible runs
- DiffReport: Structured comparison between two runs
- run_real_agent: Main entry point for thin bridge

Usage:
    from ash_hawk.bridge import run_real_agent, TelemetrySink

    class MySink(TelemetrySink):
        async def on_iteration(self, data: dict) -> None:
            print(f"Iteration {data['iteration']}")

    transcript, outcome = await run_real_agent(
        agent_path=Path(".dawn-kestrel/agents/bolt-merlin"),
        input="Find all TODOs",
        telemetry_sink=MySink(),
    )
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pydantic as pd

logger = logging.getLogger(__name__)

# File extensions included in provenance hash scans.
_HASHABLE_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".py", ".yaml", ".yml", ".json", ".txt", ".toml", ".cfg", ".ini"}
)

__all__ = [
    "TelemetrySink",
    "TranscriptData",
    "OutcomeData",
    "RunResult",
    "RunManifest",
    "DiffReport",
    "DiffFieldChange",
    "compute_file_hash",
    "compute_directory_hashes",
    "run_real_agent",
]


class TelemetrySink(Protocol):
    """Protocol for receiving telemetry events from agent runs.

    Implement this to capture real-time telemetry from agent execution.
    All methods are optional - implement only what you need.
    """

    async def on_iteration_start(self, data: dict[str, Any]) -> None:
        """Called at the start of each iteration.

        Args:
            data: Contains 'iteration', 'session_id', 'max_iterations'
        """
        ...

    async def on_iteration_end(self, data: dict[str, Any]) -> None:
        """Called at the end of each iteration.

        Args:
            data: Contains 'iteration', 'duration_ms', 'response_length'
        """
        ...

    async def on_action_decision(self, data: dict[str, Any]) -> None:
        """Called when a policy decision is made.

        Args:
            data: Contains 'decision', 'action_type', 'risk_level'
        """
        ...

    async def on_tool_result(self, data: dict[str, Any]) -> None:
        """Called after a tool execution completes.

        Args:
            data: Contains 'tool_name', 'status', 'duration_ms'
        """
        ...

    async def on_run_complete(self, data: dict[str, Any]) -> None:
        """Called when the agent run completes.

        Args:
            data: Contains 'success', 'error', 'total_iterations'
        """
        ...


@dataclass
class TranscriptData:
    """Captured transcript from agent run.

    Contains the full conversation history, tool calls, and trace events
    from a real agent execution.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, int] = field(
        default_factory=lambda: {
            "input": 0,
            "output": 0,
            "reasoning": 0,
            "cache_read": 0,
            "cache_write": 0,
        }
    )
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    agent_response: str = ""
    error_trace: str | None = None

    def to_eval_transcript(self) -> Any:
        """Convert to EvalTranscript for compatibility with existing graders."""
        from ash_hawk.types import EvalTranscript, TokenUsage

        return EvalTranscript(
            messages=self.messages,
            tool_calls=self.tool_calls,
            trace_events=self.trace_events,
            token_usage=TokenUsage(
                input=self.token_usage.get("input", 0),
                output=self.token_usage.get("output", 0),
                reasoning=self.token_usage.get("reasoning", 0),
                cache_read=self.token_usage.get("cache_read", 0),
                cache_write=self.token_usage.get("cache_write", 0),
            ),
            cost_usd=self.cost_usd,
            duration_seconds=self.duration_seconds,
            agent_response=self.agent_response,
            error_trace=self.error_trace,
        )


@dataclass
class OutcomeData:
    """Captured outcome from agent run.

    Indicates whether the agent succeeded or failed.
    """

    success: bool
    message: str = ""
    error: str | None = None

    def to_eval_outcome(self) -> Any:
        """Convert to EvalOutcome for compatibility."""
        from ash_hawk.types import EvalOutcome, FailureMode

        if self.success:
            return EvalOutcome.success()
        return EvalOutcome.failure(
            FailureMode.AGENT_ERROR,
            self.error or self.message,
        )


@dataclass
class RunResult:
    """Result from run_real_agent.

    Contains transcript, outcome, and provenance manifest.
    """

    transcript: TranscriptData
    outcome: OutcomeData
    run_id: str = ""
    iterations: int = 0
    tools_used: list[str] = field(default_factory=list)
    manifest: RunManifest | None = None


class RunManifest(pd.BaseModel):
    """Provenance identity for a thin run.

    Captures all config hashes, variant tag, and metadata needed to
    reproduce or compare runs. Every ``thin run`` writes a manifest
    alongside the transcript.
    """

    run_id: str = pd.Field(description="Unique run identifier")
    scenario_path: str = pd.Field(description="Path to scenario YAML")
    scenario_hash: str = pd.Field(description="SHA-256 of scenario file content")
    agent_path: str = pd.Field(description="Path to agent directory")
    agent_hash: str = pd.Field(
        default="", description="SHA-256 of primary agent file (agent.md / AGENT.md)"
    )
    skill_hashes: dict[str, str] = pd.Field(
        default_factory=dict, description="{skill_name: SHA-256 of skill file}"
    )
    tool_hashes: dict[str, str] = pd.Field(
        default_factory=dict, description="{tool_name: SHA-256 of tool file}"
    )
    policy_hash: str = pd.Field(default="", description="SHA-256 of policy.md if present")
    model_name: str = pd.Field(default="", description="Model used for the run")
    variant: str = pd.Field(default="", description="Free-form variant tag (--variant flag)")
    seed: int | None = pd.Field(default=None, description="Deterministic seed if provided")
    grader_set: list[str] = pd.Field(default_factory=list, description="Grader type names used")
    timestamp: str = pd.Field(description="ISO 8601 UTC timestamp")
    ash_hawk_version: str = pd.Field(default="0.1.1", description="Ash-hawk harness version")

    model_config = pd.ConfigDict(extra="forbid")


class DiffFieldChange(pd.BaseModel):
    field: str = pd.Field(description="Manifest field name")
    baseline: str = pd.Field(default="", description="Value in baseline run")
    candidate: str = pd.Field(default="", description="Value in candidate run")


class DiffReport(pd.BaseModel):
    """Structured comparison between two thin runs.

    Produced by ``ash-hawk thin diff`` to answer: what changed,
    did it help, and should I keep it?
    """

    baseline_run_id: str = pd.Field(description="Run ID of the baseline")
    candidate_run_id: str = pd.Field(description="Run ID of the candidate")
    baseline_score: float | None = pd.Field(
        default=None, description="Aggregate score of baseline run"
    )
    candidate_score: float | None = pd.Field(
        default=None, description="Aggregate score of candidate run"
    )
    score_delta: float | None = pd.Field(
        default=None, description="candidate_score - baseline_score"
    )
    field_changes: list[DiffFieldChange] = pd.Field(
        default_factory=list, description="Manifest fields that differ"
    )
    grader_deltas: dict[str, dict[str, Any]] = pd.Field(
        default_factory=dict,
        description="{grader_type: {baseline_score, candidate_score, delta, flipped}}",
    )
    recommendation: str = pd.Field(
        default="",
        description="keep / reject / inconclusive recommendation",
    )
    timestamp: str = pd.Field(description="ISO 8601 UTC timestamp")

    model_config = pd.ConfigDict(extra="forbid")


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file's contents.

    Args:
        path: File to hash.

    Returns:
        Hex-encoded SHA-256 digest, or empty string if file cannot be read.
    """
    try:
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()
    except OSError:
        return ""


def compute_directory_hashes(
    directory: Path,
    extensions: frozenset[str] | None = None,
) -> dict[str, str]:
    """Compute SHA-256 hashes for all text files in a directory tree.

    Scans recursively for files matching the given extensions and returns
    a mapping from relative path (posix) to hex digest.

    Args:
        directory: Root directory to scan.
        extensions: File extensions to include (with leading dot).
            Defaults to ``_HASHABLE_EXTENSIONS``.

    Returns:
        Dict mapping relative posix path to SHA-256 hex digest.
    """
    if extensions is None:
        extensions = _HASHABLE_EXTENSIONS

    if not directory.is_dir():
        return {}

    hashes: dict[str, str] = {}
    for child in sorted(directory.rglob("*")):
        if not child.is_file():
            continue
        if child.suffix.lower() not in extensions:
            continue
        rel = child.relative_to(directory).as_posix()
        digest = compute_file_hash(child)
        if digest:
            hashes[rel] = digest

    return hashes


def build_run_manifest(
    *,
    run_id: str | None,
    scenario_path: Path,
    agent_path: Path,
    model_name: str,
    variant: str = "",
    seed: int | None = None,
    grader_set: list[str] | None = None,
    ash_hawk_version: str = "0.1.1",
) -> RunManifest:
    """Build a RunManifest with provenance hashes for a thin run.

    Computes SHA-256 hashes of the scenario file, agent directory,
    and all discoverable skill/tool/policy files.

    Args:
        run_id: Unique run ID (generated if not provided).
        scenario_path: Path to the scenario YAML file.
        agent_path: Path to the agent directory.
        model_name: LLM model identifier used for the run.
        variant: Free-form variant tag from ``--variant`` flag.
        seed: Optional deterministic seed.
        grader_set: List of grader type names.
        ash_hawk_version: Current ash-hawk version.

    Returns:
        Populated RunManifest ready for serialization.
    """
    effective_run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"

    scenario_hash = compute_file_hash(scenario_path)

    agent_hash = ""
    for candidate in ("AGENT.md", "agent.md"):
        agent_file = agent_path / candidate
        if agent_file.is_file():
            agent_hash = compute_file_hash(agent_file)
            break

    skill_hashes: dict[str, str] = {}
    skills_dir = agent_path / "skills"
    if skills_dir.is_dir():
        skill_hashes = compute_directory_hashes(skills_dir)

    tool_hashes: dict[str, str] = {}
    tools_dir = agent_path / "tools"
    if tools_dir.is_dir():
        tool_hashes = compute_directory_hashes(tools_dir)

    policy_hash = ""
    for candidate in ("policy.md", "POLICY.md"):
        policy_file = agent_path / candidate
        if policy_file.is_file():
            policy_hash = compute_file_hash(policy_file)
            break

    return RunManifest(
        run_id=effective_run_id,
        scenario_path=str(scenario_path),
        scenario_hash=scenario_hash,
        agent_path=str(agent_path),
        agent_hash=agent_hash,
        skill_hashes=skill_hashes,
        tool_hashes=tool_hashes,
        policy_hash=policy_hash,
        model_name=model_name,
        variant=variant,
        seed=seed,
        grader_set=grader_set or [],
        timestamp=datetime.now(UTC).isoformat(),
        ash_hawk_version=ash_hawk_version,
    )


async def run_real_agent(
    agent_path: Path,
    input: str,
    telemetry_sink: TelemetrySink,
    fixtures: dict[str, Path] | None = None,
    overlays: dict[str, str] | None = None,
    *,
    max_iterations: int = 10,
    workdir: Path | None = None,
    run_id: str | None = None,
) -> RunResult:
    """Run a real dawn-kestrel agent_v2 with telemetry capture.

    This is the thin bridge - it does NOT construct execution contexts.
    It loads the real agent from disk and runs it with telemetry hooks.

    Args:
        agent_path: Path to .dawn-kestrel/agents/{name} directory
        input: User prompt/task for the agent
        telemetry_sink: Callback for telemetry events
        fixtures: Optional fixture paths to inject into workdir
        overlays: Optional candidate content to test before write-back
        max_iterations: Maximum iterations for the agent loop
        workdir: Working directory for agent execution (defaults to cwd)
        run_id: Optional run ID for tracking

    Returns:
        RunResult containing transcript and outcome

    Raises:
        ImportError: If dawn-kestrel is not installed
        ValueError: If agent_path does not exist or is invalid
    """
    from ash_hawk.bridge.dawn_kestrel import DawnKestrelBridge

    bridge = DawnKestrelBridge(
        agent_path=agent_path,
        telemetry_sink=telemetry_sink,
        max_iterations=max_iterations,
        workdir=workdir,
        run_id=run_id,
    )
    return await bridge.run(input, fixtures=fixtures, overlays=overlays)
