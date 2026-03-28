from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MockToolCall:
    tool_name: str
    outcome: str = "success"
    error_message: str | None = None


@dataclass
class MockRunArtifact:
    run_id: str
    outcome: str = "success"
    tool_calls: list[MockToolCall] = field(default_factory=list)
