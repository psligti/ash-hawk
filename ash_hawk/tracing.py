# type-hygiene: skip-file
"""Structured tracing for Ash Hawk evaluation runs."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A single trace span."""

    name: str
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # ok | error | timeout

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or time.monotonic()
        return end - self.start_time

    def finish(self, status: str = "ok") -> None:
        """Finish this span. Guard against double-call — if already finished, no-op."""
        if self.end_time is not None:
            return
        self.end_time = time.monotonic()
        self.status = status

    def add_event(self, name: str, **attrs: Any) -> None:
        self.events.append({"name": name, "ts": time.monotonic(), **attrs})


class TraceContext:
    """Accumulates spans during a trial run."""

    def __init__(self, trial_id: str, run_id: str | None = None) -> None:
        self.trial_id = trial_id
        self.run_id = run_id
        self._spans: list[Span] = []
        self._stack: list[Span] = []

    def start_span(self, name: str, **attrs: Any) -> Span:
        span = Span(name=name, attributes=attrs)
        self._spans.append(span)
        self._stack.append(span)
        return span

    def end_span(self, status: str = "ok") -> Span | None:
        if not self._stack:
            return None
        span = self._stack.pop()
        span.finish(status)
        return span

    @property
    def current_span(self) -> Span | None:
        return self._stack[-1] if self._stack else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "run_id": self.run_id,
            "spans": [
                {
                    "name": s.name,
                    "duration_s": round(s.duration_seconds, 4),
                    "status": s.status,
                    "attributes": s.attributes,
                    "events": s.events,
                }
                for s in self._spans
            ],
        }

    def write_jsonl(self, path: Path) -> None:
        """Write trace as JSONL. Catches OSError and logs warning."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(self.to_dict()) + "\n")
        except OSError:
            logger.warning("Failed to write trace JSONL", extra={"path": str(path)})


class TelemetryLog:
    """Structured JSONL telemetry logger for eval runs.

    Writes one JSON line per event to ``.ash-hawk/telemetry.jsonl`` (or
    whatever path is configured via ``ASH_HAWK_TELEMETRY_PATH``).  Every
    event includes an ISO timestamp, event kind, and a flat payload dict.

    The file is opened lazily on first write so merely importing this module
    has zero I/O cost.
    """

    def __init__(self) -> None:
        self._path: Path | None = None
        self._file_handle: Any = None

    @property
    def path(self) -> Path:
        if self._path is not None:
            return self._path
        env = os.getenv("ASH_HAWK_TELEMETRY_PATH", "")
        if env:
            self._path = Path(env).expanduser().resolve()
        else:
            storage = os.getenv("ASH_HAWK_STORAGE_PATH", ".ash-hawk")
            self._path = Path(storage).expanduser().resolve() / "telemetry.jsonl"
        return self._path

    def _ensure_handle(self) -> Any:
        if self._file_handle is not None:
            return self._file_handle
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self.path, "a")  # noqa: SIM115
        except OSError:
            logger.warning("telemetry: cannot open %s — telemetry disabled", self.path)
        return self._file_handle

    def emit(self, event: str, **payload: Any) -> None:
        handle = self._ensure_handle()
        if handle is None:
            return
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
        }
        if payload:
            record["payload"] = payload
        try:
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
        except OSError:
            logger.warning("telemetry: write failed")

    def close(self) -> None:
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except OSError:
                pass
            self._file_handle = None


_telemetry: TelemetryLog | None = None


def get_telemetry() -> TelemetryLog:
    global _telemetry
    if _telemetry is None:
        _telemetry = TelemetryLog()
    return _telemetry


def reset_telemetry() -> None:
    global _telemetry
    if _telemetry is not None:
        _telemetry.close()
    _telemetry = None
