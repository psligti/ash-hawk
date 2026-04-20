from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path


class ImproveRunBundle:
    def __init__(self, base_dir: Path | None = None, run_id: str | None = None) -> None:
        root = (base_dir or Path(".ash-hawk")).resolve()
        self.run_id = run_id or f"improve-{uuid.uuid4().hex[:12]}"
        self.path = root / "improve-runs" / self.run_id
        self.path.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative_path: str) -> Path:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative_path: str, data: object) -> Path:
        path = self._resolve(relative_path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
        tmp.replace(path)
        return path

    def write_text(self, relative_path: str, content: str) -> Path:
        path = self._resolve(relative_path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
        return path

    def append_event(self, event_type: str, **payload: object) -> Path:
        path = self._resolve("events.jsonl")
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event_type,
            "payload": payload,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return path

    def write_summary(self, content: str) -> Path:
        return self.write_text("summary.md", content)


__all__ = ["ImproveRunBundle"]
