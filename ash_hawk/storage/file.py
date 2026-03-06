"""File-based storage backend for Ash Hawk using JSON."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, cast

import aiofiles
import aiofiles.os
from pydantic import BaseModel

from ash_hawk.scenario.trace import TraceEvent, write_trace_jsonl
from ash_hawk.storage import StoredTrial
from ash_hawk.types import (
    EvalRunSummary,
    EvalSuite,
    EvalTrial,
    RunEnvelope,
    ToolSurfacePolicy,
    TrialEnvelope,
)


def _dump_model(model: BaseModel) -> dict[str, Any]:
    model_type = type(model)
    result: dict[str, Any] = {}
    computed_fields = model_type.model_computed_fields.keys()
    for field_name in model_type.model_fields.keys():
        if field_name in computed_fields:
            continue
        value = getattr(model, field_name)
        if isinstance(value, BaseModel):
            result[field_name] = _dump_model(value)
        elif isinstance(value, dict):
            nested: dict[str, Any] = {}
            for raw_key, raw_val in value.items():
                key = str(raw_key)
                val: Any = raw_val
                nested[key] = _dump_model(val) if isinstance(val, BaseModel) else val
            result[field_name] = nested
        elif isinstance(value, list):
            nested_list: list[Any] = []
            for raw_item in value:
                item: Any = raw_item
                nested_list.append(_dump_model(item) if isinstance(item, BaseModel) else item)
            result[field_name] = nested_list
        else:
            result[field_name] = value
    return result


class FileStorage:
    """File-based storage backend using JSON with atomic writes.

    Directory structure:
        {base_path}/{suite_id}/suite.json
        {base_path}/{suite_id}/runs/{run_id}/envelope.json
        {base_path}/{suite_id}/runs/{run_id}/trials/{trial_id}.json
        {base_path}/{suite_id}/runs/{run_id}/summary.json
    """

    def __init__(self, base_path: str | Path) -> None:
        self._base_path = Path(base_path)

    def _suite_path(self, suite_id: str) -> Path:
        return self._base_path / suite_id

    def _suite_file(self, suite_id: str) -> Path:
        return self._suite_path(suite_id) / "suite.json"

    def _runs_path(self, suite_id: str) -> Path:
        return self._suite_path(suite_id) / "runs"

    def _run_path(self, suite_id: str, run_id: str) -> Path:
        return self._runs_path(suite_id) / run_id

    def _envelope_file(self, suite_id: str, run_id: str) -> Path:
        return self._run_path(suite_id, run_id) / "envelope.json"

    def _trials_path(self, suite_id: str, run_id: str) -> Path:
        return self._run_path(suite_id, run_id) / "trials"

    def _trial_file(self, suite_id: str, run_id: str, trial_id: str) -> Path:
        return self._trials_path(suite_id, run_id) / f"{trial_id}.json"

    def _trial_trace_file(self, suite_id: str, run_id: str, trial_id: str) -> Path:
        return self._trials_path(suite_id, run_id) / f"{trial_id}.trace.jsonl"

    def _summary_file(self, suite_id: str, run_id: str) -> Path:
        return self._run_path(suite_id, run_id) / "summary.json"

    async def _ensure_dir(self, path: Path) -> None:
        await aiofiles.os.makedirs(str(path), exist_ok=True)

    async def _atomic_write(self, path: Path, content: str) -> None:
        await self._ensure_dir(path.parent)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        async with aiofiles.open(temp_path, "w") as f:
            await f.write(content)
        await aiofiles.os.replace(str(temp_path), str(path))

    async def _atomic_write_trace(self, path: Path, events: list[TraceEvent]) -> None:
        await self._ensure_dir(path.parent)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")

        def _write_and_replace() -> None:
            write_trace_jsonl(temp_path, events)
            os.replace(temp_path, path)

        await asyncio.to_thread(_write_and_replace)

    async def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            async with aiofiles.open(path, "r") as f:
                content = await f.read()
            return json.loads(content)
        except FileNotFoundError:
            return None

    async def save_suite(self, suite: EvalSuite) -> None:
        await self._atomic_write(
            self._suite_file(suite.id),
            json.dumps(_dump_model(suite), indent=2),
        )

    async def load_suite(self, suite_id: str) -> EvalSuite | None:
        data = await self._read_json(self._suite_file(suite_id))
        if data is None:
            return None
        return EvalSuite.model_validate(data)

    async def save_run_envelope(self, suite_id: str, envelope: RunEnvelope) -> None:
        await self._atomic_write(
            self._envelope_file(suite_id, envelope.run_id),
            envelope.model_dump_json(indent=2),
        )

    async def load_run_envelope(self, suite_id: str, run_id: str) -> RunEnvelope | None:
        data = await self._read_json(self._envelope_file(suite_id, run_id))
        if data is None:
            return None
        return RunEnvelope.model_validate(data)

    async def save_trial(
        self,
        suite_id: str,
        run_id: str,
        trial: EvalTrial,
        envelope: TrialEnvelope,
        policy: ToolSurfacePolicy,
    ) -> None:
        stored = {
            "trial": _dump_model(trial),
            "envelope": _dump_model(envelope),
            "policy": _dump_model(policy),
        }
        await self._atomic_write(
            self._trial_file(suite_id, run_id, trial.id),
            json.dumps(stored, indent=2),
        )

        trace_events = []
        if trial.result is not None:
            trace_events = trial.result.transcript.trace_events
        if trace_events:
            events = [TraceEvent.model_validate(event) for event in trace_events]
            await self._atomic_write_trace(
                self._trial_trace_file(suite_id, run_id, trial.id),
                events,
            )

    async def load_trial(self, suite_id: str, run_id: str, trial_id: str) -> StoredTrial | None:
        data = await self._read_json(self._trial_file(suite_id, run_id, trial_id))
        if data is None:
            return None
        return StoredTrial(
            trial=EvalTrial.model_validate(data["trial"]),
            envelope=TrialEnvelope.model_validate(data["envelope"]),
            policy=ToolSurfacePolicy.model_validate(data["policy"]),
        )

    async def list_runs(self, suite_id: str) -> list[str]:
        runs_path = self._runs_path(suite_id)
        try:
            entries: list[str] = [str(entry) for entry in await aiofiles.os.listdir(str(runs_path))]
            return sorted(entries)
        except FileNotFoundError:
            return []

    async def list_suites(self) -> list[str]:
        try:
            entries: list[str] = [
                str(entry) for entry in await aiofiles.os.listdir(str(self._base_path))
            ]
            suite_ids: list[str] = []
            for entry in entries:
                suite_file = self._suite_file(entry)
                try:
                    await aiofiles.os.stat(str(suite_file))
                    suite_ids.append(entry)
                except FileNotFoundError:
                    pass
            return sorted(suite_ids)
        except FileNotFoundError:
            return []

    async def save_summary(self, suite_id: str, run_id: str, summary: EvalRunSummary) -> None:
        await self._atomic_write(
            self._summary_file(suite_id, run_id),
            json.dumps(_dump_model(summary), indent=2),
        )

    async def load_summary(self, suite_id: str, run_id: str) -> EvalRunSummary | None:
        data = await self._read_json(self._summary_file(suite_id, run_id))
        if data is None:
            return None
        return EvalRunSummary.model_validate(data)
