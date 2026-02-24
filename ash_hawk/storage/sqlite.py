"""SQLite-based storage backend for Ash Hawk using aiosqlite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite
from pydantic import BaseModel

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
            result[field_name] = {
                k: _dump_model(v) if isinstance(v, BaseModel) else v for k, v in value.items()
            }
        elif isinstance(value, list):
            result[field_name] = [
                _dump_model(item) if isinstance(item, BaseModel) else item for item in value
            ]
        else:
            result[field_name] = value
    return result


class SQLiteStorage:
    """SQLite-based storage backend using aiosqlite.

    Schema:
        - suites: id (PK), name, description, data (JSON)
        - run_envelopes: suite_id, run_id, data (JSON)
        - trials: suite_id, run_id, trial_id, trial_data (JSON),
                  envelope_data (JSON), policy_data (JSON)
        - summaries: suite_id, run_id, data (JSON)
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialize SQLite storage.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def _get_db(self) -> aiosqlite.Connection:
        """Get or create the database connection."""
        if self._db is None:
            # Ensure parent directory exists
            if self._db_path.parent:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row
            await self._init_schema()
        return self._db

    async def _init_schema(self) -> None:
        """Initialize the database schema."""
        db = await self._get_db()
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS suites (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                data TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS run_envelopes (
                suite_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY (suite_id, run_id),
                FOREIGN KEY (suite_id) REFERENCES suites(id)
            );

            CREATE TABLE IF NOT EXISTS trials (
                suite_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                trial_id TEXT NOT NULL,
                trial_data TEXT NOT NULL,
                envelope_data TEXT NOT NULL,
                policy_data TEXT NOT NULL,
                PRIMARY KEY (suite_id, run_id, trial_id),
                FOREIGN KEY (suite_id, run_id) REFERENCES run_envelopes(suite_id, run_id)
            );

            CREATE TABLE IF NOT EXISTS summaries (
                suite_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY (suite_id, run_id),
                FOREIGN KEY (suite_id, run_id) REFERENCES run_envelopes(suite_id, run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_trials_suite ON trials(suite_id);
            CREATE INDEX IF NOT EXISTS idx_envelopes_suite ON run_envelopes(suite_id);
        """)
        await db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def save_suite(self, suite: EvalSuite) -> None:
        """Save an evaluation suite definition."""
        db = await self._get_db()
        data = json.dumps(_dump_model(suite))
        await db.execute(
            """
            INSERT OR REPLACE INTO suites (id, name, description, data)
            VALUES (?, ?, ?, ?)
            """,
            (suite.id, suite.name, suite.description, data),
        )
        await db.commit()

    async def load_suite(self, suite_id: str) -> EvalSuite | None:
        """Load an evaluation suite by ID."""
        db = await self._get_db()
        async with db.execute("SELECT data FROM suites WHERE id = ?", (suite_id,)) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return EvalSuite.model_validate(json.loads(row["data"]))

    async def save_run_envelope(self, suite_id: str, envelope: RunEnvelope) -> None:
        """Save a run envelope."""
        db = await self._get_db()
        data = json.dumps(_dump_model(envelope))
        await db.execute(
            """
            INSERT OR REPLACE INTO run_envelopes (suite_id, run_id, data)
            VALUES (?, ?, ?)
            """,
            (suite_id, envelope.run_id, data),
        )
        await db.commit()

    async def load_run_envelope(self, suite_id: str, run_id: str) -> RunEnvelope | None:
        """Load a run envelope."""
        db = await self._get_db()
        async with db.execute(
            "SELECT data FROM run_envelopes WHERE suite_id = ? AND run_id = ?",
            (suite_id, run_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return RunEnvelope.model_validate(json.loads(row["data"]))

    async def save_trial(
        self,
        suite_id: str,
        run_id: str,
        trial: EvalTrial,
        envelope: TrialEnvelope,
        policy: ToolSurfacePolicy,
    ) -> None:
        """Save a trial with its envelope and policy."""
        db = await self._get_db()
        trial_data = json.dumps(_dump_model(trial))
        envelope_data = json.dumps(_dump_model(envelope))
        policy_data = json.dumps(_dump_model(policy))
        await db.execute(
            """
            INSERT OR REPLACE INTO trials
            (suite_id, run_id, trial_id, trial_data, envelope_data, policy_data)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (suite_id, run_id, trial.id, trial_data, envelope_data, policy_data),
        )
        await db.commit()

    async def load_trial(self, suite_id: str, run_id: str, trial_id: str) -> StoredTrial | None:
        """Load a stored trial with all associated data."""
        db = await self._get_db()
        async with db.execute(
            """
            SELECT trial_data, envelope_data, policy_data
            FROM trials
            WHERE suite_id = ? AND run_id = ? AND trial_id = ?
            """,
            (suite_id, run_id, trial_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return StoredTrial(
                trial=EvalTrial.model_validate(json.loads(row["trial_data"])),
                envelope=TrialEnvelope.model_validate(json.loads(row["envelope_data"])),
                policy=ToolSurfacePolicy.model_validate(json.loads(row["policy_data"])),
            )

    async def list_runs(self, suite_id: str) -> list[str]:
        """List all run IDs for a suite."""
        db = await self._get_db()
        async with db.execute(
            "SELECT run_id FROM run_envelopes WHERE suite_id = ? ORDER BY run_id",
            (suite_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [row["run_id"] for row in rows]

    async def list_suites(self) -> list[str]:
        """List all suite IDs."""
        db = await self._get_db()
        async with db.execute("SELECT id FROM suites ORDER BY id") as cursor:
            rows = await cursor.fetchall()
            return [row["id"] for row in rows]

    async def save_summary(self, suite_id: str, run_id: str, summary: EvalRunSummary) -> None:
        """Save a run summary."""
        db = await self._get_db()
        data = json.dumps(_dump_model(summary))
        await db.execute(
            """
            INSERT OR REPLACE INTO summaries (suite_id, run_id, data)
            VALUES (?, ?, ?)
            """,
            (suite_id, run_id, data),
        )
        await db.commit()

    async def load_summary(self, suite_id: str, run_id: str) -> EvalRunSummary | None:
        """Load a run summary."""
        db = await self._get_db()
        async with db.execute(
            "SELECT data FROM summaries WHERE suite_id = ? AND run_id = ?",
            (suite_id, run_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return EvalRunSummary.model_validate(json.loads(row["data"]))

    # Additional query methods for filtering

    async def query_trials_by_suite(self, suite_id: str) -> list[StoredTrial]:
        """Query all trials for a suite."""
        db = await self._get_db()
        async with db.execute(
            """
            SELECT trial_data, envelope_data, policy_data
            FROM trials
            WHERE suite_id = ?
            ORDER BY run_id, trial_id
            """,
            (suite_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                StoredTrial(
                    trial=EvalTrial.model_validate(json.loads(row["trial_data"])),
                    envelope=TrialEnvelope.model_validate(json.loads(row["envelope_data"])),
                    policy=ToolSurfacePolicy.model_validate(json.loads(row["policy_data"])),
                )
                for row in rows
            ]

    async def query_trials_by_task(self, suite_id: str, task_id: str) -> list[StoredTrial]:
        """Query all trials for a specific task within a suite."""
        db = await self._get_db()
        # Use JSON extraction to filter by task_id within trial_data
        async with db.execute(
            """
            SELECT trial_data, envelope_data, policy_data
            FROM trials
            WHERE suite_id = ? AND json_extract(trial_data, '$.task_id') = ?
            ORDER BY run_id, trial_id
            """,
            (suite_id, task_id),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                StoredTrial(
                    trial=EvalTrial.model_validate(json.loads(row["trial_data"])),
                    envelope=TrialEnvelope.model_validate(json.loads(row["envelope_data"])),
                    policy=ToolSurfacePolicy.model_validate(json.loads(row["policy_data"])),
                )
                for row in rows
            ]

    async def query_trials_by_status(self, suite_id: str, status: str) -> list[StoredTrial]:
        """Query all trials with a specific status within a suite."""
        db = await self._get_db()
        async with db.execute(
            """
            SELECT trial_data, envelope_data, policy_data
            FROM trials
            WHERE suite_id = ? AND json_extract(trial_data, '$.status') = ?
            ORDER BY run_id, trial_id
            """,
            (suite_id, status),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                StoredTrial(
                    trial=EvalTrial.model_validate(json.loads(row["trial_data"])),
                    envelope=TrialEnvelope.model_validate(json.loads(row["envelope_data"])),
                    policy=ToolSurfacePolicy.model_validate(json.loads(row["policy_data"])),
                )
                for row in rows
            ]

    async def query_trials_by_time_range(
        self,
        suite_id: str,
        start_time: str,
        end_time: str,
    ) -> list[StoredTrial]:
        """Query trials within a time range based on envelope created_at."""
        db = await self._get_db()
        async with db.execute(
            """
            SELECT trial_data, envelope_data, policy_data
            FROM trials
            WHERE suite_id = ?
            AND json_extract(envelope_data, '$.created_at') >= ?
            AND json_extract(envelope_data, '$.created_at') <= ?
            ORDER BY run_id, trial_id
            """,
            (suite_id, start_time, end_time),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                StoredTrial(
                    trial=EvalTrial.model_validate(json.loads(row["trial_data"])),
                    envelope=TrialEnvelope.model_validate(json.loads(row["envelope_data"])),
                    policy=ToolSurfacePolicy.model_validate(json.loads(row["policy_data"])),
                )
                for row in rows
            ]
