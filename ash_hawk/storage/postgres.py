"""PostgreSQL storage backend for Ash Hawk using asyncpg."""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from ash_hawk.storage import StoredTrial
from ash_hawk.types import (
    EvalRunSummary,
    EvalSuite,
    EvalTrial,
    RunEnvelope,
    ToolSurfacePolicy,
    TrialEnvelope,
)


class PostgresStorage:
    """PostgreSQL storage backend using asyncpg with connection pooling.

    Schema:
        - suites: Suite definitions with JSONB data
        - run_envelopes: Run envelope metadata
        - trials: Trial data with JSONB columns
        - summaries: Run summary data

    All tables use IF NOT EXISTS for idempotent schema creation.
    """

    def __init__(
        self,
        connection_string: str,
        *,
        min_pool_size: int = 5,
        max_pool_size: int = 20,
    ) -> None:
        """Initialize PostgreSQL storage.

        Args:
            connection_string: PostgreSQL connection string
                (e.g., postgresql://user:pass@host:port/dbname)
            min_pool_size: Minimum number of connections in the pool
            max_pool_size: Maximum number of connections in the pool
        """
        self._connection_string = connection_string
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create the connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._connection_string,
                min_size=self._min_pool_size,
                max_size=self._max_pool_size,
            )
        return self._pool

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def initialize_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS suites (
                    suite_id TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS run_envelopes (
                    suite_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    data JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (suite_id, run_id)
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trials (
                    suite_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    trial_id TEXT NOT NULL,
                    trial_data JSONB NOT NULL,
                    envelope_data JSONB NOT NULL,
                    policy_data JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (suite_id, run_id, trial_id)
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    suite_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    data JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (suite_id, run_id)
                )
            """)

            # Create indexes for efficient queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_suites_created_at ON suites(created_at)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_run_envelopes_suite_id ON run_envelopes(suite_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_run_envelopes_created_at ON run_envelopes(created_at)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trials_suite_id ON trials(suite_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trials_run_id ON trials(run_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trials_trial_id ON trials(trial_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_summaries_created_at ON summaries(created_at)
            """)

    async def save_suite(self, suite: EvalSuite) -> None:
        """Save an evaluation suite definition."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO suites (suite_id, data)
                VALUES ($1, $2)
                ON CONFLICT (suite_id) DO UPDATE SET data = EXCLUDED.data
                """,
                suite.id,
                json.dumps(suite.model_dump(mode="json")),
            )

    async def load_suite(self, suite_id: str) -> EvalSuite | None:
        """Load an evaluation suite by ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM suites WHERE suite_id = $1",
                suite_id,
            )
            if row is None:
                return None
            return EvalSuite.model_validate(row["data"])

    async def save_run_envelope(self, suite_id: str, envelope: RunEnvelope) -> None:
        """Save a run envelope."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO run_envelopes (suite_id, run_id, data)
                VALUES ($1, $2, $3)
                ON CONFLICT (suite_id, run_id) DO UPDATE SET data = EXCLUDED.data
                """,
                suite_id,
                envelope.run_id,
                json.dumps(envelope.model_dump(mode="json")),
            )

    async def load_run_envelope(self, suite_id: str, run_id: str) -> RunEnvelope | None:
        """Load a run envelope."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM run_envelopes WHERE suite_id = $1 AND run_id = $2",
                suite_id,
                run_id,
            )
            if row is None:
                return None
            return RunEnvelope.model_validate(row["data"])

    async def save_trial(
        self,
        suite_id: str,
        run_id: str,
        trial: EvalTrial,
        envelope: TrialEnvelope,
        policy: ToolSurfacePolicy,
    ) -> None:
        """Save a trial with its envelope and policy."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO trials (suite_id, run_id, trial_id, trial_data, envelope_data, policy_data)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (suite_id, run_id, trial_id) DO UPDATE SET
                    trial_data = EXCLUDED.trial_data,
                    envelope_data = EXCLUDED.envelope_data,
                    policy_data = EXCLUDED.policy_data
                """,
                suite_id,
                run_id,
                trial.id,
                json.dumps(trial.model_dump(mode="json")),
                json.dumps(envelope.model_dump(mode="json")),
                json.dumps(policy.model_dump(mode="json")),
            )

    async def load_trial(self, suite_id: str, run_id: str, trial_id: str) -> StoredTrial | None:
        """Load a stored trial with all associated data."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT trial_data, envelope_data, policy_data
                FROM trials
                WHERE suite_id = $1 AND run_id = $2 AND trial_id = $3
                """,
                suite_id,
                run_id,
                trial_id,
            )
            if row is None:
                return None
            return StoredTrial(
                trial=EvalTrial.model_validate(row["trial_data"]),
                envelope=TrialEnvelope.model_validate(row["envelope_data"]),
                policy=ToolSurfacePolicy.model_validate(row["policy_data"]),
            )

    async def list_runs(self, suite_id: str) -> list[str]:
        """List all run IDs for a suite."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT run_id FROM run_envelopes
                WHERE suite_id = $1
                ORDER BY created_at
                """,
                suite_id,
            )
            return [row["run_id"] for row in rows]

    async def list_suites(self) -> list[str]:
        """List all suite IDs."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT suite_id FROM suites ORDER BY created_at")
            return [row["suite_id"] for row in rows]

    async def save_summary(self, suite_id: str, run_id: str, summary: EvalRunSummary) -> None:
        """Save a run summary."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO summaries (suite_id, run_id, data)
                VALUES ($1, $2, $3)
                ON CONFLICT (suite_id, run_id) DO UPDATE SET data = EXCLUDED.data
                """,
                suite_id,
                run_id,
                json.dumps(summary.model_dump(mode="json")),
            )

    async def load_summary(self, suite_id: str, run_id: str) -> EvalRunSummary | None:
        """Load a run summary."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM summaries WHERE suite_id = $1 AND run_id = $2",
                suite_id,
                run_id,
            )
            if row is None:
                return None
            return EvalRunSummary.model_validate(row["data"])

    async def __aenter__(self) -> "PostgresStorage":
        """Async context manager entry."""
        await self._get_pool()
        await self.initialize_schema()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
