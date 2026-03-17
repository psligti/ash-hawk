"""Persistent provenance tracker using SQLite backend."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite

from ash_hawk.contracts import CuratedLesson, ImprovementProposal


class ProvenanceRecord:
    """Tracks the provenance of a curated lesson."""

    def __init__(
        self,
        lesson_id: str,
        source_proposal_id: str,
        origin_run_id: str,
        origin_review_id: str | None = None,
    ) -> None:
        self.lesson_id = lesson_id
        self.source_proposal_id = source_proposal_id
        self.origin_run_id = origin_run_id
        self.origin_review_id = origin_review_id
        self.created_at = datetime.now(UTC)
        self.metadata: dict[str, Any] = {}


class PersistentProvenanceTracker:
    """SQLite-backed provenance tracker for run-to-lesson lineage.

    Enables:
    - Tracing why a lesson exists
    - Finding all lessons from a specific run
    - Audit trail for behavioral changes
    - Durable storage across sessions
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else Path(".ash-hawk/provenance.db")
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            if self._db_path.parent:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row
        return self._db

    async def _init_schema(self) -> None:
        if self._initialized:
            return

        db = await self._get_db()
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS provenance (
                lesson_id TEXT PRIMARY KEY,
                source_proposal_id TEXT NOT NULL,
                origin_run_id TEXT NOT NULL,
                origin_review_id TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_provenance_run ON provenance(origin_run_id);
            CREATE INDEX IF NOT EXISTS idx_provenance_review ON provenance(origin_review_id);
            CREATE INDEX IF NOT EXISTS idx_provenance_proposal ON provenance(source_proposal_id);
        """)
        await db.commit()
        self._initialized = True

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        async with self._lock:
            db = await self._get_db()
            await db.execute("BEGIN")
            try:
                yield db
                await db.commit()
            except Exception:
                await db.rollback()
                raise

    async def track(
        self,
        lesson: CuratedLesson,
        proposal: ImprovementProposal,
    ) -> ProvenanceRecord:
        await self._init_schema()
        record = ProvenanceRecord(
            lesson_id=lesson.lesson_id,
            source_proposal_id=proposal.proposal_id,
            origin_run_id=proposal.origin_run_id,
            origin_review_id=proposal.origin_review_id,
        )

        async with self._transaction() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO provenance
                (lesson_id, source_proposal_id, origin_run_id, origin_review_id, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.lesson_id,
                    record.source_proposal_id,
                    record.origin_run_id,
                    record.origin_review_id,
                    record.created_at.isoformat(),
                    json.dumps(record.metadata),
                ),
            )

        return record

    async def get_lesson_provenance(self, lesson_id: str) -> ProvenanceRecord | None:
        await self._init_schema()
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM provenance WHERE lesson_id = ?", (lesson_id,))
        row = await cursor.fetchone()
        if row is None:
            return None

        record = ProvenanceRecord(
            lesson_id=row["lesson_id"],
            source_proposal_id=row["source_proposal_id"],
            origin_run_id=row["origin_run_id"],
            origin_review_id=row["origin_review_id"],
        )
        record.created_at = datetime.fromisoformat(row["created_at"])
        record.metadata = json.loads(row["metadata"])
        return record

    async def get_lessons_from_run(self, run_id: str) -> list[str]:
        await self._init_schema()
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT lesson_id FROM provenance WHERE origin_run_id = ?", (run_id,)
        )
        rows = await cursor.fetchall()
        return [row["lesson_id"] for row in rows]

    async def get_lessons_from_review(self, review_id: str) -> list[str]:
        await self._init_schema()
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT lesson_id FROM provenance WHERE origin_review_id = ?", (review_id,)
        )
        rows = await cursor.fetchall()
        return [row["lesson_id"] for row in rows]

    async def get_lineage(self, lesson_id: str) -> dict[str, Any]:
        record = await self.get_lesson_provenance(lesson_id)
        if not record:
            return {}

        return {
            "lesson_id": record.lesson_id,
            "source_proposal_id": record.source_proposal_id,
            "origin_run_id": record.origin_run_id,
            "origin_review_id": record.origin_review_id,
            "created_at": record.created_at.isoformat(),
        }

    async def export_audit_trail(self) -> list[dict[str, Any]]:
        await self._init_schema()
        db = await self._get_db()
        cursor = await db.execute("SELECT lesson_id FROM provenance")
        rows = await cursor.fetchall()
        return [await self.get_lineage(row["lesson_id"]) for row in rows]

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


__all__ = ["ProvenanceRecord", "PersistentProvenanceTracker"]
