"""Persistent lesson store using SQLite backend with concurrency support."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import aiosqlite

from ash_hawk.contracts import CuratedLesson


class PersistentLessonStore:
    """SQLite-backed lesson store with async operations and concurrency locking.

    Provides durable storage for curated lessons with:
    - Version history tracking
    - Agent-based indexing
    - Proposal-based lookup
    - Experiment scoping for parallel trial isolation
    - Thread-safe concurrent access via asyncio.Lock
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize the persistent lesson store.

        Args:
            db_path: Path to SQLite database file. Defaults to .ash-hawk/lessons.db
        """
        self._db_path = Path(db_path) if db_path else Path(".ash-hawk/lessons.db")
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def _get_db(self) -> aiosqlite.Connection:
        """Get or create the database connection."""
        if self._db is None:
            if self._db_path.parent:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row
        return self._db

    async def _init_schema(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return

        db = await self._get_db()
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS lessons (
                lesson_id TEXT PRIMARY KEY,
                source_proposal_id TEXT NOT NULL,
                applies_to_agents TEXT NOT NULL,
                lesson_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                lesson_payload TEXT NOT NULL,
                validation_status TEXT NOT NULL DEFAULT 'approved',
                version INTEGER NOT NULL DEFAULT 1,
                parent_lesson_id TEXT,
                rollback_of TEXT,
                evidence_summary TEXT,
                impact_metrics TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT,
                applied_at TEXT,
                experiment_id TEXT,
                strategy TEXT,
                sub_strategies TEXT NOT NULL DEFAULT '[]',
                UNIQUE(source_proposal_id)
            );

            CREATE INDEX IF NOT EXISTS idx_lessons_agent ON lessons(applies_to_agents);
            CREATE INDEX IF NOT EXISTS idx_lessons_status ON lessons(validation_status);
            CREATE INDEX IF NOT EXISTS idx_lessons_type ON lessons(lesson_type);
            CREATE INDEX IF NOT EXISTS idx_lessons_experiment ON lessons(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_lessons_strategy ON lessons(strategy);

            CREATE TABLE IF NOT EXISTS lesson_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lesson_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                snapshot TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (lesson_id) REFERENCES lessons(lesson_id)
            );

            CREATE INDEX IF NOT EXISTS idx_history_lesson ON lesson_history(lesson_id);
        """)
        await db.commit()
        self._initialized = True

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Context manager for database transactions with locking."""
        async with self._lock:
            db = await self._get_db()
            await self._init_schema()
            try:
                yield db
                await db.commit()
            except Exception:
                await db.rollback()
                raise

    async def store(self, lesson: CuratedLesson) -> str:
        """Store a lesson persistently.

        Args:
            lesson: The lesson to store.

        Returns:
            The lesson ID.
        """
        async with self._transaction() as db:
            # Snapshot current version if updating
            existing = await self._get_raw(db, lesson.lesson_id)
            if existing:
                await self._save_history(db, existing)

            await db.execute(
                """
                INSERT OR REPLACE INTO lessons (
                    lesson_id, source_proposal_id, applies_to_agents, lesson_type,
                    title, description, lesson_payload, validation_status, version,
                    parent_lesson_id, rollback_of, evidence_summary, impact_metrics,
                    created_at, updated_at, applied_at, experiment_id, strategy, sub_strategies
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lesson.lesson_id,
                    lesson.source_proposal_id,
                    json.dumps(lesson.applies_to_agents),
                    lesson.lesson_type,
                    lesson.title,
                    lesson.description,
                    json.dumps(lesson.lesson_payload),
                    lesson.validation_status,
                    lesson.version,
                    lesson.parent_lesson_id,
                    lesson.rollback_of,
                    lesson.evidence_summary,
                    json.dumps(lesson.impact_metrics),
                    lesson.created_at.isoformat() if lesson.created_at else None,
                    lesson.updated_at.isoformat() if lesson.updated_at else None,
                    lesson.applied_at.isoformat() if lesson.applied_at else None,
                    lesson.experiment_id,
                    str(lesson.strategy) if lesson.strategy else None,
                    json.dumps([str(s) for s in lesson.sub_strategies])
                    if lesson.sub_strategies
                    else "[]",
                ),
            )
        return lesson.lesson_id

    async def _save_history(self, db: aiosqlite.Connection, lesson: CuratedLesson) -> None:
        """Save a lesson snapshot to history."""
        snapshot = {
            "lesson_id": lesson.lesson_id,
            "source_proposal_id": lesson.source_proposal_id,
            "applies_to_agents": lesson.applies_to_agents,
            "lesson_type": lesson.lesson_type,
            "title": lesson.title,
            "description": lesson.description,
            "lesson_payload": lesson.lesson_payload,
            "validation_status": lesson.validation_status,
            "version": lesson.version,
            "parent_lesson_id": lesson.parent_lesson_id,
            "rollback_of": lesson.rollback_of,
            "evidence_summary": lesson.evidence_summary,
            "impact_metrics": lesson.impact_metrics,
            "created_at": lesson.created_at.isoformat() if lesson.created_at else None,
            "updated_at": lesson.updated_at.isoformat() if lesson.updated_at else None,
            "applied_at": lesson.applied_at.isoformat() if lesson.applied_at else None,
        }
        await db.execute(
            """
            INSERT INTO lesson_history (lesson_id, version, snapshot, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (lesson.lesson_id, lesson.version, json.dumps(snapshot), datetime.now(UTC).isoformat()),
        )

    def _row_to_lesson(self, row: aiosqlite.Row) -> CuratedLesson:
        """Convert a database row to CuratedLesson."""
        from ash_hawk.strategies import Strategy, SubStrategy

        strategy = None
        if row["strategy"]:
            try:
                strategy = Strategy(row["strategy"])
            except ValueError:
                pass

        sub_strategies = []
        if row["sub_strategies"]:
            try:
                for s in json.loads(row["sub_strategies"]):
                    try:
                        sub_strategies.append(SubStrategy(s))
                    except ValueError:
                        pass
            except json.JSONDecodeError:
                pass

        return CuratedLesson(
            lesson_id=row["lesson_id"],
            source_proposal_id=row["source_proposal_id"],
            applies_to_agents=json.loads(row["applies_to_agents"]),
            lesson_type=row["lesson_type"],
            title=row["title"],
            description=row["description"],
            lesson_payload=json.loads(row["lesson_payload"]),
            validation_status=row["validation_status"],
            version=row["version"],
            parent_lesson_id=row["parent_lesson_id"],
            rollback_of=row["rollback_of"],
            evidence_summary=row["evidence_summary"],
            impact_metrics=json.loads(row["impact_metrics"]),
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else datetime.now(UTC),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
            applied_at=datetime.fromisoformat(row["applied_at"]) if row["applied_at"] else None,
            experiment_id=row["experiment_id"],
            strategy=strategy,
            sub_strategies=sub_strategies,
        )

    async def _get_raw(self, db: aiosqlite.Connection, lesson_id: str) -> CuratedLesson | None:
        """Get a lesson without transaction context."""
        async with db.execute("SELECT * FROM lessons WHERE lesson_id = ?", (lesson_id,)) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_lesson(row)

    async def get(self, lesson_id: str) -> CuratedLesson | None:
        """Get a lesson by ID.

        Args:
            lesson_id: The lesson ID to look up.

        Returns:
            The lesson if found, None otherwise.
        """
        async with self._transaction() as db:
            return await self._get_raw(db, lesson_id)

    async def get_by_proposal(self, proposal_id: str) -> CuratedLesson | None:
        """Get a lesson by its source proposal ID.

        Args:
            proposal_id: The proposal ID to look up.

        Returns:
            The lesson if found, None otherwise.
        """
        async with self._transaction() as db:
            async with db.execute(
                "SELECT * FROM lessons WHERE source_proposal_id = ?", (proposal_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_lesson(row)

    async def get_for_agent(
        self,
        agent_id: str,
        experiment_id: str | None = None,
    ) -> list[CuratedLesson]:
        """Get all approved lessons for an agent.

        Args:
            agent_id: The agent ID to get lessons for.
            experiment_id: Optional experiment scope for isolation.

        Returns:
            List of approved lessons for the agent.
        """
        async with self._transaction() as db:
            if experiment_id:
                query = """
                    SELECT * FROM lessons
                    WHERE applies_to_agents LIKE ?
                    AND validation_status = 'approved'
                    AND experiment_id = ?
                """
                params = (f'%"{agent_id}"%', experiment_id)
            else:
                query = """
                    SELECT * FROM lessons
                    WHERE applies_to_agents LIKE ?
                    AND validation_status = 'approved'
                    AND experiment_id IS NULL
                """
                params = (f'%"{agent_id}"%',)

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_lesson(row) for row in rows]

    async def list_all(
        self,
        status: str | None = None,
        lesson_type: str | None = None,
        experiment_id: str | None = None,
        strategy: str | None = None,
    ) -> list[CuratedLesson]:
        """List all lessons with optional filtering.

        Args:
            status: Filter by validation status.
            lesson_type: Filter by lesson type.
            experiment_id: Filter by experiment ID.
            strategy: Filter by strategy.

        Returns:
            List of matching lessons.
        """
        async with self._transaction() as db:
            conditions: list[str] = []
            params: list[Any] = []

            if status:
                conditions.append("validation_status = ?")
                params.append(status)
            if lesson_type:
                conditions.append("lesson_type = ?")
                params.append(lesson_type)
            if experiment_id:
                conditions.append("experiment_id = ?")
                params.append(experiment_id)
            if strategy:
                conditions.append("strategy = ?")
                params.append(strategy)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"SELECT * FROM lessons WHERE {where_clause} ORDER BY created_at DESC"

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_lesson(row) for row in rows]

    async def update_status(
        self,
        lesson_id: str,
        new_status: Literal["approved", "deprecated", "rolled_back"],
    ) -> CuratedLesson | None:
        """Update a lesson's validation status.

        Args:
            lesson_id: The lesson ID to update.
            new_status: The new validation status.

        Returns:
            The updated lesson if found, None otherwise.
        """
        async with self._transaction() as db:
            existing = await self._get_raw(db, lesson_id)
            if not existing:
                return None

            # Save history before update
            await self._save_history(db, existing)

            now = datetime.now(UTC)
            await db.execute(
                """
                UPDATE lessons
                SET validation_status = ?, updated_at = ?, version = version + 1
                WHERE lesson_id = ?
                """,
                (new_status, now.isoformat(), lesson_id),
            )

            return await self._get_raw(db, lesson_id)

    async def delete(self, lesson_id: str) -> bool:
        """Delete a lesson.

        Args:
            lesson_id: The lesson ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        async with self._transaction() as db:
            cursor = await db.execute("DELETE FROM lessons WHERE lesson_id = ?", (lesson_id,))
            return cursor.rowcount > 0

    async def get_history(self, lesson_id: str) -> list[CuratedLesson]:
        """Get version history for a lesson.

        Args:
            lesson_id: The lesson ID to get history for.

        Returns:
            List of historical lesson versions.
        """
        async with self._transaction() as db:
            async with db.execute(
                """
                SELECT snapshot FROM lesson_history
                WHERE lesson_id = ?
                ORDER BY version DESC
                """,
                (lesson_id,),
            ) as cursor:
                rows = await cursor.fetchall()
                lessons = []
                for row in rows:
                    data = json.loads(row["snapshot"])
                    lessons.append(
                        CuratedLesson(
                            lesson_id=data["lesson_id"],
                            source_proposal_id=data["source_proposal_id"],
                            applies_to_agents=data["applies_to_agents"],
                            lesson_type=data["lesson_type"],
                            title=data["title"],
                            description=data["description"],
                            lesson_payload=data["lesson_payload"],
                            validation_status=data["validation_status"],
                            version=data["version"],
                            parent_lesson_id=data.get("parent_lesson_id"),
                            rollback_of=data.get("rollback_of"),
                            evidence_summary=data.get("evidence_summary"),
                            impact_metrics=data.get("impact_metrics", {}),
                            created_at=datetime.fromisoformat(data["created_at"])
                            if data.get("created_at")
                            else datetime.now(UTC),
                            updated_at=datetime.fromisoformat(data["updated_at"])
                            if data.get("updated_at")
                            else None,
                            applied_at=datetime.fromisoformat(data["applied_at"])
                            if data.get("applied_at")
                            else None,
                        )
                    )
                return lessons

    async def rollback(
        self,
        lesson_id: str,
        target_version: int | None = None,
    ) -> CuratedLesson | None:
        """Rollback a lesson to a previous version.

        Args:
            lesson_id: The lesson ID to rollback.
            target_version: Specific version to rollback to. If None, rollback to previous.

        Returns:
            The rolled-back lesson if successful, None otherwise.
        """
        async with self._transaction() as db:
            history = await self.get_history(lesson_id)
            if not history:
                return None

            if target_version is None:
                if len(history) < 1:
                    return None
                target = history[0]  # Most recent history entry
            else:
                matching = [h for h in history if h.version == target_version]
                if not matching:
                    return None
                target = matching[0]

            current = await self._get_raw(db, lesson_id)
            if not current:
                return None

            # Create rollback lesson
            rolled_back = CuratedLesson(
                lesson_id=current.lesson_id,
                source_proposal_id=target.source_proposal_id,
                applies_to_agents=target.applies_to_agents,
                lesson_type=target.lesson_type,
                title=target.title,
                description=target.description,
                lesson_payload=target.lesson_payload,
                validation_status="rolled_back",
                version=current.version + 1,
                parent_lesson_id=target.parent_lesson_id,
                created_at=target.created_at,
                updated_at=datetime.now(UTC),
                rollback_of=lesson_id,
            )

            await self.store(rolled_back)
            return rolled_back

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            self._initialized = False
