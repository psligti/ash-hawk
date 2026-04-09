from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path

from ash_hawk.types import compute_directory_hashes

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".yaml", ".yml", ".json", ".py", ".txt", ".toml"}
)

_MAX_FILE_SIZE: int = 1_000_000  # 1 MB


class AgentMutator:
    """Read and write agent directories during the improvement loop.

    Provides snapshot, scan, write (with backup and atomic write),
    revert, diff, and cleanup operations. All writes are validated
    for path traversal, extension whitelist, and size cap.

    Args:
        agent_path: Path to the agent directory.
        run_id: Unique identifier for the current improvement run.
    """

    def __init__(self, agent_path: Path, run_id: str) -> None:
        self.agent_path: Path = agent_path.resolve()
        self.run_id: str = run_id
        self.backup_dir: Path = Path(".ash-hawk") / "mutations" / run_id / "backups"
        self._snapshot_manifest: dict[str, str] | None = None

    def snapshot(self) -> str:
        """Snapshot the current state of the agent directory.

        Uses ``compute_directory_hashes`` to capture a manifest of
        ``{relative_path: sha256_hex}`` entries.

        Returns:
            A concatenated SHA-256 digest of all sorted manifest entries,
            suitable for quick equality comparisons.
        """
        manifest = compute_directory_hashes(self.agent_path)
        self._snapshot_manifest = manifest

        sorted_items = sorted(manifest.items())
        concatenated = "".join(f"{path}:{h}" for path, h in sorted_items)
        return hashlib.sha256(concatenated.encode()).hexdigest()

    def scan(self) -> dict[str, str]:
        """Scan the agent directory and return file contents.

        Walks ``agent_path``, reading all text files while skipping
        hidden directories (starting with ``.``) and binary files.

        Returns:
            A dict mapping relative POSIX paths to file contents.
        """
        contents: dict[str, str] = {}
        if not self.agent_path.is_dir():
            return contents

        for child in sorted(self.agent_path.rglob("*")):
            if not child.is_file():
                continue

            rel = child.relative_to(self.agent_path)
            if any(part.startswith(".") for part in rel.parts):
                continue

            try:
                if child.stat().st_size > _MAX_FILE_SIZE:
                    continue
            except OSError:
                continue

            try:
                text = child.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            except OSError:
                continue

            contents[rel.as_posix()] = text

        return contents

    def read_file(self, relative_path: str) -> str | None:
        """Read a single file from the agent directory.

        Args:
            relative_path: POSIX-style path relative to ``agent_path``.

        Returns:
            File contents as a string, or ``None`` if the file does not
            exist or cannot be read as UTF-8 text.
        """
        target = (self.agent_path / relative_path).resolve()
        if not target.is_relative_to(self.agent_path):
            return None
        if not target.is_file():
            return None
        try:
            return target.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return None

    def write_file(self, relative_path: str, content: str) -> None:
        """Write content to a file in the agent directory.

        Performs full security validation, backs up the existing file
        (once per snapshot), and uses an atomic write strategy.

        Args:
            relative_path: POSIX-style path relative to ``agent_path``.
            content: Text content to write.

        Raises:
            ValueError: If path traversal is detected, the extension is
                not whitelisted, or the content exceeds the size cap.
        """
        target = (self.agent_path / relative_path).resolve()

        # --- Security validation ---
        if not target.is_relative_to(self.agent_path):
            raise ValueError(
                f"Path traversal detected: {relative_path!r} resolves outside agent directory"
            )

        if target.suffix.lower() not in _ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Disallowed file extension: {target.suffix!r} "
                f"(allowed: {sorted(_ALLOWED_EXTENSIONS)})"
            )

        if len(content) > _MAX_FILE_SIZE:
            raise ValueError(f"Content size {len(content)} exceeds maximum {_MAX_FILE_SIZE} bytes")

        # --- Backup (once per file per snapshot) ---
        if target.exists():
            backup_path = self.backup_dir / relative_path
            if not backup_path.exists():
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup_path)

        # --- Atomic write ---
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp_path.write_text(content, encoding="utf-8")
            os.rename(tmp_path, target)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def backup_all(self) -> None:
        """Back up every file in the agent directory for full revert.

        Copies all files (respecting the same filters as :meth:`scan`)
        into the backup directory so that :meth:`revert_all` can restore
        them after an external process (e.g. a coding agent) modifies
        files directly on disk.
        """
        if not self.agent_path.is_dir():
            return
        for child in sorted(self.agent_path.rglob("*")):
            if not child.is_file():
                continue
            rel = child.relative_to(self.agent_path)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if child.suffix.lower() not in _ALLOWED_EXTENSIONS:
                continue
            try:
                if child.stat().st_size > _MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            backup_path = self.backup_dir / rel
            if backup_path.exists():
                continue
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, backup_path)

    def revert_all(self) -> None:
        """Revert all mutated files to their pre-snapshot state.

        Restores every backed-up file to its original location within
        ``agent_path``.  If no backups exist this is a no-op.
        """
        if not self.backup_dir.is_dir():
            return

        for backed_up in sorted(self.backup_dir.rglob("*")):
            if not backed_up.is_file():
                continue
            rel = backed_up.relative_to(self.backup_dir)
            original = self.agent_path / rel
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backed_up, original)

        self._snapshot_manifest = None

    def diff_since_snapshot(self, snapshot_hash: str) -> dict[str, str]:
        """Compute changes since the snapshot was taken.

        Compares the current directory state against the stored snapshot
        manifest.

        Args:
            snapshot_hash: The snapshot hash returned by :meth:`snapshot`.
                Used to confirm the caller is referencing the correct snapshot.

        Returns:
            A dict mapping relative paths to change status strings:
            ``"added"``, ``"modified"``, or ``"removed"``.

        Raises:
            RuntimeError: If no snapshot has been taken.
        """
        if self._snapshot_manifest is None:
            raise RuntimeError("No snapshot has been taken; call snapshot() first")

        current = compute_directory_hashes(self.agent_path)
        before = self._snapshot_manifest

        changes: dict[str, str] = {}

        for path, _hash in current.items():
            if path not in before:
                changes[path] = "added"
            elif before[path] != _hash:
                changes[path] = "modified"

        for path in before:
            if path not in current:
                changes[path] = "removed"

        return changes

    def cleanup(self) -> None:
        """Remove the backup directory tree.

        Safe to call when no backup directory exists — silently does nothing.
        """
        if self.backup_dir.is_dir():
            shutil.rmtree(self.backup_dir)


__all__ = ["AgentMutator"]
