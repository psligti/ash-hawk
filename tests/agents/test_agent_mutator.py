"""Tests for AgentMutator."""

from __future__ import annotations

from pathlib import Path

import pytest

from ash_hawk.agents.agent_mutator import AgentMutator


@pytest.fixture
def agent_dir(tmp_path: Path) -> Path:
    """Create a minimal agent directory with AGENT.md and skills/."""
    d = tmp_path / "agent"
    d.mkdir()
    (d / "AGENT.md").write_text("# Test Agent\n")
    skills = d / "skills"
    skills.mkdir()
    (skills / "review.md").write_text("Review skill\n")
    return d


@pytest.fixture
def mutator(agent_dir: Path) -> AgentMutator:
    """Create an AgentMutator pointed at the fixture agent dir."""
    return AgentMutator(agent_path=agent_dir, run_id="test-run-001")


class TestAgentMutatorSnapshot:
    """Test snapshot operations."""

    def test_snapshot_returns_hash(self, mutator: AgentMutator) -> None:
        result = mutator.snapshot()
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_snapshot_consistent(self, mutator: AgentMutator) -> None:
        first = mutator.snapshot()
        second = mutator.snapshot()
        assert first == second

    def test_snapshot_detects_changes(self, mutator: AgentMutator, agent_dir: Path) -> None:
        original = mutator.snapshot()
        (agent_dir / "AGENT.md").write_text("# Modified Agent\n")
        changed = mutator.snapshot()
        assert changed != original


class TestAgentMutatorScan:
    """Test scan operations."""

    def test_scan_reads_text_files(self, mutator: AgentMutator, agent_dir: Path) -> None:
        (agent_dir / "skills" / "test.md").write_text("Test skill\n")
        contents = mutator.scan()
        assert "AGENT.md" in contents
        assert "skills/review.md" in contents
        assert "skills/test.md" in contents
        assert contents["AGENT.md"] == "# Test Agent\n"

    def test_scan_skips_hidden_dirs(self, mutator: AgentMutator, agent_dir: Path) -> None:
        hidden = agent_dir / ".hidden"
        hidden.mkdir()
        (hidden / "secret.md").write_text("secret\n")
        contents = mutator.scan()
        assert all(".hidden" not in k for k in contents)

    def test_scan_skips_binary_files(self, mutator: AgentMutator, agent_dir: Path) -> None:
        (agent_dir / "image.bin").write_bytes(b"\x00\x01\x02\xff\xfe\xfd")
        contents = mutator.scan()
        assert "image.bin" not in contents

    def test_scan_skips_oversized_files(self, mutator: AgentMutator, agent_dir: Path) -> None:
        big = agent_dir / "huge.md"
        big.write_text("x" * 1_000_001)
        contents = mutator.scan()
        assert "huge.md" not in contents

    def test_scan_empty_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty-agent"
        empty_dir.mkdir()
        m = AgentMutator(agent_path=empty_dir, run_id="empty")
        assert m.scan() == {}


class TestAgentMutatorWriteFile:
    """Test write_file operations."""

    def test_write_file_creates_new(self, mutator: AgentMutator, agent_dir: Path) -> None:
        mutator.write_file("tools/helper.md", "# Helper\n")
        written = agent_dir / "tools" / "helper.md"
        assert written.is_file()
        assert written.read_text() == "# Helper\n"

    def test_write_file_updates_existing(self, mutator: AgentMutator, agent_dir: Path) -> None:
        mutator.write_file("AGENT.md", "# Updated\n")
        assert (agent_dir / "AGENT.md").read_text() == "# Updated\n"

    def test_write_file_creates_backup(self, mutator: AgentMutator, agent_dir: Path) -> None:
        original = (agent_dir / "AGENT.md").read_text()
        mutator.write_file("AGENT.md", "# Changed\n")
        backup = mutator.backup_dir / "AGENT.md"
        assert backup.is_file()
        assert backup.read_text() == original

    def test_write_file_backup_once_per_file(self, mutator: AgentMutator, agent_dir: Path) -> None:
        mutator.write_file("AGENT.md", "# First change\n")
        backup = mutator.backup_dir / "AGENT.md"
        first_backup_text = backup.read_text()

        mutator.write_file("AGENT.md", "# Second change\n")
        assert backup.read_text() == first_backup_text

    def test_write_file_path_traversal(self, mutator: AgentMutator) -> None:
        with pytest.raises(ValueError, match="Path traversal"):
            mutator.write_file("../etc/passwd", "malicious")

    def test_write_file_disallowed_extension(self, mutator: AgentMutator) -> None:
        with pytest.raises(ValueError, match="Disallowed file extension"):
            mutator.write_file("malware.exe", "payload")

    def test_write_file_size_cap(self, mutator: AgentMutator) -> None:
        big_content = "x" * 1_000_001
        with pytest.raises(ValueError, match="exceeds maximum"):
            mutator.write_file("big.md", big_content)

    def test_write_file_atomic(self, mutator: AgentMutator, agent_dir: Path) -> None:
        mutator.write_file("skills/new.md", "new content\n")
        tmp_files = list(agent_dir.rglob("*.tmp"))
        assert len(tmp_files) == 0


class TestAgentMutatorRevert:
    """Test revert and cleanup operations."""

    def test_revert_restores_files(self, mutator: AgentMutator, agent_dir: Path) -> None:
        original_text = (agent_dir / "AGENT.md").read_text()
        mutator.write_file("AGENT.md", "# Overwritten\n")
        assert (agent_dir / "AGENT.md").read_text() == "# Overwritten\n"

        mutator.revert_all()
        assert (agent_dir / "AGENT.md").read_text() == original_text

    def test_revert_no_mutations(self, mutator: AgentMutator) -> None:
        mutator.revert_all()

    def test_cleanup_removes_backup_dir(self, mutator: AgentMutator, agent_dir: Path) -> None:
        mutator.write_file("AGENT.md", "# Changed\n")
        assert mutator.backup_dir.is_dir()

        mutator.cleanup()
        assert not mutator.backup_dir.exists()

    def test_cleanup_no_backup_dir(self, mutator: AgentMutator) -> None:
        mutator.cleanup()


class TestAgentMutatorDiff:
    """Test diff_since_snapshot operations."""

    def test_diff_no_changes(self, mutator: AgentMutator) -> None:
        snap = mutator.snapshot()
        assert mutator.diff_since_snapshot(snap) == {}

    def test_diff_modified_file(self, mutator: AgentMutator, agent_dir: Path) -> None:
        snap = mutator.snapshot()
        (agent_dir / "AGENT.md").write_text("# Modified\n")
        diff = mutator.diff_since_snapshot(snap)
        assert "AGENT.md" in diff
        assert diff["AGENT.md"] == "modified"

    def test_diff_added_file(self, mutator: AgentMutator, agent_dir: Path) -> None:
        snap = mutator.snapshot()
        (agent_dir / "new-file.md").write_text("new\n")
        diff = mutator.diff_since_snapshot(snap)
        assert "new-file.md" in diff
        assert diff["new-file.md"] == "added"

    def test_diff_removed_file(self, mutator: AgentMutator, agent_dir: Path) -> None:
        snap = mutator.snapshot()
        (agent_dir / "skills" / "review.md").unlink()
        diff = mutator.diff_since_snapshot(snap)
        assert "skills/review.md" in diff
        assert diff["skills/review.md"] == "removed"

    def test_diff_without_snapshot(self, mutator: AgentMutator) -> None:
        with pytest.raises(RuntimeError, match="No snapshot"):
            mutator.diff_since_snapshot("fake-hash")
