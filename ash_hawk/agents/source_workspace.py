from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess, run  # nosec B404
from typing import Iterator

from ash_hawk.types import compute_directory_hashes

_PROJECT_MARKERS = ("pyproject.toml", ".git", "setup.py")
_GIT_ENV_REMOVE = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR")


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in _GIT_ENV_REMOVE:
        env.pop(key, None)
    return env


def _run_git(
    args: list[str], cwd: Path, extra_env: dict[str, str] | None = None
) -> CompletedProcess[str]:
    git_binary = shutil.which("git")
    if git_binary is None:
        raise ValueError("git executable not found")
    env = _git_env()
    if extra_env is not None:
        env.update(extra_env)
    return run(  # nosec B603
        [git_binary, *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _default_commit_identity_env(repo_root: Path) -> dict[str, str]:
    name_result = _run_git(["config", "user.name"], repo_root)
    email_result = _run_git(["config", "user.email"], repo_root)
    if name_result.returncode == 0 and email_result.returncode == 0:
        if name_result.stdout.strip() and email_result.stdout.strip():
            return {}
    return {
        "GIT_AUTHOR_NAME": "Ash Hawk Improve",
        "GIT_AUTHOR_EMAIL": "ash-hawk-improve@local",
        "GIT_COMMITTER_NAME": "Ash Hawk Improve",
        "GIT_COMMITTER_EMAIL": "ash-hawk-improve@local",
    }


def infer_agent_name_from_path(agent_path: Path) -> str:
    resolved = agent_path.resolve()
    if resolved.name in {"agent", "agents"} and resolved.parent.name:
        return resolved.parent.name
    return resolved.stem


def detect_source_root(agent_path: Path) -> Path:
    resolved = agent_path.resolve()
    for candidate in [resolved, *resolved.parents]:
        if any((candidate / marker).exists() for marker in _PROJECT_MARKERS):
            return candidate
    return resolved


def detect_package_name(agent_path: Path) -> str | None:
    resolved = agent_path.resolve()
    package_name: str | None = None
    for candidate in [resolved, *resolved.parents]:
        init_file = candidate / "__init__.py"
        if init_file.is_file():
            package_name = candidate.name
            continue
        if package_name is not None:
            break
    return package_name


def detect_agent_config_path(agent_path: Path) -> Path:
    resolved = agent_path.resolve()
    search_roots = [resolved, *resolved.parents]
    for candidate in search_roots:
        direct_config = candidate / "agent_config.yaml"
        if direct_config.exists():
            return direct_config

        dawn_config = candidate / ".dawn-kestrel" / "agent_config.yaml"
        if dawn_config.exists():
            return dawn_config

    source_root = detect_source_root(resolved)
    return source_root / ".dawn-kestrel" / "agent_config.yaml"


@dataclass
class IsolatedAgentWorkspace:
    repo_root: Path
    source_root: Path
    original_agent_path: Path
    workspace_parent: Path
    workspace_root: Path
    workspace_agent_path: Path

    def seed_from_original(self) -> list[str]:
        return _sync_directory_contents(self.original_agent_path, self.workspace_agent_path)

    def sync_back(self) -> list[str]:
        return _sync_directory_contents(self.workspace_agent_path, self.original_agent_path)

    def cleanup(self) -> None:
        _run_git(["worktree", "remove", "--force", str(self.workspace_root)], self.repo_root)
        shutil.rmtree(self.workspace_parent, ignore_errors=True)


@dataclass
class RepoCommit:
    repo_root: Path
    branch: str | None
    commit_sha: str
    committed_paths: list[str]
    message_title: str
    message_body: str


def _sync_directory_contents(source_dir: Path, destination_dir: Path) -> list[str]:
    source_hashes = compute_directory_hashes(source_dir)
    destination_hashes = compute_directory_hashes(destination_dir)
    changed_paths: set[str] = set()

    for rel_path, digest in source_hashes.items():
        if destination_hashes.get(rel_path) == digest:
            continue
        source = source_dir / rel_path
        destination = destination_dir / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        changed_paths.add(rel_path)

    for rel_path in destination_hashes:
        if rel_path in source_hashes:
            continue
        destination = destination_dir / rel_path
        if destination.exists():
            destination.unlink()
        changed_paths.add(rel_path)

    return sorted(changed_paths)


def detect_git_repo_root(path: Path) -> Path | None:
    result = _run_git(["rev-parse", "--show-toplevel"], path.resolve())
    if result.returncode != 0:
        return None
    stdout = result.stdout.strip()
    return Path(stdout).resolve() if stdout else None


def commit_agent_changes(
    repo_root: Path,
    agent_path: Path,
    changed_paths: list[str],
    *,
    message_title: str,
    message_body: str,
) -> RepoCommit:
    if not changed_paths:
        raise ValueError("Cannot create improve commit without changed paths")

    resolved_repo_root = repo_root.resolve()
    resolved_agent_path = agent_path.resolve()
    relative_agent_path = resolved_agent_path.relative_to(resolved_repo_root)
    repo_paths = [str((relative_agent_path / rel_path).as_posix()) for rel_path in changed_paths]

    staged_before = _run_git(["diff", "--cached", "--name-only"], resolved_repo_root)
    if staged_before.returncode != 0:
        stderr = staged_before.stderr.strip() or staged_before.stdout.strip() or "unknown git error"
        raise ValueError(f"Failed to inspect staged changes before improve commit: {stderr}")

    preexisting_staged = [
        line.strip() for line in staged_before.stdout.splitlines() if line.strip()
    ]
    unexpected_staged = sorted(path for path in preexisting_staged if path not in repo_paths)
    if unexpected_staged:
        raise ValueError(
            "Cannot create improve commit with pre-existing staged changes outside the kept mutation: "
            + ", ".join(unexpected_staged[:5])
        )

    add_result = _run_git(["add", "-A", "--", *repo_paths], resolved_repo_root)
    if add_result.returncode != 0:
        stderr = add_result.stderr.strip() or add_result.stdout.strip() or "unknown git error"
        raise ValueError(f"Failed to stage improve mutation paths: {stderr}")

    diff_result = _run_git(
        ["diff", "--cached", "--name-only", "--", *repo_paths], resolved_repo_root
    )
    if diff_result.returncode != 0:
        stderr = diff_result.stderr.strip() or diff_result.stdout.strip() or "unknown git error"
        _run_git(["reset", "--mixed", "--", *repo_paths], resolved_repo_root)
        raise ValueError(f"Failed to inspect staged improve mutation paths: {stderr}")

    committed_paths = [line.strip() for line in diff_result.stdout.splitlines() if line.strip()]
    if not committed_paths:
        _run_git(["reset", "--mixed", "--", *repo_paths], resolved_repo_root)
        raise ValueError("No staged improve mutation paths were available to commit")

    identity_env = _default_commit_identity_env(resolved_repo_root)
    commit_result = _run_git(
        ["commit", "-m", message_title, "-m", message_body],
        resolved_repo_root,
        extra_env=identity_env,
    )
    if commit_result.returncode != 0:
        _run_git(["reset", "--mixed", "--", *repo_paths], resolved_repo_root)
        stderr = commit_result.stderr.strip() or commit_result.stdout.strip() or "unknown git error"
        raise ValueError(f"Failed to create improve commit: {stderr}")

    sha_result = _run_git(["rev-parse", "HEAD"], resolved_repo_root)
    if sha_result.returncode != 0:
        stderr = sha_result.stderr.strip() or sha_result.stdout.strip() or "unknown git error"
        raise ValueError(f"Failed to resolve improve commit sha: {stderr}")

    branch_result = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], resolved_repo_root)
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None

    return RepoCommit(
        repo_root=resolved_repo_root,
        branch=branch or None,
        commit_sha=sha_result.stdout.strip(),
        committed_paths=committed_paths,
        message_title=message_title,
        message_body=message_body,
    )


def prepare_isolated_agent_workspace(
    agent_path: Path,
    run_id: str,
    workspace_id: str,
) -> IsolatedAgentWorkspace:
    original_agent_path = agent_path.resolve()
    source_root = detect_source_root(original_agent_path)
    repo_root = detect_git_repo_root(source_root)
    if repo_root is None:
        raise ValueError(f"Agent path {original_agent_path} is not inside a git repository")

    workspace_parent = Path(tempfile.mkdtemp(prefix=f"ash-hawk-improve-{run_id}-{workspace_id}-"))
    workspace_root = workspace_parent / "worktree"
    result = _run_git(["worktree", "add", "--detach", str(workspace_root), "HEAD"], repo_root)
    if result.returncode != 0:
        shutil.rmtree(workspace_parent, ignore_errors=True)
        stderr = result.stderr.strip() or result.stdout.strip() or "unknown git worktree error"
        raise ValueError(f"Failed to create git worktree for {original_agent_path}: {stderr}")

    relative_agent_path = original_agent_path.relative_to(repo_root)
    workspace_agent_path = workspace_root / relative_agent_path

    workspace = IsolatedAgentWorkspace(
        repo_root=repo_root,
        source_root=source_root,
        original_agent_path=original_agent_path,
        workspace_parent=workspace_parent,
        workspace_root=workspace_root,
        workspace_agent_path=workspace_agent_path,
    )
    workspace.seed_from_original()
    return workspace


@contextmanager
def import_package_from_agent_path(package_name: str, agent_path: Path | None) -> Iterator[None]:
    if agent_path is None:
        yield
        return

    resolved_agent_path = agent_path.resolve()
    import_root: Path | None = None
    for candidate in [resolved_agent_path, *resolved_agent_path.parents]:
        if candidate.name == package_name:
            import_root = candidate.parent
            break

    if import_root is None:
        import_root = detect_source_root(resolved_agent_path)

    import_root_str = str(import_root)
    original_sys_path = list(sys.path)
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == package_name or name.startswith(f"{package_name}.")
    }

    for name in saved_modules:
        sys.modules.pop(name, None)

    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)
    importlib.invalidate_caches()

    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)
        sys.path[:] = original_sys_path
        sys.modules.update(saved_modules)
        importlib.invalidate_caches()


__all__ = [
    "IsolatedAgentWorkspace",
    "RepoCommit",
    "commit_agent_changes",
    "detect_package_name",
    "detect_git_repo_root",
    "detect_agent_config_path",
    "detect_source_root",
    "import_package_from_agent_path",
    "infer_agent_name_from_path",
    "prepare_isolated_agent_workspace",
]
