from __future__ import annotations

import os
from pathlib import Path
from subprocess import TimeoutExpired  # nosec B404
from subprocess import run as run_subprocess  # nosec B404

IGNORED_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        ".env",
        ".idea",
        ".vscode",
        "dist",
        "build",
        "egg-info",
        ".eggs",
        ".next",
        ".nuxt",
        ".cache",
        ".terraform",
        ".DS_Store",
        ".ash-hawk",
        ".bolt-merlin",
    }
)

FORBIDDEN_PATH_PATTERNS: frozenset[str] = frozenset(
    {
        ".ash-hawk",
        ".dawn-kestrel",
        ".scenario.yaml",
    }
)

DANGEROUS_GLOB_PATTERNS = frozenset({"**/*", "**", "**/", "*", "*/**", "**/**"})

MAX_RESULTS_DEFAULT = 1_000
MAX_DEPTH_DEFAULT = 10
MAX_GREP_RESULTS = 200
MAX_GREP_FILE_SIZE = 1_000_000


def normalize_workspace_relative_path(file_path: Path, base: Path) -> Path:
    if file_path.is_absolute():
        return file_path
    parts = file_path.parts
    if parts and parts[0] == base.name:
        trimmed = parts[1:]
        if trimmed:
            return base / Path(*trimmed)
        return base
    return base / file_path


def resolve_path(raw_path: str | Path, base: Path) -> Path:
    path = raw_path if isinstance(raw_path, Path) else Path(raw_path)
    return normalize_workspace_relative_path(path, base)


def ensure_workspace_contained(path: Path, workspace_root: Path) -> Path:
    resolved_root = workspace_root.resolve()
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Path '{path}' resolves outside the workspace root '{resolved_root}'."
        ) from exc
    return resolved_path


def resolve_path_within_workspace(raw_path: str | Path, base: Path) -> Path:
    return ensure_workspace_contained(resolve_path(raw_path, base), base)


def check_workspace_path_error(
    path: Path, workspace_root: Path, tool_name: str
) -> tuple[Path | None, str | None]:
    try:
        resolved = ensure_workspace_contained(path, workspace_root)
    except ValueError:
        return None, (
            f"❌ Access denied: path '{path}' resolves outside the workspace root '{workspace_root.resolve()}'. "
            f"The {tool_name} tool may only access files inside the active workspace."
        )
    return resolved, None


def display_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def is_dangerous_glob(pattern: str) -> bool:
    return pattern.strip("/") in DANGEROUS_GLOB_PATTERNS


def is_forbidden_glob_pattern(pattern: str) -> str | None:
    for forbidden in FORBIDDEN_PATH_PATTERNS:
        stripped = forbidden.lstrip(".")
        if stripped in pattern or forbidden in pattern:
            return str(forbidden)
    return None


def is_forbidden_path(path: Path) -> str | None:
    path_str = str(path)
    try:
        resolved = path.resolve()
        resolved_str = str(resolved)
    except (OSError, ValueError):
        resolved_str = path_str
    for pattern in FORBIDDEN_PATH_PATTERNS:
        segment = "/" + pattern
        if (
            segment in path_str
            or segment in resolved_str
            or path_str.endswith(pattern)
            or resolved_str.endswith(pattern)
        ):
            return str(pattern)
        if pattern in Path(path_str).parts or pattern in Path(resolved_str).parts:
            return str(pattern)
    return None


def check_forbidden_path_error(path: Path, tool_name: str) -> str | None:
    forbidden = is_forbidden_path(path)
    if forbidden is None:
        return None
    return (
        f"❌ Access denied: path '{path}' matches forbidden pattern '{forbidden}'. "
        f"The {tool_name} tool may not access evaluation infrastructure or session metadata. "
        f"Focus only on source files in the project."
    )


def is_forbidden_command(command: str) -> str | None:
    for forbidden in FORBIDDEN_PATH_PATTERNS:
        if forbidden in command:
            return str(forbidden)
    return None


def safe_walk(
    base: Path, max_depth: int = MAX_DEPTH_DEFAULT, max_results: int = MAX_RESULTS_DEFAULT
) -> list[Path]:
    seen_inodes: set[tuple[int, int]] = set()
    results: list[Path] = []

    def inode_key(path: Path) -> tuple[int, int]:
        try:
            stat = path.stat()
            return (stat.st_dev, stat.st_ino)
        except (OSError, ValueError):
            return (0, 0)

    def walk(current: Path, depth: int) -> None:
        if depth > max_depth or len(results) >= max_results:
            return
        try:
            entries = list(os.scandir(current))
        except (PermissionError, OSError):
            return
        for entry in entries:
            if len(results) >= max_results:
                return
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name in IGNORED_DIR_NAMES:
                        continue
                    path = Path(entry.path)
                    key = inode_key(path)
                    if key in seen_inodes:
                        continue
                    seen_inodes.add(key)
                    walk(path, depth + 1)
                elif entry.is_file(follow_symlinks=False):
                    results.append(Path(entry.path))
            except (OSError, ValueError):
                continue

    seen_inodes.add(inode_key(base))
    walk(base, 0)
    return results


def matches_include(path: Path, include: str) -> bool:
    if "{" in include and "}" in include:
        suffixes = include[include.index("{") + 1 : include.index("}")].split(",")
        return any(path.name.endswith(s.strip().lstrip("*")) for s in suffixes)
    return path.match(include)


def format_read_output(
    file_path: str | Path, content: str, offset: int = 1, total_lines: int | None = None
) -> str:
    lines = content.splitlines()
    shown = len(lines)
    header = f"📄 **{Path(file_path).name}**"
    if total_lines is not None and shown < total_lines:
        header += f"  (lines {offset}–{offset + shown - 1} of {total_lines})"
    elif offset > 1:
        header += f"  (from line {offset})"
    return f"{header}\n\n```{Path(file_path).suffix.lstrip('.') or 'text'}\n{content}\n```"


def format_write_output(file_path: str | Path, size: int) -> str:
    return (
        f"✅ Wrote **{Path(file_path).name}** ({human_bytes(size)}) → `{Path(file_path).parent}/`"
    )


def format_edit_output(
    file_path: str | Path, old_lines: list[str], new_lines: list[str], line_num: int | None = None
) -> str:
    location = f" (line {line_num})" if line_num else ""
    return f"✏️ Edited **{Path(file_path).name}**{location}\n```diff\n{mini_diff(old_lines, new_lines)}\n```"


def format_glob_output(pattern: str, base: str | Path, matches: list[str]) -> str:
    count = len(matches)
    if count == 0:
        return f"📭 No files matching `{pattern}` in `{base}`"
    header = f"📂 **{count} file{'s' if count != 1 else ''}** matching `{pattern}`"
    body = "\n".join(f"  `{item}`" for item in matches[:50])
    if count > 50:
        body += f"\n  … and {count - 50} more"
    return f"{header}\n{body}"


def format_grep_output(
    pattern: str, base: str | Path, results: list[str], *, truncated: bool = False
) -> str:
    count = len(results)
    if count == 0:
        return f"📭 No matches for `{pattern}` in `{base}`"
    body = "\n".join(results)
    if truncated:
        body += f"\n… (truncated at {count} matches)"
    return f"🔍 **{count} match{'es' if count != 1 else ''}** for `{pattern}`\n\n```\n{body}\n```"


def format_bash_output(command: str, output: str, exit_code: int = 0) -> str:
    icon = "✅" if exit_code == 0 else "❌"
    short_cmd = command if len(command) <= 60 else command[:57] + "…"
    body = output.rstrip()
    if not body:
        return f"{icon} `$ {short_cmd}` — no output"
    if len(body) > 5000:
        body = body[:5000] + "\n… (output truncated)"
    return f"{icon} `$ {short_cmd}`\n```\n{body}\n```"


def format_bash_error(command: str, output: str, exit_code: int) -> str:
    short_cmd = command if len(command) <= 60 else command[:57] + "…"
    body = output.rstrip()
    if body:
        if len(body) > 5000:
            body = body[:5000] + "\n… (output truncated)"
        return f"❌ `$ {short_cmd}` — exit {exit_code}\n```\n{body}\n```"
    return f"❌ `$ {short_cmd}` — exit {exit_code}"


def format_test_output(output: str, *, passed: bool) -> str:
    icon = "✅" if passed else "❌"
    status = "passed" if passed else "failed"
    body = output.rstrip()
    if len(body) > 5000:
        body = body[:5000] + "\n… (output truncated)"
    return f"{icon} **Tests {status}**\n```\n{body}\n```"


STATUS_ICONS = {"completed": "✅", "in_progress": "🔄", "pending": "⏳", "cancelled": "❌"}


def render_todo_line(line: str) -> str:
    stripped = line.strip()
    if not stripped.startswith("- ["):
        return line
    rest = stripped[6:]
    if rest.startswith("["):
        bracket_end = rest.index("]")
        priority = rest[1:bracket_end]
        rest = rest[bracket_end + 2 :]
    else:
        priority = ""
    if rest.endswith(")"):
        paren_start = rest.rfind("(")
        if paren_start > 0:
            status = rest[paren_start + 1 : -1]
            content = rest[:paren_start].strip()
            icon = STATUS_ICONS.get(status, "·")
            return f"  {icon} **[{priority}]** {content}"
    return f"  · {rest}"


def todo_summary(in_progress: int, pending: int, done: int, cancelled: int) -> str:
    parts: list[str] = []
    if in_progress:
        parts.append(f"{in_progress} in progress")
    if pending:
        parts.append(f"{pending} pending")
    if done:
        parts.append(f"{done} completed")
    if cancelled:
        parts.append(f"{cancelled} cancelled")
    return " · ".join(parts)


def format_todoread_output(content: str, file_path: str | Path) -> str:
    lines = [line for line in content.splitlines() if line.strip()]
    if not lines or (len(lines) == 1 and lines[0].startswith("# ")):
        return "📋 Todo list is empty"
    total = sum(1 for line in lines if line.strip().startswith("- ["))
    done = sum(1 for line in lines if "(completed)" in line)
    in_progress = sum(1 for line in lines if "(in_progress)" in line)
    pending = sum(1 for line in lines if "(pending)" in line)
    cancelled = sum(1 for line in lines if "(cancelled)" in line)
    header = f"📋 **Todo List** ({done}/{total} done) — `{Path(file_path).name}`"
    body = "\n".join(render_todo_line(line) for line in lines if line.strip().startswith("- ["))
    return f"{header}\n\n{body}\n\n{todo_summary(in_progress, pending, done, cancelled)}"


def format_todowrite_output(todos: list[dict[str, object]], file_path: str | Path) -> str:
    count = len(todos)
    if count == 0:
        return f"📋 **Cleared todo list** — `{Path(file_path).name}`"
    done = sum(1 for todo in todos if todo.get("status") == "completed")
    in_progress = sum(1 for todo in todos if todo.get("status") == "in_progress")
    pending = sum(1 for todo in todos if todo.get("status") == "pending")
    cancelled = sum(1 for todo in todos if todo.get("status") == "cancelled")
    header = f"📋 **Updated Todo List** ({done}/{count} done) — `{Path(file_path).name}`"
    rendered = [
        f"  {STATUS_ICONS.get(str(todo.get('status', 'pending')), '·')} **[{todo.get('priority', 'medium')}]** {todo.get('content', '')}"
        for todo in todos
    ]
    body = "\n".join(rendered)
    return f"{header}\n\n{body}\n\n{todo_summary(in_progress, pending, done, cancelled)}"


def format_error(tool_name: str, message: str) -> str:
    return f"❌ **{tool_name}**: {message}"


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def mini_diff(old_lines: list[str], new_lines: list[str], max_context: int = 6) -> str:
    parts: list[str] = []
    for line in old_lines[:max_context]:
        parts.append(f"- {line}")
    for line in new_lines[:max_context]:
        parts.append(f"+ {line}")
    if len(old_lines) > max_context or len(new_lines) > max_context:
        parts.append("…")
    return "\n".join(parts)


def verify_edit_applied(file_path: Path, new_string: str) -> bool:
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError:
        return False
    if not new_string.strip():
        return True
    if new_string in content:
        return True
    for line in new_string.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) > 5 and stripped in content:
            return True
    return False


def format_verification_block(file_path: Path, new_string: str, verified: bool) -> str:
    del file_path
    status = "✅ VERIFIED" if verified else "❌ NOT VERIFIED"
    marker_line = ""
    for line in new_string.splitlines():
        stripped = line.strip()
        if stripped:
            marker_line = stripped
            break
    if not marker_line:
        marker_line = "<empty replacement>"
    if len(marker_line) > 120:
        marker_line = marker_line[:117] + "..."
    block = (
        f"\n---\n**POST-EDIT VERIFICATION**: {status}\nExpected change marker: `{marker_line}`\n"
    )
    if not verified:
        block += (
            "The edit was applied but the expected content was NOT confirmed in the file. "
            "You MUST re-read the file and verify that the marker above appears in the output. "
            "If it does not, the edit FAILED — retry or use `write`.\n"
        )
    return block


def suggest_glob_for_missing_file(file_path: Path, base: Path) -> str:
    if file_path.parent != base:
        return ""
    return (
        f" Hint: '{file_path.name}' was not found in the workspace root. "
        f"Use `glob('**/{file_path.name}')` to locate the actual file path."
    )


def run_shell_command(command: str, *, cwd: Path, timeout: int) -> tuple[int, str, str | None]:
    try:
        completed = run_subprocess(  # nosec
            ["zsh", "-lc", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return completed.returncode, output, None
    except TimeoutExpired:
        return -1, "", f"timeout after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return -1, "", str(exc)


def detect_test_command(cwd: Path) -> str:
    if (cwd / "pyproject.toml").exists():
        return "uv run pytest"
    if (cwd / "pytest.ini").exists() or (cwd / "setup.cfg").exists():
        return "pytest"
    if (cwd / "package.json").exists():
        return "npm test"
    return "pytest"


def todo_file_path(base: Path, session_id: str, file_path: str | None = None) -> Path:
    if file_path:
        return resolve_path(file_path, base)
    return base / ".ash-hawk" / "thin_runtime" / "todos" / f"{session_id}.md"


def write_todo_markdown(todos: list[dict[str, object]]) -> str:
    lines = ["# Todos", ""]
    for todo in todos:
        status = str(todo.get("status", "pending"))
        priority = str(todo.get("priority", "medium"))
        content = str(todo.get("content", ""))
        check = "x" if status == "completed" else " "
        lines.append(f"- [{check}] [{priority}] {content} ({status})")
    lines.append("")
    return "\n".join(lines)
