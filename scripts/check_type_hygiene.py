"""Pre-commit hook: enforce type hygiene.

Rules:
  FORBID (exit 1):  typing.Any, typing.Optional
  WARN  (exit 0):   bare dict / typing.Dict without type parameters

Usage:
  pre-commit (automatic)  or  python scripts/check_type_hygiene.py [files...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Patterns — compiled once at import.
# ---------------------------------------------------------------------------

# Matches:  Any, Any], Any), Any,, Any=   but NOT  AnyOf, Anything, etc.
# Also catches:  from typing import ... Any ...  /  import Any
_RE_ANY = re.compile(
    r"""
    (?:^from\s+typing\s+import\s[^;]*\bAny\b   # `from typing import Any`
    |^import\s+typing\b.*\bAny\b                # `import typing ... Any`
    |\bAny\b(?=[\]\s,\)=])                       # standalone `Any`
    )
    """,
    re.VERBOSE,
)

# Matches:  Optional, Optional[  etc.
_RE_OPTIONAL = re.compile(
    r"""
    (?:^from\s+typing\s+import\s[^;]*\bOptional\b
    |^import\s+typing\b.*\bOptional\b
    |\bOptional\b(?=[\[\s,\)=])
    )
    """,
    re.VERBOSE,
)

# Matches bare `dict` used as a type annotation (no [<params>]).
# Positive:   dict, dict], dict), dict,   at end of annotation zone
# Negative:   dict[str, int], OrderedDict, defaultdict
_RE_BARE_DICT = re.compile(
    r"""
    \bdict\b          # the word `dict`
    (?!               # NOT followed by `[` (parameterised)
        \[
    )
    (?=[\]\s,\)=:])   # followed by annotation boundary chars
    """,
    re.VERBOSE,
)

# Same for typing.Dict
_RE_DICT_CLASS = re.compile(
    r"""
    (?:^from\s+typing\s+import\s[^;]*\bDict\b
    |\bDict\b(?=[\[\s,\)=])
    )
    """,
    re.VERBOSE,
)

# Strip quoted strings to avoid false positives in descriptions/comments
_RE_QUOTED_STRING = re.compile(r"""(?:"[^"]*"|'[^']*')""")
_RE_SKIP_LINE = re.compile(r"^\s*(#|'''|\"\"\")")

# File-level skip marker
_MARKER_SKIP = "# type-hygiene: skip-file"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _check_file(path: Path) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return (errors, warnings) as lists of (line_no, message)."""
    errors: list[tuple[int, str]] = []
    warnings: list[tuple[int, str]] = []

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return errors, warnings

    if _MARKER_SKIP in text:
        return errors, warnings

    lines = text.splitlines()
    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if _RE_SKIP_LINE.match(line):
            continue

        check_line = _RE_QUOTED_STRING.sub("", raw_line)

        # Skip inline suppressions
        if "# type: ignore" in line:
            continue

        # --- FORBID: Any ---
        if _RE_ANY.search(check_line):
            errors.append((idx, f"Forbidden `Any`: {raw_line.strip()}"))

        if _RE_OPTIONAL.search(check_line):
            errors.append(
                (idx, f"Forbidden `Optional` — use `X | None` instead: {raw_line.strip()}")
            )

        if _RE_BARE_DICT.search(check_line):
            if "dict(" not in check_line:
                warnings.append((idx, f"Avoid bare `dict` — use `dict[K, V]`: {raw_line.strip()}"))

        if _RE_DICT_CLASS.search(check_line):
            warnings.append((idx, f"Avoid `typing.Dict` — use `dict[K, V]`: {raw_line.strip()}"))

    return errors, warnings


def main() -> int:
    """Entry point. Returns 1 if any errors found, 0 otherwise."""
    raw_args = [Path(f) for f in sys.argv[1:]]

    if not raw_args:
        files = sorted(Path("ash_hawk").rglob("*.py"))
    else:
        files: list[Path] = []
        for arg in raw_args:
            if arg.is_dir():
                files.extend(sorted(arg.rglob("*.py")))
            else:
                files.append(arg)

    total_errors = 0
    total_warnings = 0

    for path in files:
        path = path.resolve()
        if not path.is_file() or path.suffix != ".py":
            continue

        errors, warnings = _check_file(path)

        for line_no, msg in errors:
            print(f"ERROR {path}:{line_no}: {msg}")
            total_errors += 1

        for line_no, msg in warnings:
            print(f"WARN  {path}:{line_no}: {msg}")
            total_warnings += 1

    if total_warnings > 0:
        print(f"\n  {total_warnings} type-hygiene warning(s) — review recommended")

    if total_errors > 0:
        print(f"\n  {total_errors} type-hygiene error(s) — COMMIT BLOCKED")
        print("  Fix: replace `Any` with a concrete type, `Optional[X]` with `X | None`")
        print("  Suppress a line with:  # type: ignore")
        print("  Suppress a file with:  # type-hygiene: skip-file")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
