# type-hygiene: skip-file
"""Ash-Hawk logging context using dawn-kestrel patterns.

This module provides evaluation context management for logging and
secret redaction utilities. Uses dawn-kestrel logging helpers when
available, falls back to standard Python logging otherwise.
"""

from __future__ import annotations

import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

# Try to import dawn-kestrel logging helpers, fall back to standard logging
try:
    from dawn_kestrel.agents.review.utils.redaction import redact_dict as dk_redact_dict

    HAS_DK_REDACTION = True
except ImportError:
    HAS_DK_REDACTION = False

logger = logging.getLogger(__name__)


@dataclass
class EvalContext:
    """Evaluation context for tracking runs, suites, and trials."""

    run_id: str | None = None
    suite_id: str | None = None
    trial_id: str | None = None


# Context variable for evaluation context
_eval_context: ContextVar[EvalContext] = ContextVar("eval_context", default=EvalContext())


def set_eval_context(
    run_id: str | None = None, suite_id: str | None = None, trial_id: str | None = None
) -> None:
    """Set the current evaluation context for logging.

    Args:
        run_id: Unique identifier for the evaluation run
        suite_id: Unique identifier for the test suite
        trial_id: Unique identifier for the individual trial
    """
    ctx = _eval_context.get()
    new_ctx = EvalContext(
        run_id=run_id if run_id is not None else ctx.run_id,
        suite_id=suite_id if suite_id is not None else ctx.suite_id,
        trial_id=trial_id if trial_id is not None else ctx.trial_id,
    )
    _eval_context.set(new_ctx)
    logger.debug(
        f"Set eval context: run={new_ctx.run_id}, suite={new_ctx.suite_id}, trial={new_ctx.trial_id}"
    )


def get_eval_context() -> EvalContext:
    """Get the current evaluation context.

    Returns:
        EvalContext with current run_id, suite_id, and trial_id
    """
    return _eval_context.get()


def clear_eval_context() -> None:
    """Clear the evaluation context, resetting all IDs to None."""
    _eval_context.set(EvalContext())
    logger.debug("Cleared eval context")


# Secret patterns to redact - matches keys containing these substrings
SECRET_PATTERNS = [
    r"api[_-]?key",
    r"secret",
    r"password",
    r"token",
    r"credential",
    r"auth[_-]?key",
    r"private[_-]?key",
    r"access[_-]?key",
]

# Compiled regex for secret key matching
_SECRET_KEY_RE = re.compile("|".join(SECRET_PATTERNS), re.IGNORECASE)

# Redaction placeholder
REDACTED = "[REDACTED]"


def _should_redact_key(key: str) -> bool:
    """Check if a key should be redacted.

    Args:
        key: The dictionary key to check

    Returns:
        True if the key matches a secret pattern
    """
    return bool(_SECRET_KEY_RE.search(key))


def redact_secrets(data: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive values from a dictionary.

    Recursively redacts values for keys matching secret patterns.
    Also redacts environment variable values in nested structures.

    Uses dawn-kestrel's redaction first, then applies our own patterns
    for defense-in-depth to catch patterns DK might miss.

    Args:
        data: Dictionary to redact

    Returns:
        New dictionary with sensitive values replaced by [REDACTED]
    """
    result = data

    if HAS_DK_REDACTION:
        try:
            result = dk_redact_dict(data)
        except Exception:
            result = data

    return _apply_ash_hawk_redaction(result)


def _apply_ash_hawk_redaction(data: dict[str, Any]) -> dict[str, Any]:
    """Apply Ash-Hawk secret patterns on top of DK redaction.

    Args:
        data: Dictionary to redact (may already have DK redaction applied)

    Returns:
        New dictionary with sensitive values redacted
    """
    result: dict[str, Any] = {}

    for key, value in data.items():
        # Check if key should be redacted
        if _should_redact_key(str(key)):
            result[key] = REDACTED
        elif isinstance(value, dict):
            result[key] = _apply_ash_hawk_redaction(value)
        elif isinstance(value, list):
            result[key] = _redact_list_recursive(value)
        else:
            result[key] = value

    return result


def _redact_list_recursive(data: list[Any]) -> list[Any]:
    """Recursively redact sensitive values from a list.

    Args:
        data: List to redact

    Returns:
        New list with sensitive values redacted in nested dicts
    """
    result: list[Any] = []

    for item in data:
        if isinstance(item, dict):
            result.append(_apply_ash_hawk_redaction(item))
        elif isinstance(item, list):
            result.append(_redact_list_recursive(item))
        else:
            result.append(item)

    return result


class EvalContextFormatter(logging.Formatter):
    """Custom log formatter that includes evaluation context.

    Adds run_id, suite_id, and trial_id to log records when available.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with eval context.

        Args:
            record: The log record to format

        Returns:
            Formatted log string
        """
        # Get current eval context
        ctx = get_eval_context()

        # Add context to record
        record.run_id = ctx.run_id or "-"
        record.suite_id = ctx.suite_id or "-"
        record.trial_id = ctx.trial_id or "-"

        return super().format(record)


def setup_eval_logging(level: int = logging.INFO, json_format: bool = False) -> None:
    """Setup logging with eval context support.

    Args:
        level: Logging level (default: INFO)
        json_format: Use JSON format for production (default: False)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handler with eval context formatter
    handler = logging.StreamHandler()
    handler.setLevel(level)

    if json_format:
        # JSON format for production
        formatter = EvalContextFormatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"run_id": "%(run_id)s", "suite_id": "%(suite_id)s", '
            '"trial_id": "%(trial_id)s", "name": "%(name)s", "message": %(message)s}'
        )
    else:
        # Human-readable format
        formatter = EvalContextFormatter(
            "%(asctime)s - %(levelname)s - [run=%(run_id)s suite=%(suite_id)s trial=%(trial_id)s] - %(name)s - %(message)s"
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


__all__ = [
    "EvalContext",
    "set_eval_context",
    "get_eval_context",
    "clear_eval_context",
    "redact_secrets",
    "REDACTED",
    "EvalContextFormatter",
    "setup_eval_logging",
]
