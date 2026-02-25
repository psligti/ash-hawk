"""pyproject.toml configuration loading.

Reads [tool.ash-hawk] section from pyproject.toml for project-level defaults.

Example pyproject.toml:
    [tool.ash-hawk]
    suite_patterns = ["*-suite.yaml", "*.suite.yaml"]
    search_paths = ["evals", "tests/evals"]
    parallelism = 4
    default_timeout = 300
    storage_backend = "file"
    storage_path = ".ash-hawk-results"
    provider = "anthropic"
    model = "claude-3-5-sonnet"
    log_level = "INFO"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pydantic as pd


class PyprojectConfig(pd.BaseModel):
    """Configuration from [tool.ash-hawk] in pyproject.toml."""

    # Discovery
    suite_patterns: list[str] = pd.Field(
        default=["*-suite.yaml", "*.suite.yaml"],
        description="Glob patterns for suite discovery",
    )
    search_paths: list[str] = pd.Field(
        default=["evals", "tests/evals"],
        description="Directories to search for suites",
    )

    # Execution defaults
    parallelism: int = pd.Field(
        default=4,
        ge=1,
        description="Default parallelism for suite execution",
    )
    default_timeout: int = pd.Field(
        default=300,
        ge=1,
        description="Default timeout in seconds",
    )

    # Storage
    storage_backend: Literal["file", "sqlite", "postgres", "s3"] = pd.Field(
        default="file",
        description="Storage backend type",
    )
    storage_path: str = pd.Field(
        default=".ash-hawk-results",
        description="Path for file-based storage",
    )

    # Agent defaults
    provider: str | None = pd.Field(
        default=None,
        description="Default LLM provider",
    )
    model: str | None = pd.Field(
        default=None,
        description="Default model identifier",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = pd.Field(
        default="INFO",
        description="Logging level",
    )

    # Additional settings
    extra: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Additional configuration options",
    )

    model_config = pd.ConfigDict(extra="allow")


def load_pyproject_config(
    start_dir: Path | None = None,
) -> PyprojectConfig:
    """Load configuration from pyproject.toml.

    Searches upward from start_dir for pyproject.toml.

    Args:
        start_dir: Directory to start searching from.
                  If None, uses current working directory.

    Returns:
        PyprojectConfig with loaded or default values.
    """
    start_dir = start_dir or Path.cwd()

    # Find pyproject.toml
    pyproject_path = _find_pyproject(start_dir)

    if pyproject_path is None:
        return PyprojectConfig()

    # Parse TOML
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            # Fallback if tomli not installed
            return PyprojectConfig()

    try:
        content = pyproject_path.read_bytes()
        data = tomllib.loads(content.decode("utf-8"))
    except Exception:
        return PyprojectConfig()

    # Extract [tool.ash-hawk] section
    ash_hawk_config = data.get("tool", {}).get("ash-hawk", {})

    return PyprojectConfig(**ash_hawk_config)


def _find_pyproject(start_dir: Path) -> Path | None:
    """Find pyproject.toml by searching upward.

    Args:
        start_dir: Directory to start searching from.

    Returns:
        Path to pyproject.toml or None if not found.
    """
    current = start_dir.resolve()

    while True:
        pyproject = current / "pyproject.toml"
        if pyproject.exists():
            return pyproject

        if current.parent == current:
            return None
        current = current.parent


def merge_pyproject_with_cli(
    pyproject_config: PyprojectConfig,
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge pyproject config with CLI overrides.

    CLI overrides take precedence over pyproject settings.

    Args:
        pyproject_config: Configuration from pyproject.toml.
        cli_overrides: CLI-provided overrides (None values are ignored).

    Returns:
        Merged configuration dictionary.
    """
    result = pyproject_config.model_dump()

    for key, value in cli_overrides.items():
        if value is not None:
            result[key] = value

    return result


__all__ = [
    "PyprojectConfig",
    "load_pyproject_config",
    "merge_pyproject_with_cli",
]
