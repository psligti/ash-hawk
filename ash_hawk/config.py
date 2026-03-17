"""Ash Hawk - Configuration management using pydantic-settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pydantic_settings
from pydantic import Field, field_validator
from pydantic_settings import (
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
)
from pydantic_settings.main import SettingsConfigDict

from ash_hawk.types import ToolSurfacePolicy

__all__ = [
    "EvalConfig",
    "config",
    "get_config",
    "reload_config",
]

StorageBackend = Literal["file", "sqlite", "postgres", "s3"]


class EvalConfig(pydantic_settings.BaseSettings):
    """Configuration for Ash Hawk evaluation harness."""

    parallelism: int = Field(
        default=4,
        ge=1,
        le=256,
        description="Maximum concurrent evaluation trials",
    )

    default_timeout_seconds: float = Field(
        default=300.0,
        ge=1.0,
        description="Default timeout for evaluation runs in seconds",
    )

    # Queue-based throttling configuration
    llm_max_workers: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Maximum concurrent LLM requests across all trials",
    )

    llm_timeout_seconds: float = Field(
        default=300.0,
        ge=1.0,
        description="Timeout for queued LLM requests in seconds",
    )

    trial_max_workers: int = Field(
        default=4,
        ge=1,
        le=256,
        description="Maximum concurrent trials via execution queue (replaces semaphore)",
    )

    storage_backend: StorageBackend = Field(
        default="file",
        description="Storage backend: file, sqlite, postgres, or s3",
    )

    storage_path: str = Field(
        default=".ash-hawk",
        description="Path for file-based storage (supports ~ expansion)",
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )

    default_tool_policy: ToolSurfacePolicy = Field(
        default_factory=ToolSurfacePolicy,
        description="Default tool access policy for evaluations",
    )

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got {v}")
        return upper_v

    @field_validator("storage_backend", mode="before")
    @classmethod
    def validate_storage_backend(cls, v: str) -> str:
        valid_backends = {"file", "sqlite", "postgres", "s3"}
        lower_v = v.lower()
        if lower_v not in valid_backends:
            raise ValueError(f"storage_backend must be one of {valid_backends}, got {v}")
        return lower_v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[pydantic_settings.BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del env_settings, dotenv_settings

        model_env_nested_delimiter = settings_cls.model_config.get("env_nested_delimiter")
        env_nested_delimiter = (
            model_env_nested_delimiter if isinstance(model_env_nested_delimiter, str) else None
        )
        model_case_sensitive = settings_cls.model_config.get("case_sensitive")
        case_sensitive = model_case_sensitive if isinstance(model_case_sensitive, bool) else None

        dotenv_paths = _dotenv_paths(settings_cls)
        return (
            init_settings,
            EnvSettingsSource(
                settings_cls,
                env_prefix="ASH_HAWK_",
                env_nested_delimiter=env_nested_delimiter,
                case_sensitive=case_sensitive,
            ),
            DotEnvSettingsSource(
                settings_cls,
                env_prefix="ASH_HAWK_",
                env_file=dotenv_paths,
                env_nested_delimiter=env_nested_delimiter,
                case_sensitive=case_sensitive,
            ),
            file_secret_settings,
        )

    def storage_path_resolved(self) -> Path:
        return Path(self.storage_path).expanduser().resolve()


APP_DIR_NAME = "ash-hawk"


def _xdg_base_dir(env_var_name: str, fallback: Path) -> Path:
    env_value = os.getenv(env_var_name)
    if env_value:
        return Path(env_value).expanduser()
    return fallback


def _app_base_dirs(kind: str) -> Path:
    home = Path.home()
    if kind == "config":
        base = _xdg_base_dir("XDG_CONFIG_HOME", home / ".config")
    elif kind == "data":
        base = _xdg_base_dir("XDG_DATA_HOME", home / ".local" / "share")
    elif kind == "cache":
        base = _xdg_base_dir("XDG_CACHE_HOME", home / ".cache")
    else:
        raise ValueError(f"Unsupported app dir kind: {kind}")

    return base / APP_DIR_NAME


def _dotenv_paths(settings_cls: type[pydantic_settings.BaseSettings]) -> tuple[Path | str, ...]:
    explicit_env_files = settings_cls.model_config.get("env_file")
    if explicit_env_files is not None:
        if isinstance(explicit_env_files, (str, Path)):
            return (explicit_env_files,)
        return tuple(explicit_env_files)

    config_dir = _app_base_dirs("config")
    return (".env", config_dir / ".env")


config: EvalConfig = EvalConfig()


def get_config() -> EvalConfig:
    """Get the global configuration singleton instance."""
    return config


def reload_config() -> EvalConfig:
    """Reload configuration by creating a new EvalConfig instance."""
    global config
    config = EvalConfig()
    return config
