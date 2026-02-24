"""Tests for Ash Hawk configuration system."""

import os
from pathlib import Path
from unittest import mock

import pytest

from ash_hawk.config import EvalConfig, get_config, reload_config
from ash_hawk.types import ToolPermission, ToolSurfacePolicy


class TestEvalConfig:
    def test_default_values(self):
        config = EvalConfig()
        assert config.parallelism == 4
        assert config.default_timeout_seconds == 300.0
        assert config.storage_backend == "file"
        assert config.storage_path == ".ash-hawk"
        assert config.log_level == "INFO"
        assert isinstance(config.default_tool_policy, ToolSurfacePolicy)

    def test_parallelism_validation_min(self):
        config = EvalConfig(parallelism=1)
        assert config.parallelism == 1

    def test_parallelism_validation_max(self):
        config = EvalConfig(parallelism=256)
        assert config.parallelism == 256

    def test_parallelism_validation_below_min(self):
        with pytest.raises(ValueError):
            EvalConfig(parallelism=0)

    def test_parallelism_validation_above_max(self):
        with pytest.raises(ValueError):
            EvalConfig(parallelism=257)

    def test_timeout_validation(self):
        config = EvalConfig(default_timeout_seconds=60.0)
        assert config.default_timeout_seconds == 60.0

    def test_timeout_validation_below_min(self):
        with pytest.raises(ValueError):
            EvalConfig(default_timeout_seconds=0.5)

    def test_log_level_validation_case_insensitive(self):
        config = EvalConfig(log_level="debug")
        assert config.log_level == "DEBUG"

        config = EvalConfig(log_level="Warning")
        assert config.log_level == "WARNING"

    def test_log_level_validation_invalid(self):
        with pytest.raises(ValueError):
            EvalConfig(log_level="INVALID")

    def test_storage_backend_validation(self):
        for backend in ["file", "sqlite", "postgres", "s3"]:
            config = EvalConfig(storage_backend=backend)
            assert config.storage_backend == backend

    def test_storage_backend_validation_case_insensitive(self):
        config = EvalConfig(storage_backend="SQLITE")
        assert config.storage_backend == "sqlite"

    def test_storage_backend_validation_invalid(self):
        with pytest.raises(ValueError):
            EvalConfig(storage_backend="invalid")

    def test_custom_tool_policy(self):
        policy = ToolSurfacePolicy(
            default_permission=ToolPermission.ALLOW,
            allowed_tools=["*"],
            denied_tools=[],
        )
        config = EvalConfig(default_tool_policy=policy)
        assert config.default_tool_policy.default_permission == ToolPermission.ALLOW

    def test_storage_path_resolved(self):
        config = EvalConfig(storage_path="~/test-path")
        resolved = config.storage_path_resolved()
        assert isinstance(resolved, Path)
        assert str(resolved).startswith(str(Path.home()))

    def test_storage_path_resolved_relative(self):
        config = EvalConfig(storage_path=".ash-hawk")
        resolved = config.storage_path_resolved()
        assert isinstance(resolved, Path)
        assert resolved.is_absolute()


class TestEvalConfigEnvVars:
    def test_env_var_override_parallelism(self):
        with mock.patch.dict(os.environ, {"ASH_HAWK_PARALLELISM": "8"}):
            config = EvalConfig()
            assert config.parallelism == 8

    def test_env_var_override_timeout(self):
        with mock.patch.dict(os.environ, {"ASH_HAWK_DEFAULT_TIMEOUT_SECONDS": "600.0"}):
            config = EvalConfig()
            assert config.default_timeout_seconds == 600.0

    def test_env_var_override_storage_backend(self):
        with mock.patch.dict(os.environ, {"ASH_HAWK_STORAGE_BACKEND": "sqlite"}):
            config = EvalConfig()
            assert config.storage_backend == "sqlite"

    def test_env_var_override_log_level(self):
        with mock.patch.dict(os.environ, {"ASH_HAWK_LOG_LEVEL": "DEBUG"}):
            config = EvalConfig()
            assert config.log_level == "DEBUG"

    def test_env_var_override_storage_path(self):
        with mock.patch.dict(os.environ, {"ASH_HAWK_STORAGE_PATH": "/custom/path"}):
            config = EvalConfig()
            assert config.storage_path == "/custom/path"


class TestGetConfig:
    def test_get_config_returns_singleton(self):
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reload_config_creates_new_instance(self):
        from ash_hawk import config as config_module

        old_config = config_module.config
        new_config = reload_config()
        assert new_config is not old_config
        assert config_module.config is new_config


class TestToolSurfacePolicyDefaults:
    def test_default_policy_restrictive(self):
        config = EvalConfig()
        policy = config.default_tool_policy

        # Default policy is restrictive - ASK for unknown tools
        assert policy.default_permission == ToolPermission.ASK
        # No tools are explicitly allowed by default (empty allowlist)
        assert policy.allowed_tools == []
        # No tools are explicitly denied by default (relies on default_permission)
        assert policy.denied_tools == []
        # Network access is disabled by default
        assert policy.network_allowed is False
        # Filesystem roots are empty by default
        assert policy.allowed_roots == []
