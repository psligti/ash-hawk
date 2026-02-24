"""Tests for ash_hawk.policy.enforcer module."""

import pytest

from ash_hawk.policy import PolicyCheckResult, PolicyEnforcer
from ash_hawk.types import FailureMode, ToolPermission, ToolSurfacePolicy


class TestPolicyCheckResult:
    """Test PolicyCheckResult dataclass."""

    def test_allowed_result(self):
        result = PolicyCheckResult(allowed=True)
        assert result.allowed is True
        assert result.failure_mode is None
        assert result.reason is None

    def test_denied_result(self):
        result = PolicyCheckResult(
            allowed=False,
            failure_mode=FailureMode.TOOL_DENIED,
            reason="Tool not allowed",
        )
        assert result.allowed is False
        assert result.failure_mode == FailureMode.TOOL_DENIED
        assert result.reason == "Tool not allowed"


class TestPolicyEnforcerInit:
    """Test PolicyEnforcer initialization."""

    def test_init_with_policy(self):
        policy = ToolSurfacePolicy(allowed_tools=["read"])
        enforcer = PolicyEnforcer(policy)
        assert enforcer.tool_call_count == 0
        assert enforcer.policy == policy

    def test_default_tool_call_count(self):
        enforcer = PolicyEnforcer(ToolSurfacePolicy())
        assert enforcer.tool_call_count == 0


class TestCheckTool:
    """Test PolicyEnforcer.check_tool method."""

    def test_allowed_tool_passes(self):
        policy = ToolSurfacePolicy(
            allowed_tools=["read", "write"],
            default_permission=ToolPermission.DENY,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_tool("read", {})
        assert result.allowed is True

    def test_denied_tool_blocked(self):
        policy = ToolSurfacePolicy(
            allowed_tools=["read", "write"],
            denied_tools=["write"],
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_tool("write", {})
        assert result.allowed is False
        assert result.failure_mode == FailureMode.TOOL_DENIED
        assert "denied" in result.reason.lower()

    def test_unknown_tool_with_default_deny(self):
        policy = ToolSurfacePolicy(
            default_permission=ToolPermission.DENY,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_tool("unknown_tool", {})
        assert result.allowed is False
        assert result.failure_mode == FailureMode.TOOL_DENIED

    def test_unknown_tool_with_default_ask(self):
        policy = ToolSurfacePolicy(
            default_permission=ToolPermission.ASK,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_tool("unknown_tool", {})
        assert result.allowed is False
        assert result.failure_mode == FailureMode.POLICY_VIOLATION

    def test_unknown_tool_with_default_allow(self):
        policy = ToolSurfacePolicy(
            default_permission=ToolPermission.ALLOW,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_tool("unknown_tool", {})
        assert result.allowed is True

    def test_glob_pattern_allowed(self):
        policy = ToolSurfacePolicy(
            allowed_tools=["read*", "grep"],
            denied_tools=["*bash*"],
        )
        enforcer = PolicyEnforcer(policy)

        assert enforcer.check_tool("read", {}).allowed is True
        assert enforcer.check_tool("read_file", {}).allowed is True
        assert enforcer.check_tool("grep", {}).allowed is True
        assert enforcer.check_tool("run_bash_cmd", {}).allowed is False

    def test_denylist_takes_precedence(self):
        policy = ToolSurfacePolicy(
            allowed_tools=["read", "write", "execute"],
            denied_tools=["write"],
        )
        enforcer = PolicyEnforcer(policy)

        assert enforcer.check_tool("read", {}).allowed is True
        assert enforcer.check_tool("write", {}).allowed is False
        assert enforcer.check_tool("write", {}).failure_mode == FailureMode.TOOL_DENIED

    def test_max_tool_calls_exceeded(self):
        policy = ToolSurfacePolicy(
            allowed_tools=["read"],
            max_tool_calls=3,
        )
        enforcer = PolicyEnforcer(policy)

        enforcer._tool_call_count = 3

        result = enforcer.check_tool("read", {})
        assert result.allowed is False
        assert result.failure_mode == FailureMode.RESOURCE_EXCEEDED
        assert "limit" in result.reason.lower()


class TestCheckPath:
    """Test PolicyEnforcer.check_path method."""

    def test_path_within_allowed_root(self):
        policy = ToolSurfacePolicy(
            allowed_roots=["/workspace"],
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_path("/workspace/file.txt", "read")
        assert result.allowed is True

    def test_path_outside_allowed_root(self):
        policy = ToolSurfacePolicy(
            allowed_roots=["/workspace"],
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_path("/etc/passwd", "read")
        assert result.allowed is False
        assert result.failure_mode == FailureMode.POLICY_VIOLATION
        assert "outside" in result.reason.lower()

    def test_no_allowed_roots_denies_all(self):
        policy = ToolSurfacePolicy(
            allowed_roots=[],
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_path("/any/path", "read")
        assert result.allowed is False
        assert result.failure_mode == FailureMode.POLICY_VIOLATION

    def test_multiple_allowed_roots(self):
        policy = ToolSurfacePolicy(
            allowed_roots=["/workspace", "/tmp"],
        )
        enforcer = PolicyEnforcer(policy)

        assert enforcer.check_path("/workspace/file.txt", "read").allowed is True
        assert enforcer.check_path("/tmp/cache", "read").allowed is True
        assert enforcer.check_path("/etc/passwd", "read").allowed is False

    def test_path_subdirectory_access(self):
        policy = ToolSurfacePolicy(
            allowed_roots=["/workspace"],
        )
        enforcer = PolicyEnforcer(policy)

        result = enforcer.check_path("/workspace/deeply/nested/file.txt", "read")
        assert result.allowed is True

    def test_tool_with_path_input(self):
        policy = ToolSurfacePolicy(
            allowed_tools=["read_file"],
            allowed_roots=["/workspace"],
        )
        enforcer = PolicyEnforcer(policy)

        result = enforcer.check_tool("read_file", {"path": "/workspace/file.txt"})
        assert result.allowed is True

        result = enforcer.check_tool("read_file", {"path": "/etc/passwd"})
        assert result.allowed is False
        assert result.failure_mode == FailureMode.POLICY_VIOLATION

    def test_tool_with_file_path_input(self):
        policy = ToolSurfacePolicy(
            allowed_tools=["edit_file"],
            allowed_roots=["/workspace"],
        )
        enforcer = PolicyEnforcer(policy)

        result = enforcer.check_tool("edit_file", {"file_path": "/workspace/file.txt"})
        assert result.allowed is True


class TestCheckNetwork:
    """Test PolicyEnforcer.check_network method."""

    def test_network_disabled(self):
        policy = ToolSurfacePolicy(
            network_allowed=False,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_network("api.example.com")
        assert result.allowed is False
        assert result.failure_mode == FailureMode.POLICY_VIOLATION
        assert "disabled" in result.reason.lower()

    def test_network_enabled_no_allowlist(self):
        policy = ToolSurfacePolicy(
            network_allowed=True,
            network_allowlist=[],
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_network("any.host.com")
        assert result.allowed is True

    def test_network_with_allowlist_matching(self):
        policy = ToolSurfacePolicy(
            network_allowed=True,
            network_allowlist=["api.example.com", "*.internal.com"],
        )
        enforcer = PolicyEnforcer(policy)

        assert enforcer.check_network("api.example.com").allowed is True
        assert enforcer.check_network("service.internal.com").allowed is True
        assert enforcer.check_network("external.com").allowed is False

    def test_network_glob_pattern(self):
        policy = ToolSurfacePolicy(
            network_allowed=True,
            network_allowlist=["*.github.com"],
        )
        enforcer = PolicyEnforcer(policy)

        assert enforcer.check_network("api.github.com").allowed is True
        assert enforcer.check_network("raw.githubusercontent.com").allowed is False


class TestIncrementToolCount:
    """Test PolicyEnforcer.increment_tool_count method."""

    def test_increment_without_limit(self):
        policy = ToolSurfacePolicy(
            max_tool_calls=None,
        )
        enforcer = PolicyEnforcer(policy)

        for _ in range(10):
            result = enforcer.increment_tool_count()
            assert result.allowed is True

        assert enforcer.tool_call_count == 10

    def test_increment_with_limit(self):
        policy = ToolSurfacePolicy(
            max_tool_calls=3,
        )
        enforcer = PolicyEnforcer(policy)

        result = enforcer.increment_tool_count()
        assert result.allowed is True
        assert enforcer.tool_call_count == 1

        result = enforcer.increment_tool_count()
        assert result.allowed is True
        assert enforcer.tool_call_count == 2

        result = enforcer.increment_tool_count()
        assert result.allowed is True
        assert enforcer.tool_call_count == 3

        result = enforcer.increment_tool_count()
        assert result.allowed is False
        assert result.failure_mode == FailureMode.RESOURCE_EXCEEDED
        assert enforcer.tool_call_count == 4

    def test_reset_tool_count(self):
        policy = ToolSurfacePolicy(
            max_tool_calls=2,
        )
        enforcer = PolicyEnforcer(policy)

        enforcer.increment_tool_count()
        enforcer.increment_tool_count()
        assert enforcer.tool_call_count == 2

        enforcer.reset_tool_count()
        assert enforcer.tool_call_count == 0

        result = enforcer.increment_tool_count()
        assert result.allowed is True


class TestCheckTimeout:
    """Test PolicyEnforcer.check_timeout method."""

    def test_within_timeout(self):
        policy = ToolSurfacePolicy(
            timeout_seconds=300.0,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_timeout(100.0)
        assert result.allowed is True

    def test_exceeds_timeout(self):
        policy = ToolSurfacePolicy(
            timeout_seconds=300.0,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_timeout(301.0)
        assert result.allowed is False
        assert result.failure_mode == FailureMode.TIMEOUT

    def test_exactly_at_timeout(self):
        policy = ToolSurfacePolicy(
            timeout_seconds=300.0,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_timeout(300.0)
        assert result.allowed is True


class TestCheckTokenBudget:
    """Test PolicyEnforcer.check_token_budget method."""

    def test_no_token_budget(self):
        policy = ToolSurfacePolicy(
            token_budget=None,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_token_budget(1000000)
        assert result.allowed is True

    def test_within_token_budget(self):
        policy = ToolSurfacePolicy(
            token_budget=10000,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_token_budget(5000)
        assert result.allowed is True

    def test_exceeds_token_budget(self):
        policy = ToolSurfacePolicy(
            token_budget=10000,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_token_budget(10001)
        assert result.allowed is False
        assert result.failure_mode == FailureMode.RESOURCE_EXCEEDED


class TestCheckCostBudget:
    """Test PolicyEnforcer.check_cost_budget method."""

    def test_no_cost_budget(self):
        policy = ToolSurfacePolicy(
            cost_budget_usd=None,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_cost_budget(1000.0)
        assert result.allowed is True

    def test_within_cost_budget(self):
        policy = ToolSurfacePolicy(
            cost_budget_usd=10.0,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_cost_budget(5.0)
        assert result.allowed is True

    def test_exceeds_cost_budget(self):
        policy = ToolSurfacePolicy(
            cost_budget_usd=10.0,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_cost_budget(10.01)
        assert result.allowed is False
        assert result.failure_mode == FailureMode.RESOURCE_EXCEEDED


class TestCheckFileSize:
    """Test PolicyEnforcer.check_file_size method."""

    def test_no_file_size_limit(self):
        policy = ToolSurfacePolicy(
            max_file_size_bytes=None,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_file_size(10_000_000)
        assert result.allowed is True

    def test_within_file_size_limit(self):
        policy = ToolSurfacePolicy(
            max_file_size_bytes=1024 * 1024,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_file_size(512 * 1024)
        assert result.allowed is True

    def test_exceeds_file_size_limit(self):
        policy = ToolSurfacePolicy(
            max_file_size_bytes=1024 * 1024,
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_file_size(2 * 1024 * 1024)
        assert result.allowed is False
        assert result.failure_mode == FailureMode.RESOURCE_EXCEEDED


class TestCheckEnvVar:
    """Test PolicyEnforcer.check_env_var method."""

    def test_no_env_var_restrictions(self):
        policy = ToolSurfacePolicy(
            env_vars_allowed=[],
        )
        enforcer = PolicyEnforcer(policy)
        result = enforcer.check_env_var("SECRET_KEY")
        assert result.allowed is True

    def test_allowed_env_var(self):
        policy = ToolSurfacePolicy(
            env_vars_allowed=["PATH", "HOME", "APP_*"],
        )
        enforcer = PolicyEnforcer(policy)

        assert enforcer.check_env_var("PATH").allowed is True
        assert enforcer.check_env_var("HOME").allowed is True
        assert enforcer.check_env_var("APP_CONFIG").allowed is True

    def test_denied_env_var(self):
        policy = ToolSurfacePolicy(
            env_vars_allowed=["PATH", "HOME"],
        )
        enforcer = PolicyEnforcer(policy)

        result = enforcer.check_env_var("SECRET_KEY")
        assert result.allowed is False
        assert result.failure_mode == FailureMode.POLICY_VIOLATION


class TestCreateSnapshot:
    """Test PolicyEnforcer.create_snapshot method."""

    def test_snapshot_returns_copy(self):
        policy = ToolSurfacePolicy(
            allowed_tools=["read", "write"],
            max_tool_calls=100,
        )
        enforcer = PolicyEnforcer(policy)

        snapshot = enforcer.create_snapshot()
        assert snapshot.allowed_tools == ["read", "write"]
        assert snapshot.max_tool_calls == 100

    def test_snapshot_is_independent(self):
        policy = ToolSurfacePolicy(
            allowed_tools=["read"],
        )
        enforcer = PolicyEnforcer(policy)

        snapshot = enforcer.create_snapshot()
        snapshot.allowed_tools.append("write")

        assert enforcer.policy.allowed_tools == ["read"]
        assert snapshot.allowed_tools == ["read", "write"]


class TestDeterministicFailureModes:
    """Test that failure modes are deterministic."""

    def test_same_input_same_failure_mode(self):
        policy = ToolSurfacePolicy(
            denied_tools=["dangerous"],
        )
        enforcer = PolicyEnforcer(policy)

        result1 = enforcer.check_tool("dangerous", {})
        result2 = enforcer.check_tool("dangerous", {})

        assert result1.failure_mode == result2.failure_mode
        assert result1.reason == result2.reason

    def test_failure_mode_is_enum(self):
        policy = ToolSurfacePolicy(
            denied_tools=["test"],
        )
        enforcer = PolicyEnforcer(policy)

        result = enforcer.check_tool("test", {})
        assert isinstance(result.failure_mode, FailureMode)


class TestIntegration:
    """Integration tests for PolicyEnforcer."""

    def test_full_policy_enforcement_workflow(self):
        policy = ToolSurfacePolicy(
            allowed_tools=["read*", "write*", "grep"],
            denied_tools=["*bash*"],
            allowed_roots=["/workspace"],
            network_allowed=True,
            network_allowlist=["api.example.com"],
            max_tool_calls=10,
            timeout_seconds=60.0,
        )
        enforcer = PolicyEnforcer(policy)

        assert enforcer.check_tool("read_file", {"path": "/workspace/test.txt"}).allowed
        assert not enforcer.check_tool("run_bash", {}).allowed
        assert enforcer.check_network("api.example.com").allowed
        assert not enforcer.check_network("external.com").allowed
        assert enforcer.check_path("/workspace/subdir/file", "read").allowed
        assert not enforcer.check_path("/etc/passwd", "read").allowed
        assert enforcer.check_timeout(30.0).allowed

        for _ in range(10):
            enforcer.increment_tool_count()

        assert not enforcer.increment_tool_count().allowed

        snapshot = enforcer.create_snapshot()
        assert snapshot.max_tool_calls == 10
