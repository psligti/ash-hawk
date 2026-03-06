"""Policy enforcement for tool surface and resource limits.

This module provides the PolicyEnforcer class which validates tool calls,
filesystem access, network requests, and resource limits against a
ToolSurfacePolicy configuration.

The enforcer returns deterministic PolicyCheckResult objects that indicate
whether an action is allowed, and if not, the specific failure mode.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ash_hawk.types import FailureMode, ToolPermission, ToolSurfacePolicy


@dataclass
class PolicyCheckResult:
    """Result of a policy check.

    Attributes:
        allowed: Whether the action is permitted by the policy.
        failure_mode: The specific failure mode if not allowed, None otherwise.
        reason: Human-readable explanation for the decision.
    """

    allowed: bool
    failure_mode: FailureMode | None = None
    reason: str | None = None


class PolicyEnforcer:
    """Enforces ToolSurfacePolicy during trial execution.

    This class provides methods to validate tool calls, filesystem access,
    network requests, and resource consumption against a configured policy.
    All checks return deterministic PolicyCheckResult objects.

    The enforcer maintains internal state for:
    - Tool call count (for max_tool_calls enforcement)
    - The policy configuration (immutable after initialization)

    Example:
        >>> policy = ToolSurfacePolicy(
        ...     allowed_tools=["read*", "write*"],
        ...     denied_tools=["*bash*"],
        ...     allowed_roots=["/workspace"],
        ...     network_allowed=True,
        ...     max_tool_calls=100,
        ... )
        >>> enforcer = PolicyEnforcer(policy)
        >>> result = enforcer.check_tool("read_file", {"path": "/workspace/file.txt"})
        >>> result.allowed
        True
    """

    def __init__(self, policy: ToolSurfacePolicy) -> None:
        """Initialize the enforcer with a policy configuration.

        Args:
            policy: The ToolSurfacePolicy to enforce.
        """
        self._policy = policy
        self._tool_call_count = 0

    @property
    def tool_call_count(self) -> int:
        """Current count of tool calls made."""
        return self._tool_call_count

    @property
    def policy(self) -> ToolSurfacePolicy:
        """The policy being enforced (read-only access)."""
        return self._policy

    def check_tool(self, tool_name: str, tool_input: dict[str, Any]) -> PolicyCheckResult:
        """Check if a tool call is permitted by the policy.

        This method performs the following checks in order:
        1. Check if max_tool_calls limit has been reached
        2. Check if the tool is in the denylist
        3. Check if the tool is in the allowlist
        4. Fall back to default_permission
        5. If tool has path input, validate path access

        Args:
            tool_name: Name of the tool to check.
            tool_input: Input parameters for the tool call.

        Returns:
            PolicyCheckResult indicating if the call is allowed.
        """
        # Check max_tool_calls limit first
        if self._policy.max_tool_calls is not None:
            if self._tool_call_count >= self._policy.max_tool_calls:
                return PolicyCheckResult(
                    allowed=False,
                    failure_mode=FailureMode.RESOURCE_EXCEEDED,
                    reason=f"Tool call limit ({self._policy.max_tool_calls}) exceeded",
                )

        # Check tool permission using policy's built-in method
        permission = self._policy.is_tool_allowed(tool_name)

        if permission == ToolPermission.DENY:
            return PolicyCheckResult(
                allowed=False,
                failure_mode=FailureMode.TOOL_DENIED,
                reason=f"Tool '{tool_name}' is denied by policy",
            )

        if permission == ToolPermission.ASK:
            return PolicyCheckResult(
                allowed=False,
                failure_mode=FailureMode.POLICY_VIOLATION,
                reason=f"Tool '{tool_name}' requires explicit permission",
            )

        # Check path access for filesystem tools
        path_result = self._check_tool_paths(tool_input)
        if not path_result.allowed:
            return path_result

        return PolicyCheckResult(allowed=True)

    def _check_tool_paths(self, tool_input: dict[str, Any]) -> PolicyCheckResult:
        """Check path access for tools that operate on the filesystem.

        Args:
            tool_input: Tool input parameters that may contain paths.

        Returns:
            PolicyCheckResult for path access.
        """
        # Common path parameter names used by various tools
        path_keys = ["path", "file_path", "dir_path", "directory", "destination", "source"]

        for key in path_keys:
            if key in tool_input:
                path_value = tool_input[key]
                if isinstance(path_value, str):
                    mode: Literal["read", "write"]
                    if key in ["destination", "source"]:
                        mode = "write"
                    else:
                        mode = "read"
                    result = self.check_path(path_value, mode)
                    if not result.allowed:
                        return result

        return PolicyCheckResult(allowed=True)

    def check_path(self, path: str, mode: Literal["read", "write"]) -> PolicyCheckResult:
        """Check if filesystem path access is permitted.

        Validates that the given path is within one of the allowed_roots
        defined in the policy.

        Args:
            path: The filesystem path to check.
            mode: Access mode - "read" or "write". Currently both modes
                  are validated against allowed_roots equally.

        Returns:
            PolicyCheckResult indicating if path access is allowed.

        Note:
            If allowed_roots is empty, all paths are denied by default.
        """
        if not self._policy.allowed_roots:
            return PolicyCheckResult(
                allowed=False,
                failure_mode=FailureMode.POLICY_VIOLATION,
                reason="No allowed roots configured - all path access denied",
            )

        try:
            abs_path = Path(path).resolve()
        except (OSError, ValueError) as e:
            return PolicyCheckResult(
                allowed=False,
                failure_mode=FailureMode.POLICY_VIOLATION,
                reason=f"Invalid path '{path}': {e}",
            )

        for allowed_root in self._policy.allowed_roots:
            try:
                root = Path(allowed_root).resolve()
                # Check if path is within the allowed root
                if str(abs_path).startswith(str(root)):
                    return PolicyCheckResult(allowed=True)
                # Also check if path equals the root
                if abs_path == root:
                    return PolicyCheckResult(allowed=True)
            except (OSError, ValueError):
                # Skip invalid allowed roots
                continue

        return PolicyCheckResult(
            allowed=False,
            failure_mode=FailureMode.POLICY_VIOLATION,
            reason=f"Path '{path}' is outside allowed roots",
        )

    def check_network(self, host: str) -> PolicyCheckResult:
        """Check if network access to a host is permitted.

        Validates network access based on:
        1. Whether network access is globally enabled
        2. Whether the host matches an entry in the allowlist

        Args:
            host: The hostname or IP address to check.

        Returns:
            PolicyCheckResult indicating if network access is allowed.
        """
        if not self._policy.network_allowed:
            return PolicyCheckResult(
                allowed=False,
                failure_mode=FailureMode.POLICY_VIOLATION,
                reason="Network access is disabled by policy",
            )

        if self._policy.network_allowlist:
            # Check if host matches any allowlist pattern
            for allowed_host in self._policy.network_allowlist:
                if fnmatch.fnmatch(host, allowed_host):
                    return PolicyCheckResult(allowed=True)

            return PolicyCheckResult(
                allowed=False,
                failure_mode=FailureMode.POLICY_VIOLATION,
                reason=f"Host '{host}' is not in network allowlist",
            )

        # Network is allowed and no allowlist = all hosts permitted
        return PolicyCheckResult(allowed=True)

    def increment_tool_count(self) -> PolicyCheckResult:
        """Increment the tool call counter and check against limit.

        This method should be called just before a tool execution to
        increment the counter. It returns a failure result if the
        increment would exceed max_tool_calls.

        Returns:
            PolicyCheckResult indicating if the increment is allowed.
            If allowed=True, the counter has been incremented.

        Note:
            This method INCREMENTS first, then checks. So if max_tool_calls=5,
            the 6th call will return failure (count becomes 6 > 5).
        """
        self._tool_call_count += 1

        if self._policy.max_tool_calls is not None:
            if self._tool_call_count > self._policy.max_tool_calls:
                return PolicyCheckResult(
                    allowed=False,
                    failure_mode=FailureMode.RESOURCE_EXCEEDED,
                    reason=f"Tool call limit ({self._policy.max_tool_calls}) exceeded",
                )

        return PolicyCheckResult(allowed=True)

    def check_timeout(self, elapsed_seconds: float) -> PolicyCheckResult:
        """Check if the trial has exceeded its timeout limit.

        Args:
            elapsed_seconds: Time elapsed since trial start.

        Returns:
            PolicyCheckResult indicating if the trial is still within
            its timeout limit.
        """
        if elapsed_seconds > self._policy.timeout_seconds:
            return PolicyCheckResult(
                allowed=False,
                failure_mode=FailureMode.TIMEOUT,
                reason=f"Trial timeout ({self._policy.timeout_seconds}s) exceeded",
            )

        return PolicyCheckResult(allowed=True)

    def check_token_budget(self, tokens_used: int) -> PolicyCheckResult:
        """Check if token usage is within budget.

        Args:
            tokens_used: Total tokens consumed so far.

        Returns:
            PolicyCheckResult indicating if token usage is within limit.
            Returns allowed=True if no token_budget is configured.
        """
        if self._policy.token_budget is not None:
            if tokens_used > self._policy.token_budget:
                return PolicyCheckResult(
                    allowed=False,
                    failure_mode=FailureMode.RESOURCE_EXCEEDED,
                    reason=f"Token budget ({self._policy.token_budget}) exceeded",
                )

        return PolicyCheckResult(allowed=True)

    def check_cost_budget(self, cost_usd: float) -> PolicyCheckResult:
        """Check if cost is within budget.

        Args:
            cost_usd: Total cost in USD consumed so far.

        Returns:
            PolicyCheckResult indicating if cost is within limit.
            Returns allowed=True if no cost_budget_usd is configured.
        """
        if self._policy.cost_budget_usd is not None:
            if cost_usd > self._policy.cost_budget_usd:
                return PolicyCheckResult(
                    allowed=False,
                    failure_mode=FailureMode.RESOURCE_EXCEEDED,
                    reason=f"Cost budget (${self._policy.cost_budget_usd}) exceeded",
                )

        return PolicyCheckResult(allowed=True)

    def check_file_size(self, size_bytes: int) -> PolicyCheckResult:
        """Check if file size is within allowed limit.

        Args:
            size_bytes: Size of the file in bytes.

        Returns:
            PolicyCheckResult indicating if file size is allowed.
            Returns allowed=True if no max_file_size_bytes is configured.
        """
        if self._policy.max_file_size_bytes is not None:
            if size_bytes > self._policy.max_file_size_bytes:
                return PolicyCheckResult(
                    allowed=False,
                    failure_mode=FailureMode.RESOURCE_EXCEEDED,
                    reason=f"File size ({size_bytes} bytes) exceeds limit ({self._policy.max_file_size_bytes} bytes)",
                )

        return PolicyCheckResult(allowed=True)

    def check_env_var(self, var_name: str) -> PolicyCheckResult:
        """Check if access to an environment variable is permitted.

        Args:
            var_name: Name of the environment variable to access.

        Returns:
            PolicyCheckResult indicating if access is allowed.
            Returns allowed=True if env_vars_allowed is empty (no restrictions).
        """
        if self._policy.env_vars_allowed:
            # Check exact match first
            if var_name in self._policy.env_vars_allowed:
                return PolicyCheckResult(allowed=True)

            # Check glob patterns
            for pattern in self._policy.env_vars_allowed:
                if fnmatch.fnmatch(var_name, pattern):
                    return PolicyCheckResult(allowed=True)

            return PolicyCheckResult(
                allowed=False,
                failure_mode=FailureMode.POLICY_VIOLATION,
                reason=f"Environment variable '{var_name}' access not allowed",
            )

        # No env_vars_allowed configured = no restrictions
        return PolicyCheckResult(allowed=True)

    def create_snapshot(self) -> ToolSurfacePolicy:
        """Create a deep copy of the current policy.

        This is used to store the policy state with trial records for
        reproducibility and auditing purposes.

        Returns:
            A deep copy of the current ToolSurfacePolicy.
        """
        return self._policy.model_copy(deep=True)

    def reset_tool_count(self) -> None:
        """Reset the tool call counter to zero.

        This is typically used when starting a new trial with the same
        enforcer instance.
        """
        self._tool_call_count = 0
