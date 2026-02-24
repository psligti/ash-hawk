"""Policy enforcement module for ash-hawk.

This module provides policy enforcement capabilities that validate tool calls,
filesystem access, network requests, and resource limits against a configured
ToolSurfacePolicy.

Key components:
- PolicyEnforcer: Main class for enforcing policy during trial execution
- PolicyCheckResult: Dataclass representing the result of a policy check
"""

from ash_hawk.policy.enforcer import PolicyCheckResult, PolicyEnforcer

__all__ = ["PolicyCheckResult", "PolicyEnforcer"]
