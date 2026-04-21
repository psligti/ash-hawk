from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from ash_hawk.thin_runtime.context import ContextSnapshot
    from ash_hawk.thin_runtime.models import DelegationRecord, ToolResult


@pytest.fixture(autouse=True)
def stub_live_runtime_paths() -> Iterator[None]:
    with patch(
        "ash_hawk.thin_runtime.runner.AgenticLoopRunner._execute_delegation"
    ) as mock_delegate:

        def _fake_execute_delegation(
            result: ToolResult, *, context: ContextSnapshot, depth: int
        ) -> tuple[DelegationRecord | None, ToolResult | None]:
            del context, depth
            delegation = result.payload.delegation
            if delegation is None:
                return None, None

            from ash_hawk.thin_runtime.models import DelegationRecord

            return (
                DelegationRecord(
                    agent_name=delegation.agent_name,
                    goal_id=delegation.goal_id,
                    selected_tool_names=[],
                    success=True,
                    error=None,
                ),
                None,
            )

        mock_delegate.side_effect = _fake_execute_delegation

        with patch("ash_hawk.thin_runtime.llm_client.call_model_text") as mock_llm_text:
            mock_llm_text.return_value = None
            yield
