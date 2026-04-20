from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest

from ash_hawk.thin_runtime.selection_policy import ToolSelectionDecision


@pytest.fixture(autouse=True)
def stub_live_runtime_paths() -> Iterator[None]:
    with patch(
        "ash_hawk.thin_runtime.runner.AgenticLoopRunner._execute_delegation"
    ) as mock_delegate:

        def _fake_execute_delegation(result, *, context, depth):
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
            with patch(
                "ash_hawk.thin_runtime.selection_policy.call_model_structured"
            ) as mock_policy:

                def _fake_policy_response(model_type, system_prompt, user_prompt):
                    del system_prompt
                    if model_type is not ToolSelectionDecision:
                        return None
                    eligible: list[str] = []
                    marker = "Eligible tools:\n"
                    if marker in user_prompt:
                        block = user_prompt.split(marker, 1)[1].split("\n\n", 1)[0]
                        for line in block.splitlines():
                            if not line.startswith("- "):
                                continue
                            name = line[2:].split(":", 1)[0].strip()
                            if name:
                                eligible.append(name)
                    if not eligible:
                        return None
                    preferred = (
                        "delegate_task"
                        if "delegate_task" in eligible and "delegate" in user_prompt.lower()
                        else eligible[0]
                    )
                    return ToolSelectionDecision(
                        selected_tool=preferred,
                        source="test_policy",
                        rationale="Deterministic policy stub for unit tests.",
                        considered_tools=eligible,
                    )

                mock_policy.side_effect = _fake_policy_response
                yield
