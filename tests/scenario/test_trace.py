from __future__ import annotations

import json
from pathlib import Path

import pytest

from ash_hawk.scenario.trace import (
    EVENT_TYPE_ARTIFACT,
    EVENT_TYPE_BUDGET,
    EVENT_TYPE_CANDIDATE_EVALUATED,
    EVENT_TYPE_DIFF,
    EVENT_TYPE_DIMENSION_SAMPLED,
    EVENT_TYPE_MODEL_MAP,
    EVENT_TYPE_MODEL_MESSAGE,
    EVENT_TYPE_MUTATION_APPLIED,
    EVENT_TYPE_POLICY_DECISION,
    EVENT_TYPE_REJECTION,
    EVENT_TYPE_TODO,
    EVENT_TYPE_TOOL_CALL,
    EVENT_TYPE_TOOL_RESULT,
    EVENT_TYPE_VERIFICATION,
    ArtifactEvent,
    BudgetEvent,
    CandidateEvaluatedEvent,
    DiffEvent,
    DimensionSampledEvent,
    ModelMessageEvent,
    MutationAppliedEvent,
    PolicyDecisionEvent,
    RejectionEvent,
    TodoEvent,
    ToolCallEvent,
    ToolResultEvent,
    TraceEvent,
    VerificationEvent,
    append_trace_jsonl,
    iter_trace_jsonl,
    write_trace_jsonl,
)


class TestEventRoundTrips:
    @pytest.mark.parametrize(
        "event_cls,event_type_literal",
        [
            (ModelMessageEvent, EVENT_TYPE_MODEL_MESSAGE),
            (ToolCallEvent, EVENT_TYPE_TOOL_CALL),
            (ToolResultEvent, EVENT_TYPE_TOOL_RESULT),
            (VerificationEvent, EVENT_TYPE_VERIFICATION),
            (TodoEvent, EVENT_TYPE_TODO),
            (DiffEvent, EVENT_TYPE_DIFF),
            (ArtifactEvent, EVENT_TYPE_ARTIFACT),
            (PolicyDecisionEvent, EVENT_TYPE_POLICY_DECISION),
            (RejectionEvent, EVENT_TYPE_REJECTION),
            (BudgetEvent, EVENT_TYPE_BUDGET),
            (DimensionSampledEvent, EVENT_TYPE_DIMENSION_SAMPLED),
            (MutationAppliedEvent, EVENT_TYPE_MUTATION_APPLIED),
            (CandidateEvaluatedEvent, EVENT_TYPE_CANDIDATE_EVALUATED),
        ],
    )
    def test_event_create_serialize_deserialize(
        self,
        event_cls: type[TraceEvent],
        event_type_literal: str,
    ) -> None:
        data = {"key": "value", "count": 42}
        event = event_cls.create(ts="2026-01-15T10:30:00Z", data=data)

        assert event.event_type == event_type_literal
        assert event.ts == "2026-01-15T10:30:00Z"
        assert event.data == data

        serialized = event.model_dump()
        deserialized = event_cls.model_validate(serialized)

        assert deserialized.event_type == event_type_literal
        assert deserialized.ts == event.ts
        assert deserialized.data == event.data

    def test_event_type_model_map_covers_all_types(self) -> None:
        all_literals = [
            EVENT_TYPE_MODEL_MESSAGE,
            EVENT_TYPE_TOOL_CALL,
            EVENT_TYPE_TOOL_RESULT,
            EVENT_TYPE_VERIFICATION,
            EVENT_TYPE_TODO,
            EVENT_TYPE_DIFF,
            EVENT_TYPE_ARTIFACT,
            EVENT_TYPE_POLICY_DECISION,
            EVENT_TYPE_REJECTION,
            EVENT_TYPE_BUDGET,
            EVENT_TYPE_DIMENSION_SAMPLED,
            EVENT_TYPE_MUTATION_APPLIED,
            EVENT_TYPE_CANDIDATE_EVALUATED,
        ]
        for literal in all_literals:
            assert literal in EVENT_TYPE_MODEL_MAP, f"Missing from map: {literal}"

    def test_iter_trace_jsonl_unknown_event_type_falls_back_to_base(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "trace.jsonl"
        payload = {
            "schema_version": 1,
            "event_type": "UnknownFutureEvent",
            "ts": "2026-01-15T10:30:00Z",
            "data": {"future": True},
        }
        jsonl_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

        events = list(iter_trace_jsonl(jsonl_path))
        assert len(events) == 1
        assert events[0].event_type == "UnknownFutureEvent"
        assert events[0].data == {"future": True}

    def test_jsonl_write_read_round_trip(self, tmp_path: Path) -> None:
        events = [
            DimensionSampledEvent.create(
                ts="2026-01-15T10:00:00Z",
                data={"experiment": 0, "dimensions": ["context_strategy"]},
            ),
            MutationAppliedEvent.create(
                ts="2026-01-15T10:00:01Z",
                data={"experiment": 0, "configuration": {"agent": "explore"}},
            ),
            CandidateEvaluatedEvent.create(
                ts="2026-01-15T10:00:02Z",
                data={"experiment": 0, "score": 0.85},
            ),
        ]

        jsonl_path = tmp_path / "roundtrip.jsonl"
        write_trace_jsonl(jsonl_path, events)

        loaded = list(iter_trace_jsonl(jsonl_path))
        assert len(loaded) == 3
        assert isinstance(loaded[0], DimensionSampledEvent)
        assert isinstance(loaded[1], MutationAppliedEvent)
        assert isinstance(loaded[2], CandidateEvaluatedEvent)
        for original, restored in zip(events, loaded):
            assert original.event_type == restored.event_type
            assert original.ts == restored.ts
            assert original.data == restored.data

    def test_append_trace_jsonl_appends_to_existing(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "append.jsonl"

        event1 = DimensionSampledEvent.create(ts="2026-01-15T10:00:00Z", data={"experiment": 0})
        event2 = MutationAppliedEvent.create(ts="2026-01-15T10:00:01Z", data={"experiment": 1})

        append_trace_jsonl(jsonl_path, event1)
        append_trace_jsonl(jsonl_path, event2)

        loaded = list(iter_trace_jsonl(jsonl_path))
        assert len(loaded) == 2
        assert loaded[0].data == {"experiment": 0}
        assert loaded[1].data == {"experiment": 1}

    def test_iter_trace_jsonl_empty_file_returns_nothing(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("", encoding="utf-8")

        events = list(iter_trace_jsonl(jsonl_path))
        assert events == []

    def test_iter_trace_jsonl_nonexistent_file_returns_nothing(self, tmp_path: Path) -> None:
        events = list(iter_trace_jsonl(tmp_path / "nope.jsonl"))
        assert events == []

    def test_base_trace_event_forbid_extra_fields(self) -> None:
        import pydantic as pd

        with pytest.raises(pd.ValidationError):
            TraceEvent(
                ts="2026-01-15T10:00:00Z",
                data={},
                unexpected_field="should_fail",  # type: ignore[call-arg]
            )
