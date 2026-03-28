from ash_hawk.scenario.trace_normalizer import normalize_eval_transcript
from ash_hawk.types import EvalTranscript


def test_normalize_eval_transcript_preserves_explicit_trace_events() -> None:
    transcript = EvalTranscript(
        trace_events=[
            {
                "schema_version": 1,
                "event_type": "PolicyDecisionEvent",
                "ts": "1970-01-01T00:00:00Z",
                "data": {"tool_name": "bash", "allowed": True},
            }
        ],
        messages=[{"role": "assistant", "content": "done"}],
        tool_calls=[],
        agent_response="done",
    )

    events = normalize_eval_transcript(transcript)

    assert any(event.event_type == "PolicyDecisionEvent" for event in events)
