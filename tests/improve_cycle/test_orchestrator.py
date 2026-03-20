from __future__ import annotations

from datetime import UTC, datetime

from ash_hawk.improve_cycle.models import MetricValue, RunArtifactBundle
from ash_hawk.improve_cycle.orchestrator import ImproveCycleOrchestrator


def test_improve_cycle_orchestrator_runs_end_to_end() -> None:
    orchestrator = ImproveCycleOrchestrator()
    run_bundle = RunArtifactBundle(
        run_id="run-1",
        experiment_id="exp-1",
        agent_id="bolt-merlin",
        eval_pack_id="bolt-merlin-eval",
        scenario_ids=["s1", "s2"],
        timestamp=datetime.now(UTC).isoformat(),
        tool_traces=[{"tool": "read"}],
        metrics=[MetricValue(name="score", value=0.4)],
    )

    result = orchestrator.run_cycle(run_bundle)

    assert result.triage.primary_owner in {"coach", "architect", "both", "block"}
    assert len(result.experiment_plans) == len(result.curated_lessons)
    assert len(result.verification_reports) == len(result.change_sets)
    assert result.history.agent_id == "bolt-merlin"
