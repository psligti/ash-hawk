from __future__ import annotations

from ash_hawk.improve.diagnose import Diagnosis
from ash_hawk.improve.failure_clustering import cluster_diagnoses


def test_cluster_diagnoses_groups_by_target_files() -> None:
    diagnoses = [
        Diagnosis(
            trial_id="trial-1",
            failure_summary="auth failure",
            root_cause="auth issue",
            suggested_fix="fix auth",
            target_files=["auth.py"],
            confidence=0.7,
        ),
        Diagnosis(
            trial_id="trial-2",
            failure_summary="another auth failure",
            root_cause="same auth layer",
            suggested_fix="fix auth carefully",
            target_files=["auth.py"],
            confidence=0.9,
        ),
        Diagnosis(
            trial_id="trial-3",
            failure_summary="db failure",
            root_cause="db issue",
            suggested_fix="fix db",
            target_files=["db.py"],
            confidence=0.8,
        ),
    ]

    clusters = cluster_diagnoses(diagnoses)

    assert len(clusters) == 2
    assert clusters[0].representative.trial_id == "trial-2"
    assert sorted(clusters[0].trial_ids) == ["trial-1", "trial-2"]
    assert clusters[0].representative.cluster_id is not None
    assert clusters[0].family == "unknown"


def test_cluster_diagnoses_groups_by_family_and_target_files() -> None:
    diagnoses = [
        Diagnosis(
            trial_id="trial-1",
            family="tool_loader",
            failure_summary="loader issue",
            root_cause="tool loader returned empty list",
            suggested_fix="fix loader",
            target_files=["tools/loader.py"],
            confidence=0.9,
        ),
        Diagnosis(
            trial_id="trial-2",
            family="tool_loader",
            failure_summary="tool registration issue",
            root_cause="registered zero tools",
            suggested_fix="fix registration",
            target_files=["tools/loader.py"],
            confidence=0.8,
        ),
        Diagnosis(
            trial_id="trial-3",
            family="tool_use_enforcement",
            failure_summary="prompt issue",
            root_cause="zero tool calls",
            suggested_fix="tighten prompt",
            target_files=["prompts/coding.md"],
            confidence=0.7,
        ),
    ]

    clusters = cluster_diagnoses(diagnoses)

    assert len(clusters) == 2
    assert clusters[0].family == "tool_loader"
