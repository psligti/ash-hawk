from __future__ import annotations

from collections import defaultdict

import pydantic as pd

from ash_hawk.improve.diagnose import Diagnosis


class FailureCluster(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    cluster_id: str
    representative: Diagnosis
    diagnoses: list[Diagnosis]
    trial_ids: list[str]
    target_files: list[str]
    family: str


def _cluster_key(diagnosis: Diagnosis) -> tuple[str, ...]:
    if diagnosis.target_files:
        return (diagnosis.family, *tuple(sorted(diagnosis.target_files)))
    return (diagnosis.family, f"trial:{diagnosis.trial_id}")


def cluster_diagnoses(diagnoses: list[Diagnosis]) -> list[FailureCluster]:
    grouped: dict[tuple[str, ...], list[Diagnosis]] = defaultdict(list)
    for diagnosis in diagnoses:
        grouped[_cluster_key(diagnosis)].append(diagnosis)

    clusters: list[FailureCluster] = []
    for index, group in enumerate(grouped.values(), start=1):
        representative = max(group, key=lambda diagnosis: diagnosis.confidence)
        cluster_id = f"cluster-{index:03d}"
        representative = representative.model_copy(update={"cluster_id": cluster_id})
        diagnoses_with_cluster = [
            diagnosis.model_copy(update={"cluster_id": cluster_id}) for diagnosis in group
        ]
        target_files = sorted({path for diagnosis in group for path in diagnosis.target_files})
        trial_ids = sorted({diagnosis.trial_id for diagnosis in group})
        clusters.append(
            FailureCluster(
                cluster_id=cluster_id,
                representative=representative,
                diagnoses=diagnoses_with_cluster,
                trial_ids=trial_ids,
                target_files=target_files,
                family=representative.family,
            )
        )

    clusters.sort(key=lambda cluster: cluster.representative.confidence, reverse=True)
    return clusters


__all__ = ["FailureCluster", "cluster_diagnoses"]
