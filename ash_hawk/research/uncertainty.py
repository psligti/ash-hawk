from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ash_hawk.research.types import HypothesisStatus

if TYPE_CHECKING:
    from ash_hawk.research.diagnosis import DiagnosisReport

logger = logging.getLogger(__name__)


def _str_list_factory() -> list[str]:
    return []


@dataclass
class HypothesisState:
    hypothesis_id: str
    description: str
    confidence: float
    evidence_for: list[str] = field(default_factory=_str_list_factory)
    evidence_against: list[str] = field(default_factory=_str_list_factory)
    missing_measurements: list[str] = field(default_factory=_str_list_factory)
    sample_count: int = 0
    last_updated: str = ""
    status: str = HypothesisStatus.ACTIVE.value

    def is_resolved(self) -> bool:
        return self.status in (
            HypothesisStatus.CONFIRMED.value,
            HypothesisStatus.REFUTED.value,
            HypothesisStatus.UNRESOLVABLE.value,
        )


@dataclass
class InformationGap:
    gap_id: str
    description: str
    proposed_measurement: str
    expected_impact: float
    proposed_by: str = "diagnosis"
    status: str = "identified"


class UncertaintyModel:
    def __init__(
        self,
        storage_path: Path | None = None,
        max_hypotheses: int = 20,
        max_evidence: int = 50,
    ) -> None:
        self.hypotheses: dict[str, HypothesisState] = {}
        self.unresolved_questions: list[str] = []
        self.information_gaps: list[InformationGap] = []
        self._storage_path = storage_path
        self._max_hypotheses = max_hypotheses
        self._max_evidence = max_evidence

    @classmethod
    def load(
        cls,
        storage_path: Path,
        max_hypotheses: int = 20,
        max_evidence: int = 50,
    ) -> UncertaintyModel:
        primary_path = storage_path / "uncertainty.json"
        backup_path = storage_path / "uncertainty.backup.json"

        def _load_json(path: Path) -> dict[str, object] | None:
            if not path.exists():
                return None
            try:
                with open(path) as handle:
                    data: object = json.load(handle)
            except json.JSONDecodeError:
                logger.warning("Failed to parse uncertainty file: %s", path)
                return None
            except OSError:
                logger.warning("Failed to read uncertainty file: %s", path)
                return None
            if isinstance(data, dict):
                return cast(dict[str, object], data)
            return None

        payload = _load_json(primary_path)
        if payload is None:
            payload = _load_json(backup_path)
            if payload is None:
                logger.warning("Uncertainty data missing or corrupted; starting fresh")
                return cls(
                    storage_path=storage_path,
                    max_hypotheses=max_hypotheses,
                    max_evidence=max_evidence,
                )

        model = cls(
            storage_path=storage_path, max_hypotheses=max_hypotheses, max_evidence=max_evidence
        )
        hypotheses_payload = payload.get("hypotheses")
        if isinstance(hypotheses_payload, dict):
            hypotheses_map = cast(dict[str, object], hypotheses_payload)
            for key, value in hypotheses_map.items():
                if not isinstance(value, dict):
                    continue
                hypothesis = _hypothesis_from_payload(key, cast(dict[str, object], value))
                if hypothesis:
                    model.hypotheses[hypothesis.hypothesis_id] = hypothesis

        questions_payload = payload.get("unresolved_questions")
        if isinstance(questions_payload, list):
            questions: list[str] = []
            for item in cast(list[object], questions_payload):
                if isinstance(item, str):
                    questions.append(item)
            model.unresolved_questions = questions

        gaps_payload = payload.get("information_gaps")
        if isinstance(gaps_payload, list):
            for item in cast(list[object], gaps_payload):
                if not isinstance(item, dict):
                    continue
                gap = _gap_from_payload(cast(dict[str, object], item))
                if gap:
                    model.information_gaps.append(gap)

        return model

    async def save(self) -> None:
        if self._storage_path is None:
            return

        storage_path = self._storage_path
        primary_path = storage_path / "uncertainty.json"
        backup_path = storage_path / "uncertainty.backup.json"

        try:
            storage_path.mkdir(parents=True, exist_ok=True)

            if primary_path.exists():
                await asyncio.to_thread(_copy_file, primary_path, backup_path)

            for hypothesis in self.hypotheses.values():
                _cap_hypothesis_evidence(hypothesis, self._max_evidence)

            payload: dict[str, object] = {
                "hypotheses": {key: asdict(value) for key, value in self.hypotheses.items()},
                "unresolved_questions": list(self.unresolved_questions),
                "information_gaps": [asdict(gap) for gap in self.information_gaps],
            }

            await asyncio.to_thread(_write_json, primary_path, payload)
        except Exception:
            logger.exception("Failed to save uncertainty model")

    def update_from_diagnosis(self, report: DiagnosisReport | object) -> None:
        hypotheses = _get_attr_list(report, "hypotheses")
        for candidate in hypotheses:
            candidate_data = _candidate_to_dict(candidate)
            if not candidate_data:
                continue
            existing = self._find_matching_hypothesis(candidate_data)
            if existing is None:
                hypothesis = _build_hypothesis(candidate_data)
                self.hypotheses[hypothesis.hypothesis_id] = hypothesis
            else:
                _merge_candidate(existing, candidate_data)

        missing_signals = _get_attr_list(report, "missing_signals")
        for signal in missing_signals:
            if not isinstance(signal, str):
                continue
            if signal not in self.unresolved_questions:
                self.unresolved_questions.append(signal)
            self._ensure_gap_for_signal(signal)

        self._prune_hypotheses()

    def should_observe_before_fixing(self, threshold: float = 0.6) -> bool:
        return self.uncertainty_level > threshold

    @property
    def uncertainty_level(self) -> float:
        active = [h for h in self.hypotheses.values() if h.status == HypothesisStatus.ACTIVE.value]
        if not active:
            return 1.0
        confidences = [h.confidence for h in active]
        max_conf = max(confidences) if confidences else 0.0
        return 1.0 - max_conf

    def rank_missing_measurements(self) -> list[tuple[str, float]]:
        ranked = [
            (gap.gap_id, gap.expected_impact)
            for gap in self.information_gaps
            if gap.status == "identified"
        ]
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def get_active_hypotheses(self) -> list[HypothesisState]:
        return [h for h in self.hypotheses.values() if h.status == HypothesisStatus.ACTIVE.value]

    def get_resolved_hypotheses(self) -> list[HypothesisState]:
        return [h for h in self.hypotheses.values() if h.is_resolved()]

    def get_competing_hypotheses(self, min_count: int = 2) -> list[HypothesisState]:
        return [h for h in self.get_active_hypotheses() if h.confidence < 0.4]

    def _prune_hypotheses(self) -> None:
        active = [h for h in self.hypotheses.values() if h.status == HypothesisStatus.ACTIVE.value]
        if len(active) <= self._max_hypotheses:
            return
        active_sorted = sorted(active, key=lambda h: h.confidence)
        to_drop = active_sorted[: len(active) - self._max_hypotheses]
        for hypothesis in to_drop:
            self.hypotheses.pop(hypothesis.hypothesis_id, None)

    def _find_matching_hypothesis(self, candidate: dict[str, object]) -> HypothesisState | None:
        description = candidate.get("description")
        cause_category = candidate.get("cause_category")
        for hypothesis in self.hypotheses.values():
            if isinstance(description, str) and hypothesis.description == description:
                return hypothesis
            if isinstance(cause_category, str) and _matches_cause_category(
                hypothesis, cause_category
            ):
                return hypothesis
        return None

    def _ensure_gap_for_signal(self, signal: str) -> None:
        for gap in self.information_gaps:
            if gap.proposed_measurement == signal:
                return
        gap_id = f"gap_{len(self.information_gaps) + 1:03d}"
        gap = InformationGap(
            gap_id=gap_id,
            description=f"Missing measurement: {signal}",
            proposed_measurement=signal,
            expected_impact=0.5,
            proposed_by="diagnosis",
            status="identified",
        )
        self.information_gaps.append(gap)


def _copy_file(source: Path, destination: Path) -> None:
    destination.write_bytes(source.read_bytes())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _hypothesis_from_payload(
    hypothesis_id: str, payload: dict[str, object]
) -> HypothesisState | None:
    description = payload.get("description")
    confidence = payload.get("confidence")
    if not isinstance(description, str) or not isinstance(confidence, int | float):
        return None
    evidence_for = _ensure_str_list(payload.get("evidence_for"))
    evidence_against = _ensure_str_list(payload.get("evidence_against"))
    missing_measurements = _ensure_str_list(payload.get("missing_measurements"))
    sample_count = payload.get("sample_count")
    last_updated = payload.get("last_updated")
    status = payload.get("status")
    return HypothesisState(
        hypothesis_id=hypothesis_id,
        description=description,
        confidence=float(confidence),
        evidence_for=evidence_for,
        evidence_against=evidence_against,
        missing_measurements=missing_measurements,
        sample_count=int(sample_count) if isinstance(sample_count, int) else 0,
        last_updated=last_updated if isinstance(last_updated, str) else "",
        status=status if isinstance(status, str) else HypothesisStatus.ACTIVE.value,
    )


def _gap_from_payload(payload: dict[str, object]) -> InformationGap | None:
    gap_id = payload.get("gap_id")
    description = payload.get("description")
    proposed_measurement = payload.get("proposed_measurement")
    expected_impact = payload.get("expected_impact")
    if not isinstance(gap_id, str) or not isinstance(description, str):
        return None
    if not isinstance(proposed_measurement, str) or not isinstance(expected_impact, int | float):
        return None
    proposed_by = payload.get("proposed_by")
    status = payload.get("status")
    return InformationGap(
        gap_id=gap_id,
        description=description,
        proposed_measurement=proposed_measurement,
        expected_impact=float(expected_impact),
        proposed_by=proposed_by if isinstance(proposed_by, str) else "diagnosis",
        status=status if isinstance(status, str) else "identified",
    )


def _ensure_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in cast(list[object], value):
        if isinstance(item, str):
            items.append(item)
    return items


def _cap_hypothesis_evidence(hypothesis: HypothesisState, max_evidence: int) -> None:
    if max_evidence <= 0:
        return
    if len(hypothesis.evidence_for) > max_evidence:
        hypothesis.evidence_for = hypothesis.evidence_for[-max_evidence:]
    if len(hypothesis.evidence_against) > max_evidence:
        hypothesis.evidence_against = hypothesis.evidence_against[-max_evidence:]
    if len(hypothesis.missing_measurements) > max_evidence:
        hypothesis.missing_measurements = hypothesis.missing_measurements[-max_evidence:]


def _get_attr_list(report: object, name: str) -> list[object]:
    value = getattr(report, name, None)
    return cast(list[object], value) if isinstance(value, list) else []


def _candidate_to_dict(candidate: object) -> dict[str, object] | None:
    if isinstance(candidate, dict):
        return cast(dict[str, object], candidate)
    description = getattr(candidate, "description", None)
    confidence = getattr(candidate, "confidence", None)
    if not isinstance(description, str) or not isinstance(confidence, int | float):
        return None
    return {
        "hypothesis_id": getattr(candidate, "hypothesis_id", None),
        "description": description,
        "confidence": float(confidence),
        "evidence_for": getattr(candidate, "evidence_for", None),
        "evidence_against": getattr(candidate, "evidence_against", None),
        "missing_measurements": getattr(candidate, "missing_measurements", None),
        "cause_category": getattr(candidate, "cause_category", None),
    }


def _build_hypothesis(candidate: dict[str, object]) -> HypothesisState:
    hypothesis_id = candidate.get("hypothesis_id")
    if isinstance(hypothesis_id, str) and hypothesis_id:
        key = hypothesis_id
    else:
        key = f"hyp_{datetime.now(UTC).timestamp():.0f}"
    description = candidate.get("description")
    evidence_for = _ensure_str_list(candidate.get("evidence_for"))
    evidence_against = _ensure_str_list(candidate.get("evidence_against"))
    missing_measurements = _ensure_str_list(candidate.get("missing_measurements"))
    confidence = candidate.get("confidence")
    return HypothesisState(
        hypothesis_id=key,
        description=description if isinstance(description, str) else "",
        confidence=float(confidence) if isinstance(confidence, int | float) else 0.0,
        evidence_for=evidence_for,
        evidence_against=evidence_against,
        missing_measurements=missing_measurements,
        sample_count=1,
        last_updated=datetime.now(UTC).isoformat(),
        status=HypothesisStatus.ACTIVE.value,
    )


def _merge_candidate(existing: HypothesisState, candidate: dict[str, object]) -> None:
    confidence = candidate.get("confidence")
    if isinstance(confidence, int | float):
        total = existing.sample_count + 1
        existing.confidence = (
            existing.confidence * existing.sample_count + float(confidence)
        ) / total
        existing.sample_count = total

    existing.evidence_for.extend(_ensure_str_list(candidate.get("evidence_for")))
    existing.evidence_against.extend(_ensure_str_list(candidate.get("evidence_against")))
    existing.missing_measurements.extend(_ensure_str_list(candidate.get("missing_measurements")))
    existing.last_updated = datetime.now(UTC).isoformat()


def _matches_cause_category(hypothesis: HypothesisState, cause_category: str) -> bool:
    marker = cause_category.lower()
    return marker in hypothesis.hypothesis_id.lower() or marker in hypothesis.description.lower()
