from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Mapping, cast
from uuid import uuid4

import pydantic as pd

from ash_hawk.contracts import ImprovementProposal
from ash_hawk.strategies.registry import Strategy, SubStrategy

if TYPE_CHECKING:
    from ash_hawk.pipeline.types import PipelineContext


class CoachRole(pd.BaseModel):
    """Coach role for generating improvement proposals from failure patterns."""

    model_config = pd.ConfigDict(extra="forbid")

    def generate_proposals(
        self,
        context: PipelineContext,
        failure_patterns: list[str],
        analyst_signals: Mapping[str, object] | None = None,
    ) -> list[ImprovementProposal]:
        signals = self._collect_signals(failure_patterns, analyst_signals)
        if not signals:
            return []

        grouped: dict[Strategy, list[dict[str, str]]] = defaultdict(list)
        for signal in signals:
            strategy, _ = self._infer_strategy_from_pattern(signal["text"])
            grouped[strategy].append(signal)

        ranked_groups = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)
        proposals: list[ImprovementProposal] = []

        for strategy, strategy_signals in ranked_groups[:3]:
            evidence = self._select_representative_signals(strategy_signals, limit=4)
            if not evidence:
                continue

            sub_strategies = self._derive_sub_strategies_for_cluster(evidence)
            proposal_type = self._strategy_to_proposal_type(strategy)
            title = self._build_cluster_title(strategy, evidence)
            rationale = self._build_cluster_rationale(strategy, evidence)
            expected_benefit = self._build_expected_benefit(strategy, len(strategy_signals))
            confidence = self._confidence_from_support(len(strategy_signals), len(signals))

            proposals.append(
                ImprovementProposal(
                    proposal_id=f"coach-{uuid4().hex[:12]}",
                    origin_run_id=context.run_artifact_id,
                    target_agent=context.target_agent,
                    proposal_type=proposal_type,
                    title=title,
                    rationale=rationale,
                    evidence_refs=[f"{s['source']}:{s['text']}" for s in evidence],
                    expected_benefit=expected_benefit,
                    risk_level=self._risk_from_signal_mix(evidence),
                    status="pending",
                    created_at=datetime.now(UTC),
                    strategy=strategy,
                    sub_strategies=sub_strategies,
                    confidence=confidence,
                    experiment_id=context.experiment_id,
                    diff_payload={
                        "hypothesis": rationale,
                        "signal_count": len(strategy_signals),
                        "signal_sources": sorted({s["source"] for s in evidence}),
                        "focus_signals": [s["text"] for s in evidence],
                        "experiment_steps": self._build_experiment_steps(strategy, evidence),
                    },
                )
            )

        contrarian = self._build_contrarian_proposal(context, ranked_groups)
        if contrarian is not None:
            proposals.append(contrarian)

        return proposals

    def _collect_signals(
        self,
        failure_patterns: list[str],
        analyst_signals: Mapping[str, object] | None,
    ) -> list[dict[str, str]]:
        signals: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def _push(source: str, value: str) -> None:
            text = value.strip()
            if not text:
                return
            key = (source, text)
            if key in seen:
                return
            seen.add(key)
            signals.append({"source": source, "text": text})

        for pattern in failure_patterns:
            _push("analyst.failure_pattern", pattern)

        if not analyst_signals:
            return signals

        for root_cause in self._as_str_list(analyst_signals.get("root_causes")):
            _push("analyst.root_cause", root_cause)

        for area in self._as_str_list(analyst_signals.get("risk_areas")):
            _push("analyst.risk_area", area)

        findings_obj = analyst_signals.get("findings")
        if isinstance(findings_obj, list):
            for finding_obj in cast(list[object], findings_obj):
                if not isinstance(finding_obj, dict):
                    continue
                raw_finding = cast(dict[object, object], finding_obj)
                category_obj = raw_finding.get("category")
                description_obj = raw_finding.get("description")
                if isinstance(category_obj, str) and isinstance(description_obj, str):
                    _push(f"analyst.finding.{category_obj}", description_obj)

        return signals

    def _as_str_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for item in cast(list[object], value):
            if isinstance(item, str):
                items.append(item)
        return items

    def _select_representative_signals(
        self,
        signals: list[dict[str, str]],
        limit: int,
    ) -> list[dict[str, str]]:
        if not signals:
            return []

        scored = sorted(signals, key=lambda s: self._signal_priority(s["source"], s["text"]))
        selected: list[dict[str, str]] = []
        covered_sources: set[str] = set()

        for signal in scored:
            if signal["source"] in covered_sources and len(selected) >= 2:
                continue
            selected.append(signal)
            covered_sources.add(signal["source"])
            if len(selected) >= limit:
                return selected

        for signal in scored:
            if signal in selected:
                continue
            selected.append(signal)
            if len(selected) >= limit:
                break

        return selected

    def _signal_priority(self, source: str, text: str) -> tuple[int, int]:
        source_rank = {
            "analyst.root_cause": 0,
            "analyst.failure_pattern": 1,
            "analyst.finding.root_cause": 2,
            "analyst.finding.efficiency": 3,
            "analyst.risk_area": 4,
        }.get(source, 5)
        return (source_rank, -len(text))

    def _derive_sub_strategies_for_cluster(
        self,
        evidence: list[dict[str, str]],
    ) -> list[SubStrategy]:
        ordered: list[SubStrategy] = []
        for signal in evidence:
            _, sub_strategies = self._infer_strategy_from_pattern(signal["text"])
            for sub_strategy in sub_strategies:
                if sub_strategy not in ordered:
                    ordered.append(sub_strategy)

        if not ordered:
            return [SubStrategy.ERROR_RECOVERY]
        return ordered[:3]

    def _build_cluster_title(self, strategy: Strategy, evidence: list[dict[str, str]]) -> str:
        primary = evidence[0]["text"]
        suffix = primary
        return f"Coach cluster ({strategy.value}): {suffix}"

    def _build_cluster_rationale(self, strategy: Strategy, evidence: list[dict[str, str]]) -> str:
        excerpts = "; ".join(signal["text"] for signal in evidence[:3])
        return f"Multiple signals point to {strategy.value}. Representative evidence: {excerpts}"

    def _build_expected_benefit(self, strategy: Strategy, support: int) -> str:
        return (
            f"Reduce repeated failures linked to {strategy.value} by prioritizing a "
            f"high-support intervention ({support} aligned signals)."
        )

    def _confidence_from_support(self, support: int, total: int) -> float:
        if total <= 0:
            return 0.5
        ratio = support / total
        return min(0.95, max(0.45, 0.45 + ratio * 0.5))

    def _risk_from_signal_mix(
        self, evidence: list[dict[str, str]]
    ) -> Literal["low", "medium", "high"]:
        combined = " ".join(signal["text"].lower() for signal in evidence)
        if "permission" in combined or "critical" in combined:
            return "high"
        if "timeout" in combined or "failed" in combined or "error" in combined:
            return "medium"
        return "low"

    def _build_experiment_steps(
        self,
        strategy: Strategy,
        evidence: list[dict[str, str]],
    ) -> list[str]:
        focus = evidence[0]["text"] if evidence else strategy.value
        if strategy == Strategy.TOOL_QUALITY:
            return [
                "Add explicit fallback path for failing tool calls",
                "Introduce retry/timeout handling where failures cluster",
                f"Validate change against signal: {focus}",
            ]
        if strategy == Strategy.POLICY_QUALITY:
            return [
                "Strengthen policy guidance for failure-prone decision points",
                "Add anti-repetition constraints to avoid single-path behavior",
                f"Validate policy against signal: {focus}",
            ]
        if strategy == Strategy.SKILL_QUALITY:
            return [
                "Expand instruction examples to cover edge-case variants",
                "Add explicit branching guidance for ambiguous situations",
                f"Validate skill updates against signal: {focus}",
            ]
        return [
            "Create targeted remediation experiment",
            "Run A/B comparison on affected scenarios",
            f"Track whether this signal improves: {focus}",
        ]

    def _build_contrarian_proposal(
        self,
        context: PipelineContext,
        ranked_groups: list[tuple[Strategy, list[dict[str, str]]]],
    ) -> ImprovementProposal | None:
        if len(ranked_groups) < 2:
            return None

        strategy, evidence = ranked_groups[-1]
        selected = self._select_representative_signals(evidence, limit=3)
        if not selected:
            return None

        return ImprovementProposal(
            proposal_id=f"coach-{uuid4().hex[:12]}",
            origin_run_id=context.run_artifact_id,
            target_agent=context.target_agent,
            proposal_type=self._strategy_to_proposal_type(strategy),
            title=f"Coach contrarian probe ({strategy.value})",
            rationale=(
                "Primary proposal stream may be over-focused. "
                f"Probe lower-frequency signals for missed leverage in {strategy.value}."
            ),
            evidence_refs=[f"{s['source']}:{s['text']}" for s in selected],
            expected_benefit="Discover non-obvious improvements hidden by dominant failure patterns",
            risk_level="high",
            status="pending",
            created_at=datetime.now(UTC),
            strategy=strategy,
            sub_strategies=self._derive_sub_strategies_for_cluster(selected),
            confidence=0.55,
            experiment_id=context.experiment_id,
            diff_payload={
                "hypothesis": "Minority signals may unlock larger gains than dominant recurring failures",
                "signal_count": len(evidence),
                "focus_signals": [s["text"] for s in selected],
                "experiment_steps": [
                    "Apply one controlled change targeting minority signal cluster",
                    "Compare against baseline on same scenario set",
                    "Keep if score delta is positive over two iterations",
                ],
            },
        )

    def _strategy_to_proposal_type(
        self, strategy: Strategy
    ) -> Literal["policy", "skill", "tool", "harness", "eval"]:
        """Map strategy to proposal type."""
        if strategy == Strategy.TOOL_QUALITY:
            return "tool"
        elif strategy == Strategy.SKILL_QUALITY:
            return "skill"
        elif strategy == Strategy.POLICY_QUALITY:
            return "policy"
        elif strategy == Strategy.HARNESS_QUALITY:
            return "harness"
        else:
            return "tool"

    def _infer_strategy_from_pattern(self, pattern: str) -> tuple[Strategy, list[SubStrategy]]:
        """Infer strategy and sub-strategies from failure pattern."""
        pattern_lower = pattern.lower()

        if "timeout" in pattern_lower or "error" in pattern_lower:
            return Strategy.TOOL_QUALITY, [SubStrategy.ERROR_RECOVERY]
        elif "policy" in pattern_lower or "rule" in pattern_lower:
            return Strategy.POLICY_QUALITY, [SubStrategy.ENGAGEMENT_POLICY]
        elif "instruction" in pattern_lower or "clarity" in pattern_lower:
            return Strategy.SKILL_QUALITY, [SubStrategy.INSTRUCTION_CLARITY]
        else:
            # Default fallback
            return Strategy.TOOL_QUALITY, [SubStrategy.ERROR_RECOVERY]


__all__ = ["CoachRole"]
