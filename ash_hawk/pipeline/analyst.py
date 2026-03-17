from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ash_hawk.contracts import ReviewFinding, ReviewMetrics, RunArtifact
from ash_hawk.strategies import Strategy, SubStrategy, get_sub_strategies

if TYPE_CHECKING:
    from ash_hawk.pipeline.translator import TranslatorOutput


@dataclass
class AnalystInput:
    artifact: RunArtifact | None = None
    focus_areas: list[str] | None = None
    translator_output: TranslatorOutput | None = None


@dataclass
class AnalystOutput:
    findings: list[ReviewFinding] = field(default_factory=list)
    metrics: ReviewMetrics = field(default_factory=lambda: ReviewMetrics(score=0.0))
    tool_efficiency: dict[str, float] = field(default_factory=dict)
    failure_patterns: list[str] = field(default_factory=list)
    risk_areas: list[str] = field(default_factory=list)
    root_causes: list[str] = field(default_factory=list)
    strategy_insights: dict[Strategy, list[str]] = field(default_factory=dict)


class AnalystRole:
    def analyze(self, input_data: AnalystInput) -> AnalystOutput:
        output = AnalystOutput()
        artifact = input_data.artifact
        if artifact is None:
            return output

        findings: list[ReviewFinding] = []
        metrics = self._calculate_metrics(artifact)
        tool_efficiency = self._calculate_tool_efficiency(artifact)
        failure_patterns = self._identify_failure_patterns(artifact)
        risk_areas = self._identify_risk_areas(artifact)

        root_causes = self._identify_root_causes(artifact, failure_patterns)
        strategy_insights = self._derive_strategy_insights(artifact, input_data.translator_output)

        findings.extend(
            self._generate_findings(
                artifact,
                failure_patterns,
                risk_areas,
                root_causes,
                strategy_insights,
            )
        )

        output.findings = findings
        output.metrics = metrics
        output.tool_efficiency = tool_efficiency
        output.failure_patterns = failure_patterns
        output.risk_areas = risk_areas
        output.root_causes = root_causes
        output.strategy_insights = strategy_insights
        return output

    def _identify_root_causes(
        self,
        artifact: RunArtifact,
        failure_patterns: list[str],
    ) -> list[str]:
        root_causes: list[str] = []

        timeout_count = sum(
            1
            for tc in artifact.tool_calls
            if tc.error_message and "timeout" in tc.error_message.lower()
        )
        if timeout_count > 0:
            root_causes.append(
                f"Timeout issues detected ({timeout_count} occurrences) - "
                "consider adjusting tool timeouts or optimizing operations"
            )

        permission_failures = [
            tc.tool_name
            for tc in artifact.tool_calls
            if tc.error_message and "permission" in tc.error_message.lower()
        ]
        if permission_failures:
            root_causes.append(
                f"Permission issues with tools: {', '.join(set(permission_failures))}"
            )

        failed_tools = [tc.tool_name for tc in artifact.tool_calls if tc.outcome == "failure"]
        if failed_tools:
            tool_failure_counts: dict[str, int] = {}
            for tool in failed_tools:
                tool_failure_counts[tool] = tool_failure_counts.get(tool, 0) + 1
            for tool, count in tool_failure_counts.items():
                root_causes.append(f"Tool {tool} failed {count} time(s)")

        if artifact.outcome == "failure" and not failure_patterns:
            root_causes.append("Run failed without specific failure patterns")

        return root_causes

    def _derive_strategy_insights(
        self,
        artifact: RunArtifact,
        translator_output: TranslatorOutput | None,
    ) -> dict[Strategy, list[str]]:
        insights: dict[Strategy, list[str]] = {}

        if translator_output:
            for finding in translator_output.structured_findings:
                if finding.strategy_mapping:
                    strategy = finding.strategy_mapping.strategy
                    if strategy not in insights:
                        insights[strategy] = []
                    insights[strategy].append(finding.description)
        else:
            insights = self._infer_strategy_insights_from_artifact(artifact)

        return insights

    def _infer_strategy_insights_from_artifact(
        self,
        artifact: RunArtifact,
    ) -> dict[Strategy, list[str]]:
        insights: dict[Strategy, list[str]] = {}

        for tc in artifact.tool_calls:
            if tc.outcome == "failure":
                if Strategy.TOOL_QUALITY not in insights:
                    insights[Strategy.TOOL_QUALITY] = []
                error_msg = tc.error_message or "unknown error"
                insights[Strategy.TOOL_QUALITY].append(f"Tool {tc.tool_name} failure: {error_msg}")

        if artifact.outcome == "failure":
            if Strategy.AGENT_BEHAVIOR not in insights:
                insights[Strategy.AGENT_BEHAVIOR] = []
            insights[Strategy.AGENT_BEHAVIOR].append("Run did not complete successfully")

        return insights

    def _calculate_metrics(self, artifact: RunArtifact) -> ReviewMetrics:
        total_calls = len(artifact.tool_calls)
        successful_calls = sum(1 for tc in artifact.tool_calls if tc.outcome == "success")

        if not artifact.tool_calls:
            return ReviewMetrics(score=0.0)

        success_rate = successful_calls / total_calls if total_calls > 0 else 0.0
        total_duration = sum(tc.duration_ms or 0 for tc in artifact.tool_calls)
        avg_duration = total_duration / total_calls if total_calls > 0 else 0

        return ReviewMetrics(
            score=success_rate,
            efficiency_score=min(1.0, 1000 / avg_duration) if avg_duration > 0 else 1.0,
            quality_score=1.0 if artifact.outcome == "success" else 0.0,
        )

    def _calculate_tool_efficiency(self, artifact: RunArtifact) -> dict[str, float]:
        if not artifact.tool_calls:
            return {}

        tool_counts: dict[str, int] = {}
        for tc in artifact.tool_calls:
            tool_counts[tc.tool_name] = tool_counts.get(tc.tool_name, 0) + 1

        total = len(artifact.tool_calls)
        unique = len(tool_counts)
        redundant = total - unique

        durations = [tc.duration_ms for tc in artifact.tool_calls if tc.duration_ms]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "unique_tool_count": float(unique),
            "total_tool_count": float(total),
            "redundancy_ratio": redundant / total if total > 0 else 0.0,
            "avg_duration_ms": float(avg_duration),
        }

    def _identify_failure_patterns(self, artifact: RunArtifact) -> list[str]:
        patterns: list[str] = []

        for tc in artifact.tool_calls:
            if tc.outcome == "failure":
                patterns.append(f"Tool {tc.tool_name} failed")
            if tc.error_message:
                error_lower = tc.error_message.lower()
                if "timeout" in error_lower or "timed out" in error_lower:
                    patterns.append("Tool call timed out")
                elif "permission" in error_lower:
                    patterns.append("Permission denied")

        if artifact.outcome == "failure":
            patterns.append("Run failed to complete")

        return patterns

    def _identify_risk_areas(self, artifact: RunArtifact) -> list[str]:
        areas: set[str] = set()

        high_risk_tools = {"write", "edit", "delete", "execute", "bash", "shell"}
        for tc in artifact.tool_calls:
            if tc.tool_name in high_risk_tools:
                areas.add("file_modification")
            if tc.tool_name in {"bash", "shell", "execute"}:
                areas.add("command_execution")

        return list(areas)

    def _generate_findings(
        self,
        artifact: RunArtifact,
        failure_patterns: list[str],
        risk_areas: list[str],
        root_causes: list[str],
        strategy_insights: dict[Strategy, list[str]],
    ) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []

        if artifact.outcome == "failure":
            findings.append(
                ReviewFinding(
                    finding_id=f"finding-{uuid4().hex[:8]}",
                    category="outcome",
                    severity="critical",
                    title="Run failed to complete",
                    description=f"Run outcome was {artifact.outcome}",
                    evidence_refs=["run.outcome"],
                    recommendation="Review failure cause and error handling",
                )
            )

        for pattern in failure_patterns:
            findings.append(
                ReviewFinding(
                    finding_id=f"finding-{uuid4().hex[:8]}",
                    category="reliability",
                    severity="warning",
                    title="Failure pattern detected",
                    description=f"Pattern: {pattern}",
                    evidence_refs=["tool_calls"],
                )
            )

        for area in risk_areas:
            findings.append(
                ReviewFinding(
                    finding_id=f"finding-{uuid4().hex[:8]}",
                    category="risk",
                    severity="info",
                    title="Risk area identified",
                    description=f"Risk area: {area}",
                    evidence_refs=["steps"],
                )
            )

        for root_cause in root_causes:
            findings.append(
                ReviewFinding(
                    finding_id=f"finding-{uuid4().hex[:8]}",
                    category="root_cause",
                    severity="warning",
                    title="Root cause identified",
                    description=root_cause,
                    evidence_refs=["analysis"],
                    recommendation=self._get_recommendation_for_root_cause(root_cause),
                )
            )

        for strategy, insights in strategy_insights.items():
            for insight in insights:
                findings.append(
                    ReviewFinding(
                        finding_id=f"finding-{uuid4().hex[:8]}",
                        category="strategy",
                        severity="info",
                        title=f"Strategy insight: {strategy.value}",
                        description=insight,
                        evidence_refs=["strategy_analysis"],
                    )
                )

        return findings

    def _get_recommendation_for_root_cause(self, root_cause: str) -> str:
        if "timeout" in root_cause.lower():
            return "Increase tool timeouts or optimize slow operations"
        elif "permission" in root_cause.lower():
            return "Review and adjust tool access permissions"
        elif "failed" in root_cause.lower():
            return "Add error handling and retry logic for tool failures"
        else:
            return "Investigate and address the identified issue"


__all__ = ["AnalystInput", "AnalystOutput", "AnalystRole"]
