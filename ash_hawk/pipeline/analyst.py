"""Analyst role for failure analysis and finding generation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from ash_hawk.contracts import ReviewFinding, ReviewMetrics

if TYPE_CHECKING:
    from dawn_kestrel.contracts.run_artifact import RunArtifact


class AnalystInput:
    """Input for the analyst role."""

    artifact: RunArtifact | None = None
    focus_areas: list[str] | None = None


class AnalystOutput:
    """Output from the analyst role."""

    def __init__(self) -> None:
        self.findings: list[ReviewFinding] = []
        self.metrics: ReviewMetrics = ReviewMetrics(score=0.0)
        self.tool_efficiency: dict[str, float] = {}
        self.failure_patterns: list[str] = []
        self.risk_areas: list[str] = []


class AnalystRole:
    """Analyzes run artifacts for failures, patterns, and risks."""

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

        findings.extend(self._generate_findings(artifact, failure_patterns, risk_areas))

        output.findings = findings
        output.metrics = metrics
        output.tool_efficiency = tool_efficiency
        output.failure_patterns = failure_patterns
        output.risk_areas = risk_areas
        return output

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

        return findings
