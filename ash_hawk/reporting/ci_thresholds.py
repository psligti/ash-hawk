"""CI threshold enforcement for gap scorecard metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ash_hawk.reporting.gap_scorecard import GapScorecardGenerator, load_scorecard
from ash_hawk.reporting.scorecard_types import GapScorecard

DEFAULT_THRESHOLDS: dict[str, float] = {
    "overall_score": 0.70,
    "security_depth": 0.50,
    "delegation_quality": 0.50,
    "agent_depth_min": 0.40,
}


def check_thresholds(
    scorecard: GapScorecard,
    thresholds: dict[str, float],
) -> tuple[bool, list[str]]:
    """Check if scorecard meets all thresholds.

    Args:
        scorecard: GapScorecard to check
        thresholds: Dict of metric name to minimum threshold

    Returns:
        Tuple of (passed, list of failure messages)
    """
    failures = []

    # Check overall score
    if "overall_score" in thresholds:
        if scorecard.overall_score < thresholds["overall_score"]:
            failures.append(
                f"overall_score: {scorecard.overall_score:.2%} < {thresholds['overall_score']:.2%}"
            )

    # Check dimension scores
    for dim, threshold in thresholds.items():
        if dim.startswith("dimension_"):
            dim_name = dim.replace("dimension_", "")
            actual = scorecard.dimension_scores.get(dim_name, 0.0)
            if actual < threshold:
                failures.append(f"dimension_{dim_name}: {actual:.2%} < {threshold:.2%}")
        elif dim in scorecard.dimension_scores:
            actual = scorecard.dimension_scores[dim]
            if actual < threshold:
                failures.append(f"{dim}: {actual:.2%} < {threshold:.2%}")

    # Check agent depth minimum
    if "agent_depth_min" in thresholds:
        min_score = min(ad.score for ad in scorecard.agent_depth) if scorecard.agent_depth else 0.0
        if min_score < thresholds["agent_depth_min"]:
            failures.append(
                f"agent_depth_min: {min_score:.2%} < {thresholds['agent_depth_min']:.2%}"
            )

    # Check for regressions
    if "no_regressions" in thresholds and thresholds["no_regressions"]:
        pass  # Would need baseline comparison

    return len(failures) == 0, failures


def check_gap_thresholds(
    scorecard_path: Path,
    thresholds: dict[str, float],
    baseline_path: Path | None = None,
    fail_on_regression: bool = True,
) -> int:
    """Check gap thresholds and return exit code.

    Args:
        scorecard_path: Path to scorecard JSON
        thresholds: Dict of metric to threshold
        baseline_path: Optional baseline for regression check
        fail_on_regression: Whether to fail on regressions

    Returns:
        Exit code (0 = pass, 1 = fail)
    """
    scorecard = load_scorecard(scorecard_path)
    passed, failures = check_thresholds(scorecard, thresholds)

    # Check baseline regression if provided
    if baseline_path and baseline_path.exists():
        baseline = load_scorecard(baseline_path)
        generator = GapScorecardGenerator()
        diff = generator.compare_baseline(scorecard, baseline)

        if diff.overall_score_delta < 0:
            failures.append(f"overall_score regression: {diff.overall_score_delta:.2%}")
            passed = False

        if diff.regression_requirements and fail_on_regression:
            failures.append(f"requirement regressions: {', '.join(diff.regression_requirements)}")
            passed = False

    # Output results
    print(f"Gap Threshold Check: {'PASS' if passed else 'FAIL'}")
    print(f"  Overall: {scorecard.overall_score:.2%}")
    print(f"  Covered: {scorecard.covered_requirements}/{scorecard.total_requirements}")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  ❌ {f}")

    return 0 if passed else 1


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point for gap threshold checking."""
    parser = argparse.ArgumentParser(description="Check gap scorecard thresholds for CI")
    parser.add_argument(
        "scorecard",
        type=Path,
        help="Path to scorecard JSON file",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        metavar="NAME=VALUE",
        help="Threshold to check (e.g., overall_score=0.75). Can be repeated.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline scorecard for regression check",
    )
    parser.add_argument(
        "--no-fail-on-regression",
        action="store_true",
        help="Don't fail on regression, only on threshold",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "github"],
        default="text",
        help="Output format",
    )

    args = parser.parse_args(argv)

    # Parse thresholds
    thresholds = DEFAULT_THRESHOLDS.copy()
    if args.threshold:
        for t in args.threshold:
            name, value = t.split("=", 1)
            thresholds[name] = float(value)

    # Run check
    scorecard = load_scorecard(args.scorecard)
    passed, failures = check_thresholds(scorecard, thresholds)

    # Check baseline
    if args.baseline and args.baseline.exists():
        baseline = load_scorecard(args.baseline)
        generator = GapScorecardGenerator()
        diff = generator.compare_baseline(scorecard, baseline)

        if diff.overall_score_delta < 0:
            failures.append(f"overall_score regression: {diff.overall_score_delta:.2%}")
            passed = False

        if diff.regression_requirements and not args.no_fail_on_regression:
            failures.append(f"requirement regressions: {', '.join(diff.regression_requirements)}")
            passed = False

    # Output
    if args.output_format == "json":
        output = {
            "passed": passed,
            "overall_score": scorecard.overall_score,
            "failures": failures,
        }
        print(json.dumps(output, indent=2))
    elif args.output_format == "github":
        # GitHub Actions format
        if passed:
            print(f"::notice::Gap thresholds passed (overall: {scorecard.overall_score:.2%})")
        else:
            for f in failures:
                print(f"::error::{f}")
    else:
        print(f"Gap Threshold Check: {'PASS' if passed else 'FAIL'}")
        print(f"  Overall: {scorecard.overall_score:.2%}")
        print(f"  Covered: {scorecard.covered_requirements}/{scorecard.total_requirements}")
        if failures:
            print("\nFailures:")
            for f in failures:
                print(f"  ❌ {f}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(cli_main())
