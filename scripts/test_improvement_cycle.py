#!/usr/bin/env python3
"""Test self-improvement loop with python-bugfix fixtures."""

import asyncio
from pathlib import Path

from ash_hawk.auto_research.cycle_runner import run_cycle
from ash_hawk.improvement.fixture_splitter import FixtureSplitter
from ash_hawk.improvement.guardrails import GuardrailConfig


async def main():
    project_root = Path("/Users/parkersligting/develop/pt/ash-hawk")
    scenarios_dir = project_root / "evals" / "python-bugfix"

    scenario_files = sorted(scenarios_dir.glob("*/scenario.yaml"))

    if len(scenario_files) < 5:
        print(f"Warning: Only {len(scenario_files)} scenarios found")
        return

    print(f"Found {len(scenario_files)} scenarios:")
    for s in scenario_files:
        print(f"  - {s.parent.name}")

    splitter = FixtureSplitter(seed=42, train_ratio=0.7)
    split = splitter.split(scenario_files)

    print(f"\nTraining scenarios ({len(split.train)}):")
    for s in split.train:
        print(f"  - {s.parent.name}")

    print(f"\nHeld-out scenarios ({len(split.heldout)}):")
    for s in split.heldout:
        print(f"  - {s.parent.name}")

    guardrail_config = GuardrailConfig(
        max_consecutive_holdout_drops=3,
        max_reverts=5,
        plateau_window=5,
        plateau_threshold=0.02,
    )

    skill_path = project_root / ".dawn-kestrel" / "skills" / "python-bugfix.md"

    print(f"\nSkill file: {skill_path}")
    print(f"Skill exists: {skill_path.exists()}")

    print("\n" + "=" * 60)
    print("Starting improvement cycle (3 iterations for testing)")
    print("=" * 60)

    result = await run_cycle(
        scenarios=split.train,
        iterations=3,
        threshold=0.02,
        storage_path=project_root / ".ash-hawk" / "improvement-test",
        project_root=project_root,
        heldout_scenarios=split.heldout,
        guardrail_config=guardrail_config,
        explicit_targets=[skill_path],
    )

    print("\n" + "=" * 60)
    print("CYCLE COMPLETE")
    print("=" * 60)
    print(f"Status: {result.status.value}")
    print(f"Total iterations: {result.total_iterations}")
    print(
        f"Initial score: {result.initial_score:.3f}"
        if result.initial_score
        else "Initial score: N/A"
    )
    print(f"Final score: {result.final_score:.3f}" if result.final_score else "Final score: N/A")
    print(
        f"Improvement: {result.improvement_delta:+.3f}"
        if result.improvement_delta
        else "Improvement: N/A"
    )

    if result.error_message:
        print(f"Error: {result.error_message}")

    return result


if __name__ == "__main__":
    result = asyncio.run(main())
