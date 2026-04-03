#!/usr/bin/env python3
"""Run self-improvement cycle for python-bugfix skill."""

import asyncio
from pathlib import Path

from ash_hawk.auto_research.cycle_runner import run_cycle
from ash_hawk.improvement.fixture_splitter import FixtureSplitter
from ash_hawk.improvement.guardrails import GuardrailConfig


async def main():
    project_root = Path("/Users/parkersligting/develop/pt/ash-hawk")
    evals_dir = project_root / "evals" / "python-bugfix"

    fixture_dirs = sorted(
        [d for d in evals_dir.iterdir() if d.is_dir() and "-" in d.name and d.name[0].isdigit()]
    )

    fixtures = [d / "scenario.yaml" for d in fixture_dirs if (d / "scenario.yaml").exists()]

    if len(fixtures) < 5:
        print(f"Warning: Only {len(fixtures)} fixtures found, need at least 5")
        return

    print(f"Found {len(fixtures)} fixtures")

    splitter = FixtureSplitter(seed=42, train_ratio=0.7)
    split = splitter.split(fixtures)

    print(f"Training fixtures: {len(split.train)}")
    print(f"Held-out fixtures: {len(split.heldout)}")

    guardrail_config = GuardrailConfig(
        max_consecutive_holdout_drops=3,
        max_reverts=5,
        plateau_window=5,
        plateau_threshold=0.02,
    )

    print("Starting improvement cycle...")
    print(f"Training on: {[p.parent.name for p in split.train[:3]]}...")
    print(f"Held-out: {[p.parent.name for p in split.heldout]}...")

    result = await run_cycle(
        scenarios=split.train,
        iterations=10,
        threshold=0.02,
        storage_path=project_root / ".ash-hawk" / "improvement-cycle",
        project_root=project_root,
        heldout_scenarios=split.heldout,
        guardrail_config=guardrail_config,
        explicit_targets=[project_root / ".dawn-kestrel" / "skills" / "python-bugfix.md"],
    )

    print("\n=== Cycle Complete ===")
    print(f"Status: {result.status.value}")
    print(f"Iterations: {result.total_iterations}")
    print(f"Initial score: {result.initial_score:.3f}")
    print(f"Final score: {result.final_score:.3f}")
    print(f"Improvement: {result.improvement_delta:+.3f}")

    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
