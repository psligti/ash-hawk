"""Fixture splitting utilities for train/held-out partitioning.

This module provides tools for deterministically splitting evaluation fixtures
into training and held-out sets for self-improvement loops.

The split is seeded for reproducibility and maintains a configurable ratio.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FixtureSplit:
    """Result of splitting fixtures into train and held-out sets.

    Attributes:
        train: List of training fixture paths.
        heldout: List of held-out fixture paths (used ONLY for success metric).
        seed: Random seed used for the split.
        ratio: Fraction of fixtures allocated to training.
    """

    train: list[Path]
    heldout: list[Path]
    seed: int
    ratio: float

    @property
    def total(self) -> int:
        """Total number of fixtures."""
        return len(self.train) + len(self.heldout)

    @property
    def train_count(self) -> int:
        """Number of training fixtures."""
        return len(self.train)

    @property
    def heldout_count(self) -> int:
        """Number of held-out fixtures."""
        return len(self.heldout)


class FixtureSplitter:
    """Splits fixtures into training and held-out sets.

    The split is deterministic for a given seed, ensuring reproducibility
    across improvement cycles.

    Example:
        >>> splitter = FixtureSplitter(seed=42, train_ratio=0.7)
        >>> fixtures = [Path("fixtures/01"), Path("fixtures/02"), ...]
        >>> split = splitter.split(fixtures)
        >>> len(split.train)  # 70% of fixtures
        >>> len(split.heldout)  # 30% of fixtures
    """

    def __init__(self, seed: int = 42, train_ratio: float = 0.7) -> None:
        """Initialize the splitter.

        Args:
            seed: Random seed for reproducibility.
            train_ratio: Fraction of fixtures to allocate to training (0.0-1.0).

        Raises:
            ValueError: If train_ratio is not between 0 and 1.
        """
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(f"train_ratio must be between 0 and 1 (exclusive), got {train_ratio}")

        self._seed = seed
        self._train_ratio = train_ratio

    @property
    def seed(self) -> int:
        """Random seed used for splitting."""
        return self._seed

    @property
    def train_ratio(self) -> float:
        """Fraction of fixtures allocated to training."""
        return self._train_ratio

    def split(self, fixtures: list[Path]) -> FixtureSplit:
        """Split fixtures into training and held-out sets.

        The split is deterministic for a given seed. Fixtures are shuffled
        and then partitioned according to the train_ratio.

        Args:
            fixtures: List of fixture paths to split.

        Returns:
            FixtureSplit with train and heldout lists.

        Note:
            If fewer than 5 fixtures are provided, all go to training.
            This prevents overfitting with tiny datasets.
        """
        if not fixtures:
            return FixtureSplit(train=[], heldout=[], seed=self._seed, ratio=self._train_ratio)

        # Edge case: small fixture counts go entirely to training
        if len(fixtures) < 5:
            return FixtureSplit(
                train=list(fixtures),
                heldout=[],
                seed=self._seed,
                ratio=self._train_ratio,
            )

        # Shuffle deterministically
        shuffled = list(fixtures)
        random.Random(self._seed).shuffle(shuffled)

        # Calculate split point
        split_index = int(len(shuffled) * self._train_ratio)

        # Ensure at least 1 fixture in heldout
        split_index = min(split_index, len(shuffled) - 1)

        return FixtureSplit(
            train=shuffled[:split_index],
            heldout=shuffled[split_index:],
            seed=self._seed,
            ratio=self._train_ratio,
        )

    def split_with_counts(self, fixtures: list[Path], train_count: int) -> FixtureSplit:
        """Split fixtures with a specific number of training fixtures.

        Alternative to split() that uses exact counts instead of ratio.

        Args:
            fixtures: List of fixture paths to split.
            train_count: Exact number of fixtures for training.

        Returns:
            FixtureSplit with train and heldout lists.

        Raises:
            ValueError: If train_count is invalid.
        """
        if not fixtures:
            return FixtureSplit(train=[], heldout=[], seed=self._seed, ratio=0.0)

        if train_count < 0:
            raise ValueError(f"train_count must be non-negative, got {train_count}")

        if train_count >= len(fixtures):
            # All to training
            return FixtureSplit(
                train=list(fixtures),
                heldout=[],
                seed=self._seed,
                ratio=1.0,
            )

        # Shuffle deterministically
        shuffled = list(fixtures)
        random.Random(self._seed).shuffle(shuffled)

        ratio = train_count / len(fixtures)

        return FixtureSplit(
            train=shuffled[:train_count],
            heldout=shuffled[train_count:],
            seed=self._seed,
            ratio=ratio,
        )


__all__ = [
    "FixtureSplit",
    "FixtureSplitter",
]
