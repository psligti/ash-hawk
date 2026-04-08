from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrainHoldoutSplit:
    """Result of splitting fixtures into train and holdout sets."""

    train: list[Path]
    heldout: list[Path]
    seed: int
    train_ratio: float
    total: int


class FixtureSplitter:
    """Deterministic train/holdout splitter using MD5 hashing.

    Assigns each fixture to train or holdout based on the MD5 hash of its
    path string combined with the seed. This ensures reproducibility across runs.
    """

    def __init__(self, seed: int = 42, train_ratio: float = 0.7) -> None:
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(f"train_ratio must be between 0.0 and 1.0, got {train_ratio}")
        self._seed = seed
        self._train_ratio = train_ratio

    def split(self, fixtures: list[Path]) -> TrainHoldoutSplit:
        logger.info(
            "Splitting %d fixtures: seed=%d train_ratio=%.2f",
            len(fixtures),
            self._seed,
            self._train_ratio,
        )

        train: list[Path] = []
        heldout: list[Path] = []

        for fixture in fixtures:
            if self._hash_fixture(fixture) < self._train_ratio:
                train.append(fixture)
            else:
                heldout.append(fixture)

        logger.info("Split result: train=%d heldout=%d", len(train), len(heldout))

        return TrainHoldoutSplit(
            train=train,
            heldout=heldout,
            seed=self._seed,
            train_ratio=self._train_ratio,
            total=len(fixtures),
        )

    def _hash_fixture(self, path: Path) -> float:
        digest = hashlib.md5(
            (str(path) + str(self._seed)).encode(), usedforsecurity=False
        ).hexdigest()
        return int(digest[:8], 16) / 0x100000000
