"""Auto-research module for iterative improvement cycles."""

from __future__ import annotations

from ash_hawk.auto_research.cycle_runner import run_cycle
from ash_hawk.auto_research.types import (
    CycleResult,
    CycleStatus,
    IterationResult,
)

__all__ = [
    "CycleResult",
    "CycleStatus",
    "IterationResult",
    "run_cycle",
]
