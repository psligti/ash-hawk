"""Experiment management package."""

from .config import ExperimentConfig
from .conflict_resolver import ConflictResolver, ResolutionResult
from .registry import Experiment, ExperimentRegistry

__all__ = [
    "Experiment",
    "ExperimentConfig",
    "ExperimentRegistry",
    "ConflictResolver",
    "ResolutionResult",
]
