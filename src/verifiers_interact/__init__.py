"""
verifiers-interact: Observation constraints and navigation telemetry for RLMEnv.

A plugin library that extends Prime Intellect's verifiers with configurable
observation constraints, enabling ablation studies on how context pressure
affects RLM navigation performance.
"""

from .constraints import (
    ObservationConstraint,
    LineLimit,
    TokenBudget,
    Unconstrained,
)
from .telemetry import NavigationMonitorRubric
from .profiles import ToolProfile


def __getattr__(name: str):
    """Lazy import for NavigationEnv — requires Linux (fcntl dependency)."""
    if name == "NavigationEnv":
        from .env import NavigationEnv

        return NavigationEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ObservationConstraint",
    "LineLimit",
    "TokenBudget",
    "Unconstrained",
    "NavigationEnv",
    "NavigationMonitorRubric",
    "ToolProfile",
]
