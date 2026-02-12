"""
verifiers-interact: Context management as a search problem.

A plugin library that extends Prime Intellect's verifiers with observation
constraints and context folding — enabling controlled ablation studies on
how context pressure affects RLM navigation performance.
"""

from .constraints import (
    ConstraintResult,
    ObservationConstraint,
    LineLimit,
    TokenBudget,
    Unconstrained,
)
from .folders import (
    ContextFolder,
    TruncateFolder,
    HeadTailFolder,
    StructureFolder,
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
    # Constraints
    "ConstraintResult",
    "ObservationConstraint",
    "LineLimit",
    "TokenBudget",
    "Unconstrained",
    # Folders
    "ContextFolder",
    "TruncateFolder",
    "HeadTailFolder",
    "StructureFolder",
    # Environment
    "NavigationEnv",
    # Telemetry
    "NavigationMonitorRubric",
    # Profiles
    "ToolProfile",
]
