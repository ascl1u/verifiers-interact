"""
Predefined experiment profiles for navigation ablation studies.

Each profile returns a dict of kwargs that can be unpacked directly
into NavigationEnv. This makes it trivial to run controlled experiments:

    from verifiers_interact import NavigationEnv, ToolProfile

    # Tight constraint with structural folding — maximum search pressure
    env = NavigationEnv(**ToolProfile.minimal(), dataset=ds, rubric=r)

    # Baseline — no constraint
    env = NavigationEnv(**ToolProfile.unconstrained(), dataset=ds, rubric=r)
"""

from __future__ import annotations

from typing import Any

from .constraints import LineLimit, TokenBudget, Unconstrained
from .folders import HeadTailFolder, StructureFolder, TruncateFolder


class ToolProfile:
    """Factory for predefined NavigationEnv configurations.

    Each method returns a dict of kwargs suitable for NavigationEnv.__init__.
    Profiles are designed for factorial ablation studies:
    - minimal vs standard vs power → vary the budget
    - Change the folder on any profile → vary the compression strategy

    Usage:
        profile = ToolProfile.standard()
        env = NavigationEnv(**profile, dataset=ds, rubric=rubric)
    """

    @staticmethod
    def minimal() -> dict[str, Any]:
        """Maximum search pressure.

        50-line window with structural folding and 100 iterations.
        The model sees only function signatures and class definitions.
        Forces it to learn precise, targeted navigation queries.
        """
        return {
            "constraint": LineLimit(50, folder=StructureFolder()),
            "max_iterations": 100,
            "max_output_length": 2048,
        }

    @staticmethod
    def standard() -> dict[str, Any]:
        """Balanced default for training.

        200-line window with head truncation and 50 iterations.
        Good balance between context pressure and task feasibility.
        """
        return {
            "constraint": LineLimit(200, folder=TruncateFolder()),
            "max_iterations": 50,
            "max_output_length": 8192,
        }

    @staticmethod
    def power() -> dict[str, Any]:
        """Generous context budget with head+tail folding.

        16K character budget (~4K tokens) with head/tail folding.
        The model sees the beginning and end of output.
        """
        return {
            "constraint": TokenBudget(16000, folder=HeadTailFolder(0.6)),
            "max_iterations": 30,
            "max_output_length": 16384,
        }

    @staticmethod
    def unconstrained() -> dict[str, Any]:
        """No constraint — pure control group.

        No truncation at all. Use as the baseline in ablation studies
        to measure the raw effect of adding observation constraints.
        """
        return {
            "constraint": Unconstrained(),
            "max_iterations": 50,
            "max_output_length": 8192,
        }
