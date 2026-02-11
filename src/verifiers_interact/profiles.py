"""
Predefined experiment profiles for navigation ablation studies.

Each profile returns a dict of kwargs that can be unpacked directly
into NavigationEnv. This makes it trivial to run controlled experiments:

    from verifiers_interact import NavigationEnv, ToolProfile

    # Tight constraint — forces maximum search compute
    env = NavigationEnv(**ToolProfile.minimal(), dataset=ds, rubric=r)

    # Baseline — no constraint
    env = NavigationEnv(**ToolProfile.unconstrained(), dataset=ds, rubric=r)
"""

from __future__ import annotations

from typing import Any

from .constraints import LineLimit, TokenBudget, Unconstrained


class ToolProfile:
    """Factory for predefined NavigationEnv configurations.

    Each method returns a dict of kwargs suitable for NavigationEnv.__init__.
    The dicts contain `constraint`, `max_iterations`, and `max_output_length`.

    Usage:
        profile = ToolProfile.standard()
        env = NavigationEnv(**profile, dataset=ds, rubric=rubric)
    """

    @staticmethod
    def minimal() -> dict[str, Any]:
        """Tight constraint — forces maximum search compute.

        50-line observation window with 100 iterations.
        The agent must learn efficient navigation strategies.
        """
        return {
            "constraint": LineLimit(50),
            "max_iterations": 100,
            "max_output_length": 2048,
        }

    @staticmethod
    def standard() -> dict[str, Any]:
        """Balanced constraint — recommended default for training.

        200-line observation window with 50 iterations.
        Good balance between context pressure and task feasibility.
        """
        return {
            "constraint": LineLimit(200),
            "max_iterations": 50,
            "max_output_length": 8192,
        }

    @staticmethod
    def power() -> dict[str, Any]:
        """Loose constraint — generous context budget.

        16K character budget (~4K tokens) with 30 iterations.
        Less compute pressure, more context per step.
        """
        return {
            "constraint": TokenBudget(16000),
            "max_iterations": 30,
            "max_output_length": 16384,
        }

    @staticmethod
    def unconstrained() -> dict[str, Any]:
        """No constraint — pure baseline for comparison.

        No truncation at all. Use as the control group in ablation studies
        to measure the raw effect of adding observation constraints.
        """
        return {
            "constraint": Unconstrained(),
            "max_iterations": 50,
            "max_output_length": 8192,
        }
