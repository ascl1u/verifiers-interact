"""
Navigation telemetry rubric for NavigationEnv.

Exports constraint-related metrics to WandB and the eval dashboard,
following the same pattern as RLMMonitorRubric and SandboxMonitorRubric
in the verifiers library.

These metrics are automatically attached when using NavigationEnv.
"""

from __future__ import annotations

import verifiers as vf
from verifiers.types import State


class NavigationMonitorRubric(vf.Rubric):
    """Metrics rubric that tracks observation constraint behavior.

    Automatically added by NavigationEnv. Exports the following metrics:

    - nav_truncation_count: Total times tool output was truncated
    - nav_lines_hidden: Total lines hidden across all truncations
    - nav_chars_hidden: Total characters hidden across all truncations
    - nav_tool_output_count: Total tool outputs processed
    - nav_truncation_rate: Fraction of outputs that were truncated (0.0-1.0)
    - nav_constraint_type: Name of the active constraint class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.nav_truncation_count)
        self.add_metric(self.nav_lines_hidden)
        self.add_metric(self.nav_chars_hidden)
        self.add_metric(self.nav_tool_output_count)
        self.add_metric(self.nav_truncation_rate)

    async def nav_truncation_count(self, state: State) -> float:
        """Number of tool outputs that were truncated by the constraint."""
        return float(state.get("nav_truncations", 0))

    async def nav_lines_hidden(self, state: State) -> float:
        """Total lines hidden across all truncated outputs."""
        return float(state.get("nav_lines_hidden", 0))

    async def nav_chars_hidden(self, state: State) -> float:
        """Total characters hidden across all truncated outputs."""
        return float(state.get("nav_chars_hidden", 0))

    async def nav_tool_output_count(self, state: State) -> float:
        """Total number of tool outputs processed through the constraint."""
        return float(state.get("nav_total_tool_outputs", 0))

    async def nav_truncation_rate(self, state: State) -> float:
        """Fraction of tool outputs that were truncated (0.0 to 1.0)."""
        total = state.get("nav_total_tool_outputs", 0)
        if total == 0:
            return 0.0
        return state.get("nav_truncations", 0) / total
