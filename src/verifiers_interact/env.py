"""
NavigationEnv: An RLMEnv subclass that enforces observation constraints.

Sits in the verifiers lifecycle at the `env_response` level, post-processing
tool outputs from the Python REPL before the model sees them. This enforces
the "Bitter Lesson" by trading context window size for inference-time compute.

Usage:
    from verifiers_interact import NavigationEnv, LineLimit

    env = NavigationEnv(
        constraint=LineLimit(200),
        dataset=dataset,
        rubric=rubric,
    )
"""

from __future__ import annotations

import logging
from typing import Any

from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, State, TrajectoryStep

from .constraints import ObservationConstraint, Unconstrained
from .telemetry import NavigationMonitorRubric

logger = logging.getLogger(__name__)


class NavigationEnv(RLMEnv):
    """An RLMEnv that enforces strict observation constraints on tool output.

    Inherits all RLMEnv functionality (Python REPL, sandbox execution,
    sub-LLM interception) and adds a constraint layer that truncates
    tool output before the model sees it.

    This enables ablation studies: run identical tasks with different
    constraints (e.g. LineLimit(50) vs LineLimit(500)) and measure
    how context pressure affects navigation performance.

    Args:
        constraint: ObservationConstraint to apply to all tool outputs.
            Defaults to Unconstrained() (no truncation, baseline mode).
        rubric: Optional Rubric for reward functions. NavigationMonitorRubric
            is automatically added on top of any user-provided rubric.
        **kwargs: All other arguments forwarded to RLMEnv (dataset, sub_model,
            max_iterations, execution_backend, etc.)

    Example:
        >>> from verifiers_interact import NavigationEnv, LineLimit, ToolProfile
        >>> env = NavigationEnv(
        ...     constraint=LineLimit(200),
        ...     dataset=my_dataset,
        ...     rubric=my_rubric,
        ...     max_iterations=50,
        ... )
        >>> # Or use a preset profile:
        >>> profile = ToolProfile.standard()
        >>> env = NavigationEnv(**profile, dataset=my_dataset, rubric=my_rubric)
    """

    def __init__(
        self,
        constraint: ObservationConstraint | None = None,
        rubric: Rubric | None = None,
        **kwargs: Any,
    ):
        self.constraint = constraint or Unconstrained()
        super().__init__(rubric=rubric, **kwargs)
        # Attach navigation telemetry metrics
        self.add_rubric(NavigationMonitorRubric())
        logger.info("NavigationEnv initialized with constraint: %r", self.constraint)

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------

    async def setup_state(self, state: State, **kwargs: Any) -> State:
        """Initialize per-rollout navigation tracking on top of RLMEnv setup."""
        state = await super().setup_state(state, **kwargs)

        # Navigation telemetry counters
        state["nav_truncations"] = 0
        state["nav_lines_hidden"] = 0
        state["nav_chars_hidden"] = 0
        state["nav_total_tool_outputs"] = 0
        state["nav_constraint_type"] = type(self.constraint).__name__

        logger.debug(
            "Navigation state initialized (constraint=%r, rollout=%s)",
            self.constraint,
            state.get("rollout_id", "?"),
        )
        return state

    # ------------------------------------------------------------------
    # Core interception: constrain tool output
    # ------------------------------------------------------------------

    async def env_response(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> Messages:
        """Process tool calls via parent, then apply observation constraint.

        Flow:
        1. ToolEnv.env_response() executes tool calls → returns tool messages
        2. We post-process each tool message's content through the constraint
        3. Truncated content gets a notice appended; metadata goes to state
        """
        # Step 1: Get raw tool output from parent chain
        # (RLMEnv → SandboxEnv → StatefulToolEnv → ToolEnv.env_response)
        tool_messages = await super().env_response(messages, state, **kwargs)

        # Step 2: Apply constraint to each tool message
        for msg in tool_messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue

            result = self.constraint.apply(content)
            msg["content"] = result.content

            # Step 3: Update navigation telemetry
            state["nav_total_tool_outputs"] = (
                state.get("nav_total_tool_outputs", 0) + 1
            )
            if result.was_truncated:
                state["nav_truncations"] = state.get("nav_truncations", 0) + 1
                state["nav_lines_hidden"] = state.get("nav_lines_hidden", 0) + result.metadata.get("lines_hidden", 0)
                state["nav_chars_hidden"] = state.get("nav_chars_hidden", 0) + result.metadata.get("chars_hidden", 0)

        return tool_messages

    # ------------------------------------------------------------------
    # Trajectory enrichment
    # ------------------------------------------------------------------

    async def add_trajectory_step(
        self, state: State, trajectory_step: TrajectoryStep
    ) -> None:
        """Inject navigation stats into each trajectory step's extras.

        This data becomes part of the training trajectory, enabling
        downstream analysis of how the agent navigated under constraints.
        """
        extras = trajectory_step.setdefault("extras", {})
        extras["nav_stats"] = {
            "constraint_type": state.get("nav_constraint_type", "unknown"),
            "truncations_so_far": state.get("nav_truncations", 0),
            "lines_hidden_so_far": state.get("nav_lines_hidden", 0),
            "chars_hidden_so_far": state.get("nav_chars_hidden", 0),
            "total_tool_outputs": state.get("nav_total_tool_outputs", 0),
        }
        await super().add_trajectory_step(state, trajectory_step)
