"""Tests for NavigationEnv — unit tests using mocks (no real sandbox)."""

import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# RLMEnv depends on prime_tunnel which requires fcntl (Linux only)
pytest.importorskip("fcntl", reason="NavigationEnv requires Linux (fcntl)")

from verifiers_interact.constraints import LineLimit, Unconstrained, ConstraintResult
from verifiers_interact.env import NavigationEnv


class TestNavigationEnvEnvResponse:
    """Test env_response constraint application using mocked parent."""

    @pytest.mark.asyncio
    async def test_constraint_applied_to_tool_messages(self):
        """Constraint should modify tool message content."""
        constraint = LineLimit(5)

        # Create a mock env that doesn't need real sandbox
        with patch.object(NavigationEnv, "__init__", lambda self, **kw: None):
            env = NavigationEnv.__new__(NavigationEnv)
            env.constraint = constraint

        # Simulate raw tool messages from parent
        raw_messages = [
            {"role": "tool", "content": "\n".join(f"line {i}" for i in range(100)), "tool_call_id": "tc1"},
        ]

        # Mock super().env_response to return raw messages
        with patch(
            "verifiers_interact.env.RLMEnv.env_response",
            new_callable=AsyncMock,
            return_value=raw_messages,
        ):
            state = {
                "nav_truncations": 0,
                "nav_lines_hidden": 0,
                "nav_chars_hidden": 0,
                "nav_total_tool_outputs": 0,
            }
            result = await env.env_response([], state)

        # Content should be truncated
        assert "[OUTPUT TRUNCATED" in result[0]["content"]
        assert "line 0" in result[0]["content"]
        # State should be updated
        assert state["nav_truncations"] == 1
        assert state["nav_lines_hidden"] == 95
        assert state["nav_total_tool_outputs"] == 1

    @pytest.mark.asyncio
    async def test_unconstrained_passthrough(self):
        """Unconstrained should not modify content."""
        constraint = Unconstrained()

        with patch.object(NavigationEnv, "__init__", lambda self, **kw: None):
            env = NavigationEnv.__new__(NavigationEnv)
            env.constraint = constraint

        original_text = "hello world\nfoo bar"
        raw_messages = [
            {"role": "tool", "content": original_text, "tool_call_id": "tc1"},
        ]

        with patch(
            "verifiers_interact.env.RLMEnv.env_response",
            new_callable=AsyncMock,
            return_value=raw_messages,
        ):
            state = {
                "nav_truncations": 0,
                "nav_lines_hidden": 0,
                "nav_chars_hidden": 0,
                "nav_total_tool_outputs": 0,
            }
            result = await env.env_response([], state)

        assert result[0]["content"] == original_text
        assert state["nav_truncations"] == 0
        assert state["nav_total_tool_outputs"] == 1

    @pytest.mark.asyncio
    async def test_non_tool_messages_ignored(self):
        """Non-tool messages should pass through without constraint."""
        constraint = LineLimit(5)

        with patch.object(NavigationEnv, "__init__", lambda self, **kw: None):
            env = NavigationEnv.__new__(NavigationEnv)
            env.constraint = constraint

        raw_messages = [
            {"role": "assistant", "content": "some assistant text"},
        ]

        with patch(
            "verifiers_interact.env.RLMEnv.env_response",
            new_callable=AsyncMock,
            return_value=raw_messages,
        ):
            state = {"nav_truncations": 0, "nav_lines_hidden": 0, "nav_chars_hidden": 0, "nav_total_tool_outputs": 0}
            result = await env.env_response([], state)

        assert result[0]["content"] == "some assistant text"
        assert state["nav_total_tool_outputs"] == 0


class TestNavigationEnvTrajectory:
    """Test add_trajectory_step metadata injection."""

    @pytest.mark.asyncio
    async def test_nav_stats_injected(self):
        """Trajectory steps should get nav_stats in extras."""
        with patch.object(NavigationEnv, "__init__", lambda self, **kw: None):
            env = NavigationEnv.__new__(NavigationEnv)
            env.constraint = LineLimit(50)

        state = {
            "nav_constraint_type": "LineLimit",
            "nav_truncations": 3,
            "nav_lines_hidden": 150,
            "nav_chars_hidden": 0,
            "nav_total_tool_outputs": 10,
        }

        step = {"prompt": [], "completion": []}

        with patch(
            "verifiers_interact.env.RLMEnv.add_trajectory_step",
            new_callable=AsyncMock,
        ):
            await env.add_trajectory_step(state, step)

        assert "nav_stats" in step["extras"]
        stats = step["extras"]["nav_stats"]
        assert stats["constraint_type"] == "LineLimit"
        assert stats["truncations_so_far"] == 3
        assert stats["lines_hidden_so_far"] == 150
        assert stats["total_tool_outputs"] == 10
