"""Tests for NavigationMonitorRubric telemetry metrics."""

import pytest

from verifiers_interact.telemetry import NavigationMonitorRubric


@pytest.fixture
def rubric():
    return NavigationMonitorRubric()


@pytest.fixture
def state_with_data():
    """Simulate a state dict after some navigation steps."""
    return {
        "nav_truncations": 5,
        "nav_lines_hidden": 250,
        "nav_chars_hidden": 10000,
        "nav_total_tool_outputs": 20,
        "nav_constraint_type": "LineLimit",
    }


@pytest.fixture
def empty_state():
    """State before any navigation has occurred."""
    return {}


class TestNavigationMonitorRubric:
    @pytest.mark.asyncio
    async def test_truncation_count(self, rubric, state_with_data):
        result = await rubric.nav_truncation_count(state_with_data)
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_lines_hidden(self, rubric, state_with_data):
        result = await rubric.nav_lines_hidden(state_with_data)
        assert result == 250.0

    @pytest.mark.asyncio
    async def test_chars_hidden(self, rubric, state_with_data):
        result = await rubric.nav_chars_hidden(state_with_data)
        assert result == 10000.0

    @pytest.mark.asyncio
    async def test_tool_output_count(self, rubric, state_with_data):
        result = await rubric.nav_tool_output_count(state_with_data)
        assert result == 20.0

    @pytest.mark.asyncio
    async def test_truncation_rate(self, rubric, state_with_data):
        result = await rubric.nav_truncation_rate(state_with_data)
        assert result == pytest.approx(0.25)  # 5/20

    @pytest.mark.asyncio
    async def test_truncation_rate_zero_outputs(self, rubric, empty_state):
        """Edge case: no outputs yet → rate should be 0.0, not ZeroDivisionError."""
        result = await rubric.nav_truncation_rate(empty_state)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_defaults_on_empty_state(self, rubric, empty_state):
        """All metrics return 0 on a fresh state."""
        assert await rubric.nav_truncation_count(empty_state) == 0.0
        assert await rubric.nav_lines_hidden(empty_state) == 0.0
        assert await rubric.nav_chars_hidden(empty_state) == 0.0
        assert await rubric.nav_tool_output_count(empty_state) == 0.0
