"""Tests for ToolProfile presets."""

from verifiers_interact.constraints import LineLimit, TokenBudget, Unconstrained
from verifiers_interact.folders import StructureFolder, TruncateFolder, HeadTailFolder
from verifiers_interact.profiles import ToolProfile


class TestToolProfile:
    def _assert_profile_shape(self, profile: dict):
        """Every profile must have these three keys."""
        assert "constraint" in profile
        assert "max_iterations" in profile
        assert "max_output_length" in profile
        assert isinstance(profile["max_iterations"], int)
        assert isinstance(profile["max_output_length"], int)

    def test_minimal(self):
        p = ToolProfile.minimal()
        self._assert_profile_shape(p)
        assert isinstance(p["constraint"], LineLimit)
        assert p["constraint"].max_lines == 50
        assert isinstance(p["constraint"].folder, StructureFolder)
        assert p["max_iterations"] == 100

    def test_standard(self):
        p = ToolProfile.standard()
        self._assert_profile_shape(p)
        assert isinstance(p["constraint"], LineLimit)
        assert p["constraint"].max_lines == 200
        assert isinstance(p["constraint"].folder, TruncateFolder)

    def test_power(self):
        p = ToolProfile.power()
        self._assert_profile_shape(p)
        assert isinstance(p["constraint"], TokenBudget)
        assert p["constraint"].max_chars == 16000
        assert isinstance(p["constraint"].folder, HeadTailFolder)

    def test_unconstrained(self):
        p = ToolProfile.unconstrained()
        self._assert_profile_shape(p)
        assert isinstance(p["constraint"], Unconstrained)

    def test_profiles_are_independent(self):
        """Each call returns a new dict (no shared state)."""
        a = ToolProfile.standard()
        b = ToolProfile.standard()
        assert a is not b
        assert a["constraint"] is not b["constraint"]
