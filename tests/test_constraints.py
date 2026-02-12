"""Tests for observation constraints."""

import pytest

from verifiers_interact.constraints import (
    ConstraintResult,
    LineLimit,
    TokenBudget,
    Unconstrained,
)
from verifiers_interact.folders import TruncateFolder, StructureFolder


# ---------------------------------------------------------------------------
# LineLimit
# ---------------------------------------------------------------------------

class TestLineLimit:
    def test_short_output_passthrough(self):
        """Output under the limit passes through unchanged."""
        c = LineLimit(50)
        text = "\n".join(f"line {i}" for i in range(30))
        result = c.apply(text)
        assert result.content == text
        assert result.was_truncated is False
        assert result.metadata["lines_hidden"] == 0
        assert result.metadata["total_lines"] == 30

    def test_exact_limit_passthrough(self):
        """Output at exactly the limit passes through."""
        c = LineLimit(10)
        text = "\n".join(f"line {i}" for i in range(10))
        result = c.apply(text)
        assert result.was_truncated is False

    def test_truncation(self):
        """Output over the limit is truncated with a notice."""
        c = LineLimit(50)
        lines = [f"line {i}" for i in range(100)]
        text = "\n".join(lines)
        result = c.apply(text)

        assert result.was_truncated is True
        assert result.metadata["lines_hidden"] == 50
        assert result.metadata["lines_shown"] == 50
        assert result.metadata["total_lines"] == 100
        assert "TRUNCATED" in result.content
        # First 50 lines should be present
        assert "line 0" in result.content
        assert "line 49" in result.content

    def test_single_line_limit(self):
        """Edge case: limit of 1 line."""
        c = LineLimit(1)
        result = c.apply("first\nsecond\nthird")
        assert result.was_truncated is True
        assert result.metadata["lines_hidden"] == 2
        assert result.content.startswith("first")

    def test_empty_string(self):
        """Empty input passes through."""
        c = LineLimit(10)
        result = c.apply("")
        assert result.was_truncated is False
        assert result.content == ""

    def test_invalid_max_lines(self):
        """max_lines < 1 raises ValueError."""
        with pytest.raises(ValueError):
            LineLimit(0)
        with pytest.raises(ValueError):
            LineLimit(-5)

    def test_repr_includes_folder(self):
        assert "TruncateFolder" in repr(LineLimit(42))
        assert "StructureFolder" in repr(LineLimit(42, folder=StructureFolder()))

    def test_custom_folder(self):
        """LineLimit delegates to custom folder when truncation triggers."""
        c = LineLimit(10, folder=StructureFolder())
        text = "\n".join(f"line {i}" for i in range(50))
        result = c.apply(text)
        assert result.was_truncated is True
        assert result.metadata["folder"] == "StructureFolder"


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_short_output_passthrough(self):
        """Output under the budget passes through unchanged."""
        c = TokenBudget(1000)
        text = "a" * 500
        result = c.apply(text)
        assert result.content == text
        assert result.was_truncated is False

    def test_exact_budget_passthrough(self):
        """Output at exactly the budget passes through."""
        c = TokenBudget(100)
        text = "x" * 100
        result = c.apply(text)
        assert result.was_truncated is False

    def test_truncation(self):
        """Output over the budget is truncated."""
        c = TokenBudget(100)
        text = "x" * 200
        result = c.apply(text)
        assert result.was_truncated is True
        assert result.metadata["total_chars"] == 200

    def test_empty_string(self):
        """Empty input passes through."""
        c = TokenBudget(100)
        result = c.apply("")
        assert result.was_truncated is False

    def test_invalid_max_chars(self):
        with pytest.raises(ValueError):
            TokenBudget(0)

    def test_repr_includes_folder(self):
        assert "TruncateFolder" in repr(TokenBudget(4000))


# ---------------------------------------------------------------------------
# Unconstrained
# ---------------------------------------------------------------------------

class TestUnconstrained:
    def test_passthrough(self):
        """Everything passes through unchanged."""
        c = Unconstrained()
        text = "x" * 100000
        result = c.apply(text)
        assert result.content == text
        assert result.was_truncated is False

    def test_metadata(self):
        c = Unconstrained()
        result = c.apply("hello")
        assert result.metadata["total_chars"] == 5

    def test_repr(self):
        assert repr(Unconstrained()) == "Unconstrained()"


# ---------------------------------------------------------------------------
# ConstraintResult
# ---------------------------------------------------------------------------

class TestConstraintResult:
    def test_dataclass_defaults(self):
        r = ConstraintResult(content="test", was_truncated=False)
        assert r.metadata == {}

    def test_dataclass_with_metadata(self):
        r = ConstraintResult(
            content="test",
            was_truncated=True,
            metadata={"lines_hidden": 50},
        )
        assert r.metadata["lines_hidden"] == 50
