"""
Observation constraints for controlling what the agent sees.

Each constraint takes raw tool output and returns a constrained version
plus metadata about what was hidden. This enables ablation studies on
how context pressure affects navigation performance.

Usage:
    constraint = LineLimit(200)
    constrained_text, meta = constraint.apply(raw_output)
    # meta = {"lines_hidden": 150, "was_truncated": True, "total_lines": 350}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConstraintResult:
    """Result of applying an observation constraint.

    Attributes:
        content: The (possibly truncated) output text.
        was_truncated: Whether the constraint actually truncated anything.
        metadata: Constraint-specific stats (lines_hidden, chars_hidden, etc.)
    """

    content: str
    was_truncated: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class ObservationConstraint(ABC):
    """Abstract base class for observation constraints.

    Subclass this to create custom constraints for ablation studies.
    The only method to implement is `apply()`.
    """

    @abstractmethod
    def apply(self, content: str) -> ConstraintResult:
        """Apply the constraint to raw tool output.

        Args:
            content: Raw output string from a tool call.

        Returns:
            ConstraintResult with (possibly truncated) content and metadata.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class LineLimit(ObservationConstraint):
    """Truncate tool output to a maximum number of lines.

    Lines beyond the limit are replaced with a notice telling the agent
    how many lines were hidden, encouraging it to use targeted queries.

    Args:
        max_lines: Maximum number of lines to keep. Default: 200.
    """

    def __init__(self, max_lines: int = 200):
        if max_lines < 1:
            raise ValueError(f"max_lines must be >= 1, got {max_lines}")
        self.max_lines = max_lines

    def apply(self, content: str) -> ConstraintResult:
        lines = content.split("\n")
        total = len(lines)

        if total <= self.max_lines:
            return ConstraintResult(
                content=content,
                was_truncated=False,
                metadata={"lines_shown": total, "lines_hidden": 0, "total_lines": total},
            )

        hidden = total - self.max_lines
        truncated = "\n".join(lines[: self.max_lines])
        truncated += (
            f"\n\n[OUTPUT TRUNCATED: {hidden} of {total} lines hidden. "
            f"Use targeted queries to navigate.]"
        )
        return ConstraintResult(
            content=truncated,
            was_truncated=True,
            metadata={
                "lines_shown": self.max_lines,
                "lines_hidden": hidden,
                "total_lines": total,
            },
        )

    def __repr__(self) -> str:
        return f"LineLimit(max_lines={self.max_lines})"


class TokenBudget(ObservationConstraint):
    """Truncate tool output to a character budget (proxy for tokens).

    Uses character count as a fast proxy for token count (~4 chars/token).
    For exact token budgeting, subclass and override with a real tokenizer.

    Args:
        max_chars: Maximum characters to keep. Default: 4000 (~1000 tokens).
    """

    def __init__(self, max_chars: int = 4000):
        if max_chars < 1:
            raise ValueError(f"max_chars must be >= 1, got {max_chars}")
        self.max_chars = max_chars

    def apply(self, content: str) -> ConstraintResult:
        total = len(content)

        if total <= self.max_chars:
            return ConstraintResult(
                content=content,
                was_truncated=False,
                metadata={"chars_shown": total, "chars_hidden": 0, "total_chars": total},
            )

        hidden = total - self.max_chars
        # Truncate at last newline before budget to avoid mid-line cuts
        truncated = content[: self.max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > self.max_chars // 2:
            truncated = truncated[:last_newline]
            hidden = total - last_newline

        truncated += (
            f"\n\n[OUTPUT TRUNCATED: ~{hidden} characters hidden. "
            f"Refine your query for more targeted results.]"
        )
        return ConstraintResult(
            content=truncated,
            was_truncated=True,
            metadata={
                "chars_shown": len(truncated),
                "chars_hidden": hidden,
                "total_chars": total,
            },
        )

    def __repr__(self) -> str:
        return f"TokenBudget(max_chars={self.max_chars})"


class Unconstrained(ObservationConstraint):
    """No-op constraint — passes everything through unchanged.

    Use as the baseline in ablation studies to measure the effect of
    constraints vs. unconstrained observation.
    """

    def apply(self, content: str) -> ConstraintResult:
        return ConstraintResult(
            content=content,
            was_truncated=False,
            metadata={"total_chars": len(content)},
        )
