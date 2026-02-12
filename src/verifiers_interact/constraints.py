"""
Observation constraints for controlling what the agent sees.

A constraint combines two decisions:
  1. **Budget** — How much output to show (line count, character count, etc.)
  2. **Folding** — How to compress what exceeds the budget

This separation is the core abstraction. It lets researchers run controlled
ablation studies: same budget, different folding strategies (or vice versa).

Usage:
    from verifiers_interact import LineLimit
    from verifiers_interact.folders import StructureFolder

    # Same 50-line budget, different folding strategies:
    naive = LineLimit(50)                                  # hard truncation
    smart = LineLimit(50, folder=StructureFolder())       # structural extraction
    split = LineLimit(50, folder=HeadTailFolder(0.7))     # head + tail
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .folders import ContextFolder


@dataclass
class ConstraintResult:
    """Result of applying an observation constraint.

    Attributes:
        content: The (possibly folded/truncated) output text.
        was_truncated: Whether the constraint actually triggered.
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
            ConstraintResult with (possibly folded) content and metadata.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class LineLimit(ObservationConstraint):
    """Constrain tool output to a maximum number of lines.

    When output exceeds the limit, the folder strategy determines
    how to compress it. Default: naive head truncation.

    Args:
        max_lines: Maximum number of lines to show. Default: 200.
        folder: ContextFolder strategy for compression. Default: TruncateFolder.
    """

    def __init__(self, max_lines: int = 200, folder: ContextFolder | None = None):
        if max_lines < 1:
            raise ValueError(f"max_lines must be >= 1, got {max_lines}")
        self.max_lines = max_lines
        # Lazy import to avoid circular dependency
        if folder is None:
            from .folders import TruncateFolder

            folder = TruncateFolder()
        self.folder = folder

    def apply(self, content: str) -> ConstraintResult:
        lines = content.split("\n")
        total = len(lines)

        if total <= self.max_lines:
            return ConstraintResult(
                content=content,
                was_truncated=False,
                metadata={"lines_shown": total, "lines_hidden": 0, "total_lines": total},
            )

        # Delegate compression to the folder
        folded = self.folder.fold(content, self.max_lines)
        hidden = total - self.max_lines

        return ConstraintResult(
            content=folded,
            was_truncated=True,
            metadata={
                "lines_shown": self.max_lines,
                "lines_hidden": hidden,
                "total_lines": total,
                "folder": type(self.folder).__name__,
            },
        )

    def __repr__(self) -> str:
        return f"LineLimit(max_lines={self.max_lines}, folder={self.folder!r})"


class TokenBudget(ObservationConstraint):
    """Constrain tool output to a character budget (proxy for tokens).

    Uses character count as a fast proxy for token count (~4 chars/token).
    When exceeded, delegates to the folder for compression.

    Args:
        max_chars: Maximum characters to keep. Default: 4000 (~1000 tokens).
        folder: ContextFolder strategy. Default: TruncateFolder.
    """

    def __init__(self, max_chars: int = 4000, folder: ContextFolder | None = None):
        if max_chars < 1:
            raise ValueError(f"max_chars must be >= 1, got {max_chars}")
        self.max_chars = max_chars
        if folder is None:
            from .folders import TruncateFolder

            folder = TruncateFolder()
        self.folder = folder

    def apply(self, content: str) -> ConstraintResult:
        total = len(content)

        if total <= self.max_chars:
            return ConstraintResult(
                content=content,
                was_truncated=False,
                metadata={"chars_shown": total, "chars_hidden": 0, "total_chars": total},
            )

        # Convert char budget to approximate line budget for the folder
        avg_line_len = max(1, total // max(1, content.count("\n") + 1))
        budget_lines = max(1, self.max_chars // avg_line_len)

        folded = self.folder.fold(content, budget_lines)
        hidden = total - self.max_chars

        return ConstraintResult(
            content=folded,
            was_truncated=True,
            metadata={
                "chars_shown": len(folded),
                "chars_hidden": hidden,
                "total_chars": total,
                "folder": type(self.folder).__name__,
            },
        )

    def __repr__(self) -> str:
        return f"TokenBudget(max_chars={self.max_chars}, folder={self.folder!r})"


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
