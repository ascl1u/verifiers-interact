"""
Context folding strategies for compressed observation.

When an ObservationConstraint determines that output exceeds the budget,
a ContextFolder determines HOW to compress it. This separates the
"how much?" decision (constraint) from the "how?" decision (folder).

The key insight: naive truncation destroys structure. A folder can
preserve the information topology of the output — keeping function
signatures, class definitions, and structural markers — while discarding
the body. This gives the model a "map" it can use to navigate deeper.

Usage:
    from verifiers_interact import LineLimit
    from verifiers_interact.folders import HeadTailFolder

    constraint = LineLimit(50, folder=HeadTailFolder(head_ratio=0.6))
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


class ContextFolder(ABC):
    """Strategy for compressing content that exceeds an observation budget.

    Subclass this to implement custom folding strategies for research.
    The `fold()` method receives the full content and a line budget,
    and must return compressed content that fits within the budget.
    """

    @abstractmethod
    def fold(self, content: str, budget_lines: int) -> str:
        """Compress content to approximately fit within a line budget.

        Args:
            content: The full, untruncated output string.
            budget_lines: Target number of lines for the compressed output.

        Returns:
            Compressed string, ideally within budget_lines.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class TruncateFolder(ContextFolder):
    """Naive head truncation — keep the first N lines, discard the rest.

    This is the baseline folder. It preserves locality (you see the start)
    but destroys global structure. Use as a control in ablation studies.
    """

    def fold(self, content: str, budget_lines: int) -> str:
        lines = content.split("\n")
        kept = lines[:budget_lines]
        hidden = len(lines) - budget_lines
        kept.append("")
        kept.append(
            f"[TRUNCATED: {hidden} of {len(lines)} lines hidden. "
            f"Use targeted queries to navigate.]"
        )
        return "\n".join(kept)


class HeadTailFolder(ContextFolder):
    """Show the first and last portions of output, eliding the middle.

    This preserves both locality (the start) and recency (the end),
    at the cost of hiding the middle. Useful for log-like output
    where the beginning has context and the end has the latest state.

    Args:
        head_ratio: Fraction of budget allocated to the head (0.0-1.0).
            Default: 0.6 (60% head, 40% tail).
    """

    def __init__(self, head_ratio: float = 0.6):
        if not 0.0 < head_ratio < 1.0:
            raise ValueError(f"head_ratio must be in (0, 1), got {head_ratio}")
        self.head_ratio = head_ratio

    def fold(self, content: str, budget_lines: int) -> str:
        lines = content.split("\n")
        total = len(lines)

        head_n = max(1, int(budget_lines * self.head_ratio))
        tail_n = max(1, budget_lines - head_n)
        hidden = total - head_n - tail_n

        head = lines[:head_n]
        tail = lines[-tail_n:]

        result = head
        result.append("")
        result.append(f"[... {hidden} lines elided ...]")
        result.append("")
        result.extend(tail)
        return "\n".join(result)

    def __repr__(self) -> str:
        return f"HeadTailFolder(head_ratio={self.head_ratio})"


# Patterns for structural markers in code
_STRUCTURE_PATTERNS = [
    re.compile(r"^\s*(class\s+\w+|def\s+\w+|async\s+def\s+\w+)"),  # Python defs
    re.compile(r"^\s*(import\s+|from\s+\w+\s+import)"),              # Imports
    re.compile(r"^\s*(#{1,3}\s+)"),                                   # Markdown headers
    re.compile(r"^\s*(function\s+\w+|const\s+\w+\s*=|export\s+)"),    # JS/TS defs
    re.compile(r"^(---|\*\*\*|===)"),                                 # Separators
]


class StructureFolder(ContextFolder):
    """Extract structural markers, discarding implementation bodies.

    Scans the content for structural patterns (function/class definitions,
    imports, markdown headers, separators) and keeps only those lines.
    This gives the model a "table of contents" it can navigate into
    with targeted queries rather than scrolling through everything.

    When structural markers fit within the budget, the remaining budget
    is filled with the first lines of content (preserving locality).

    Args:
        extra_patterns: Additional regex patterns to treat as structural.
    """

    def __init__(self, extra_patterns: list[re.Pattern] | None = None):
        self.patterns = list(_STRUCTURE_PATTERNS)
        if extra_patterns:
            self.patterns.extend(extra_patterns)

    def fold(self, content: str, budget_lines: int) -> str:
        lines = content.split("\n")
        total = len(lines)

        # Extract structural lines with their line numbers
        structural: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            if any(p.match(line) for p in self.patterns):
                structural.append((i, line))

        # If structural markers fill the budget, use them
        if len(structural) >= budget_lines:
            result_lines = [line for _, line in structural[:budget_lines]]
        else:
            # Mix: structural markers first, then head-fill remaining budget
            struct_set = {i for i, _ in structural}
            result_lines = [line for _, line in structural]
            remaining = budget_lines - len(structural)
            for i, line in enumerate(lines):
                if remaining <= 0:
                    break
                if i not in struct_set:
                    result_lines.append(line)
                    remaining -= 1

        hidden = total - len(result_lines)
        result_lines.append("")
        result_lines.append(
            f"[FOLDED: showing {len(result_lines) - 2} structural markers from "
            f"{total} lines. Query specific symbols to expand.]"
        )
        return "\n".join(result_lines)

    def __repr__(self) -> str:
        return f"StructureFolder(patterns={len(self.patterns)})"
