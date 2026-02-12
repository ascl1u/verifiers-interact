"""Tests for ContextFolder implementations."""

from verifiers_interact.folders import (
    ContextFolder,
    TruncateFolder,
    HeadTailFolder,
    StructureFolder,
)


def _make_lines(n: int, prefix: str = "line") -> str:
    return "\n".join(f"{prefix} {i}" for i in range(n))


def _make_python_code(n_funcs: int = 5, body_lines: int = 10) -> str:
    """Generate fake Python code with structure."""
    parts = ["import os", "import sys", "from pathlib import Path", ""]
    for i in range(n_funcs):
        parts.append(f"def function_{i}(x, y):")
        for j in range(body_lines):
            parts.append(f"    result_{j} = x + y + {j}")
        parts.append(f"    return result_{body_lines - 1}")
        parts.append("")
    parts.append("class MyClass:")
    for i in range(3):
        parts.append(f"    def method_{i}(self):")
        parts.append(f"        pass")
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# TruncateFolder
# ---------------------------------------------------------------------------

class TestTruncateFolder:
    def test_basic_truncation(self):
        f = TruncateFolder()
        content = _make_lines(100)
        result = f.fold(content, 10)
        lines = result.split("\n")
        assert "line 0" in result
        assert "line 9" in result
        assert "[TRUNCATED" in result

    def test_keeps_exactly_budget_lines(self):
        f = TruncateFolder()
        content = _make_lines(50)
        result = f.fold(content, 20)
        # Should have 20 content lines + 1 blank + 1 notice = 22
        lines = result.split("\n")
        assert lines[0] == "line 0"
        assert lines[19] == "line 19"

    def test_repr(self):
        assert repr(TruncateFolder()) == "TruncateFolder()"


# ---------------------------------------------------------------------------
# HeadTailFolder
# ---------------------------------------------------------------------------

class TestHeadTailFolder:
    def test_shows_head_and_tail(self):
        f = HeadTailFolder(head_ratio=0.5)
        content = _make_lines(100)
        result = f.fold(content, 20)
        # Should show first ~10 and last ~10 lines
        assert "line 0" in result
        assert "line 99" in result
        assert "elided" in result

    def test_head_ratio_controls_split(self):
        f = HeadTailFolder(head_ratio=0.8)
        content = _make_lines(100)
        result = f.fold(content, 10)
        # 80% of 10 = 8 head lines, 2 tail lines
        assert "line 7" in result   # last head line
        assert "line 99" in result  # tail
        assert "line 98" in result  # tail

    def test_invalid_ratio_raises(self):
        import pytest
        with pytest.raises(ValueError):
            HeadTailFolder(head_ratio=0.0)
        with pytest.raises(ValueError):
            HeadTailFolder(head_ratio=1.0)

    def test_repr(self):
        assert "0.6" in repr(HeadTailFolder(0.6))


# ---------------------------------------------------------------------------
# StructureFolder
# ---------------------------------------------------------------------------

class TestStructureFolder:
    def test_extracts_python_structure(self):
        f = StructureFolder()
        code = _make_python_code(3, 10)
        result = f.fold(code, 20)
        # Should contain def, class, import lines
        assert "import os" in result
        assert "import sys" in result
        assert "def function_0" in result
        assert "class MyClass" in result
        assert "[FOLDED" in result

    def test_extracts_imports(self):
        f = StructureFolder()
        code = "import foo\nfrom bar import baz\nx = 1\ny = 2\nz = 3"
        result = f.fold(code, 3)
        assert "import foo" in result
        assert "from bar import baz" in result

    def test_fills_remaining_budget_with_head(self):
        f = StructureFolder()
        # Code with very few structural markers
        code = "\n".join(f"x_{i} = {i}" for i in range(50))
        result = f.fold(code, 10)
        # No structural markers, so should show first 10 lines
        assert "x_0 = 0" in result
        assert "x_9 = 9" in result

    def test_custom_patterns(self):
        import re
        custom = [re.compile(r"^SECTION:")]
        f = StructureFolder(extra_patterns=custom)
        content = "SECTION: intro\nsome text\nmore text\nSECTION: body\nstuff"
        result = f.fold(content, 3)
        assert "SECTION: intro" in result
        assert "SECTION: body" in result

    def test_repr(self):
        f = StructureFolder()
        assert "StructureFolder" in repr(f)


# ---------------------------------------------------------------------------
# Constraint + Folder composition
# ---------------------------------------------------------------------------

class TestConstraintFolderComposition:
    def test_linelimit_with_structure_folder(self):
        from verifiers_interact.constraints import LineLimit
        c = LineLimit(10, folder=StructureFolder())
        code = _make_python_code(3, 10)
        result = c.apply(code)
        assert result.was_truncated
        assert "def function_0" in result.content
        assert result.metadata["folder"] == "StructureFolder"

    def test_linelimit_with_headtail_folder(self):
        from verifiers_interact.constraints import LineLimit
        c = LineLimit(10, folder=HeadTailFolder(0.5))
        content = _make_lines(100)
        result = c.apply(content)
        assert result.was_truncated
        assert "line 0" in result.content
        assert "line 99" in result.content
        assert "elided" in result.content

    def test_linelimit_default_uses_truncate(self):
        from verifiers_interact.constraints import LineLimit
        c = LineLimit(10)
        content = _make_lines(50)
        result = c.apply(content)
        assert result.was_truncated
        assert "TRUNCATED" in result.content

    def test_under_budget_ignores_folder(self):
        from verifiers_interact.constraints import LineLimit
        c = LineLimit(100, folder=StructureFolder())
        content = _make_lines(10)
        result = c.apply(content)
        assert not result.was_truncated
        assert result.content == content
