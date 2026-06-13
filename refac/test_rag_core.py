#!/usr/bin/env python3
"""
Tests for VaultRAG refactored core — chunking and retrieval.
Run: cd /home/fede/scripts/refac && python -m pytest test_rag_core.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_core import (
    _chunk_generic_code,
    _chunk_python,
    _parse_cartelle_from_context,
    _vector_query,
    chunk_file,
    chunk_text,
    cosine_sim,
    strip_emoji,
    AppState,
    VaultRagContext,
    CODE_EXTENSIONS,
)


# ═══════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def ctx() -> VaultRagContext:
    """Minimal VaultRagContext for testing (no real ChromaDB or LLM)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        yield VaultRagContext(
            vault_path=tmp / "vault",
            db_path=tmp / "db",
            history_file=tmp / "history.json",
            state_file=tmp / "state.json",
            corrections_file=tmp / "corrections.json",
            error_memory_file=tmp / "error_memory.json",
            graph_file=tmp / "graph.json",
        )


# ═══════════════════════════════════════════════════════════
#  CHUNK_TEXT TESTS
# ═══════════════════════════════════════════════════════════

class TestChunkText:
    def test_basic_chunking(self) -> None:
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 2
        # Check overlap
        assert chunks[0][450:500] == chunks[1][:50]

    def test_small_text(self) -> None:
        text = "hello"
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "hello"

    def test_empty_text(self) -> None:
        chunks = chunk_text("", chunk_size=500, chunk_overlap=50)
        assert chunks == []

    def test_exact_size(self) -> None:
        text = "B" * 500
        # step = 500 - 50 = 450, so chunks at [0:500] and [450:500]
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 2
        assert len(chunks[0]) == 500
        assert len(chunks[1]) == 50

    def test_overlap_respected(self) -> None:
        text = "X" * 600
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
        # step = 500 - 100 = 400, so chunks at [0:500], [400:600]
        assert len(chunks) == 2
        assert chunks[0][400:500] == chunks[1][:100]


# ═══════════════════════════════════════════════════════════
#  PYTHON CHUNKING TESTS
# ═══════════════════════════════════════════════════════════

class TestChunkPython:
    def test_single_function(self) -> None:
        source = (
            "def hello():\n"
            "    print('hello')\n"
            "\n"
            "def world():\n"
            "    print('world')\n"
        )
        chunks = _chunk_python(source, "test.py")
        assert len(chunks) >= 2
        # Check that function names appear in chunk prefixes
        all_text = "\n".join(chunks)
        assert "def: hello" in all_text
        assert "def: world" in all_text

    def test_class(self) -> None:
        source = (
            "class Foo:\n"
            "    def bar(self):\n"
            "        pass\n"
        )
        chunks = _chunk_python(source, "test.py")
        all_text = "\n".join(chunks)
        assert "class: Foo" in all_text

    def test_syntax_error_fallback(self) -> None:
        source = "def broken(\n    no close paren"
        chunks = _chunk_python(source, "test.py")
        # Should fall back to linear chunking
        assert len(chunks) >= 1

    def test_top_level_code(self) -> None:
        source = (
            "import os\n"
            "x = 42\n"
            "\n"
            "def foo():\n"
            "    pass\n"
        )
        chunks = _chunk_python(source, "test.py")
        all_text = "\n".join(chunks)
        assert "top-level" in all_text
        assert "import os" in all_text

    def test_async_function(self) -> None:
        source = (
            "async def fetch():\n"
            "    return 'data'\n"
        )
        chunks = _chunk_python(source, "test.py")
        all_text = "\n".join(chunks)
        assert "def: fetch" in all_text


# ═══════════════════════════════════════════════════════════
#  GENERIC CODE CHUNKING TESTS (C/C++/Bash)
# ═══════════════════════════════════════════════════════════

class TestChunkGenericCode:
    def test_c_function(self) -> None:
        source = (
            "int main(int argc, char **argv) {\n"
            "    return 0;\n"
            "}\n"
            "\n"
            "void helper(void) {\n"
            "    return;\n"
            "}\n"
        )
        chunks = _chunk_generic_code(source, "test.c", "c")
        assert len(chunks) >= 2
        all_text = "\n".join(chunks)
        assert "function:" in all_text

    def test_cpp_function(self) -> None:
        source = (
            "int compute(int x, int y) {\n"
            "    return x + y;\n"
            "}\n"
        )
        chunks = _chunk_generic_code(source, "test.cpp", "cpp")
        assert len(chunks) >= 1

    def test_bash_function(self) -> None:
        source = (
            "my_func() {\n"
            '    echo "hello"\n'
            "}\n"
            "\n"
            "other_func() {\n"
            '    echo "world"\n'
            "}\n"
        )
        chunks = _chunk_generic_code(source, "test.sh", "sh")
        assert len(chunks) >= 2

    def test_no_functions_fallback(self) -> None:
        source = "#include <stdio.h>\nint x = 5;\n"
        chunks = _chunk_generic_code(source, "test.c", "c")
        # No function boundaries found → fallback to linear chunking
        assert len(chunks) >= 1

    def test_unknown_lang_fallback(self) -> None:
        source = "some random text"
        chunks = _chunk_generic_code(source, "test.xyz", "xyz")
        assert len(chunks) >= 1


# ═══════════════════════════════════════════════════════════
#  CHUNK_FILE DISPATCH TESTS
# ═══════════════════════════════════════════════════════════

class TestChunkFile:
    def test_default_mode_ignores_extension(self) -> None:
        """In default mode, all files use linear chunking."""
        content = "def foo():\n    pass\n"
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            p = Path(f.name)
            chunks = chunk_file(p, content, mode="default")
            # Should NOT use AST chunking
            assert len(chunks) >= 1
            # No "def:" prefix in default mode
            assert "def:" not in chunks[0]

    def test_codice_mode_python(self) -> None:
        content = "def hello():\n    print('hi')\n"
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            p = Path(f.name)
            chunks = chunk_file(p, content, mode="codice")
            all_text = "\n".join(chunks)
            assert "def: hello" in all_text

    def test_codice_mode_c(self) -> None:
        content = "int main() {\n    return 0;\n}\n"
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            p = Path(f.name)
            chunks = chunk_file(p, content, mode="codice")
            assert len(chunks) >= 1

    def test_unknown_extension_linear(self) -> None:
        content = "some text here"
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            p = Path(f.name)
            chunks = chunk_file(p, content, mode="codice")
            # .txt not in CODE_EXTENSIONS → linear
            assert len(chunks) >= 1


# ═══════════════════════════════════════════════════════════
#  COSINE SIMILARITY TESTS
# ═══════════════════════════════════════════════════════════

class TestCosineSim:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 0.0, 0.0])
        assert cosine_sim(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_sim(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_sim(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert cosine_sim(a, b) == 0.0


# ═══════════════════════════════════════════════════════════
#  STRIP EMOJI TESTS
# ═══════════════════════════════════════════════════════════

class TestStripEmoji:
    def test_plain_text(self) -> None:
        assert strip_emoji("hello world") == "hello world"

    def test_removes_emoji(self) -> None:
        # Common emoji in supplementary planes
        result = strip_emoji("hello \U0001f600")
        assert "\U0001f600" not in result

    def test_preserves_ascii(self) -> None:
        assert strip_emoji("hello! 123 @#$") == "hello! 123 @#$"


# ═══════════════════════════════════════════════════════════
#  CARTELLE PARSING TESTS
# ═══════════════════════════════════════════════════════════

class TestParseCartelle:
    def test_single_cartella(self) -> None:
        ctx_text = "- MyFolder/\n"
        result = _parse_cartelle_from_context(ctx_text)
        assert result == ["MyFolder"]

    def test_multiple_cartelle(self) -> None:
        ctx_text = "- Folder1/\n- Folder2/\n"
        result = _parse_cartelle_from_context(ctx_text)
        assert result == ["Folder1", "Folder2"]

    def test_empty_context(self) -> None:
        assert _parse_cartelle_from_context("") == []

    def test_no_cartella_pattern(self) -> None:
        ctx_text = "just some text"
        assert _parse_cartelle_from_context(ctx_text) == []


# ═══════════════════════════════════════════════════════════
#  APP STATE TESTS
# ═══════════════════════════════════════════════════════════

class TestAppState:
    def test_default_state(self) -> None:
        state = AppState()
        assert state.mode == "default"
        assert state.lang == "auto"
        assert state.materia == ""

    def test_set_mode(self) -> None:
        state = AppState()
        assert state.set_mode("codice") is True
        assert state.mode == "codice"
        assert state.set_mode("invalid") is False

    def test_reset(self) -> None:
        state = AppState()
        state.set_mode("codice")
        state.lang = "python"
        state.reset()
        assert state.mode == "default"
        assert state.lang == "python"  # reset does NOT touch lang

    def test_materia_property(self) -> None:
        state = AppState()
        state.materie_attive = {"math": "context1", "physics": "context2"}
        result = state.materia
        assert "math" in result
        assert "physics" in result

    def test_materia_context(self) -> None:
        state = AppState()
        state.materie_attive = {"math": "ctx1", "physics": "ctx2"}
        result = state.materia_context
        assert "ctx1" in result
        assert "ctx2" in result
        assert "---" in result


# ═══════════════════════════════════════════════════════════
#  VECTOR QUERY TESTS (mocked ChromaDB)
# ═══════════════════════════════════════════════════════════

class TestVectorQuery:
    def _make_mock_collection(self, docs: list[str], types: list[str],
                              paths: list[str], files: list[str]) -> MagicMock:
        col = MagicMock()
        col.count.return_value = len(docs)
        col.query.return_value = {
            "documents": [docs],
            "metadatas": [
                [{"type": t, "path": p, "file": f}
                 for t, p, f in zip(types, paths, files)]
            ],
        }
        return col

    def test_no_filter_returns_all(self, ctx: VaultRagContext) -> None:
        col = self._make_mock_collection(
            docs=["doc1", "doc2"],
            types=[".py", ".md"],
            paths=["/a.py", "/b.md"],
            files=["a.py", "b.md"],
        )
        ctx.state.lang = "auto"

        with patch.object(ctx, "get_collection", return_value=col):
            with patch.object(ctx, "cached_encode", return_value=np.array([0.1, 0.2])):
                docs, metas = _vector_query(ctx, "test", q_embed=[0.1, 0.2], n=2)
                assert len(docs) == 2

    def test_language_filter_python(self, ctx: VaultRagContext) -> None:
        col = self._make_mock_collection(
            docs=["py_doc", "md_doc", "py_doc2"],
            types=[".py", ".md", ".py"],
            paths=["/a.py", "/b.md", "/c.py"],
            files=["a.py", "b.md", "c.py"],
        )
        ctx.state.lang = "python"

        with patch.object(ctx, "get_collection", return_value=col):
            with patch.object(ctx, "cached_encode", return_value=np.array([0.1, 0.2])):
                docs, metas = _vector_query(ctx, "test", q_embed=[0.1, 0.2], n=2)
                assert all(m["type"] == ".py" for m in metas)
                assert len(docs) == 2

    def test_language_filter_c(self, ctx: VaultRagContext) -> None:
        col = self._make_mock_collection(
            docs=["c_doc", "py_doc", "h_doc"],
            types=[".c", ".py", ".h"],
            paths=["/a.c", "/b.py", "/c.h"],
            files=["a.c", "b.py", "c.h"],
        )
        ctx.state.lang = "c"

        with patch.object(ctx, "get_collection", return_value=col):
            with patch.object(ctx, "cached_encode", return_value=np.array([0.1, 0.2])):
                docs, metas = _vector_query(ctx, "test", q_embed=[0.1, 0.2], n=2)
                assert all(m["type"] in (".c", ".h") for m in metas)

    def test_fallback_when_no_results(self, ctx: VaultRagContext) -> None:
        """When language filter yields no results, should fallback to global."""
        col = self._make_mock_collection(
            docs=["py_doc"],
            types=[".py"],
            paths=["/a.py"],
            files=["a.py"],
        )
        ctx.state.lang = "c"  # looking for .c/.h, but only .py exists

        with patch.object(ctx, "get_collection", return_value=col):
            with patch.object(ctx, "cached_encode", return_value=np.array([0.1, 0.2])):
                docs, metas = _vector_query(ctx, "test", q_embed=[0.1, 0.2], n=1)
                # Fallback should return the .py doc
                assert len(docs) == 1
