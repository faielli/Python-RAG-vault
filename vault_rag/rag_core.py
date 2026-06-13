#!/usr/bin/env python3
"""
VaultRAG — RAG Core
All RAG logic: chunking, embedding, retrieval, graph, corrections.
Dependency injection via the VaultRagContext class — no module-level globals for state.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import chromadb
import networkx as nx
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

log = logging.getLogger("vaultrag.core")

# ── Pre-compiled regexes ─────────────────────────────────
_RE_TRIPLE = re.compile(
    r"^(?!\s)[\w:*&<>][\w\s:*&<>,]*\s+\w+\s*\([^;{]*\)\s*\{", re.MULTILINE
)
_RE_BASH_FN = re.compile(r"^(?:function\s+\w+|\w+)\s*\(\s*\)\s*\{", re.MULTILINE)
_RE_CART = re.compile(r"^[-*]\s+(.+?)/\s*$", re.MULTILINE)
_RE_WORD = re.compile(r"\b\w{3,}\b")

# ── Constants ─────────────────────────────────────────────
SYSTEM_PROMPT_DEFAULT = (
    "Sei un assistente che risponde in base ai file del vault forniti. "
    "Rispondi sempre in italiano. Sii conciso e diretto. "
    "Non usare emoji o simboli Unicode speciali, solo testo ASCII. "
    "Se la risposta non e' nei file, dillo chiaramente senza inventare. "
    "Formatta le risposte in markdown: **grassetto** per termini chiave, "
    "`codice` per comandi e nomi file, ```bash per blocchi di codice shell, "
    "## per titoli di sezione, - per liste. Non usare emoji."
)

SYSTEM_PROMPT_CODICE = (
    "Sei un assistente esperto di programmazione. "
    "Il tuo obiettivo principale e' aiutare a scrivere, analizzare, debuggare e migliorare codice. "
    "Rispondi sempre in italiano, ma mantieni i nomi tecnici (variabili, funzioni, librerie) in inglese. "
    "Segui queste regole rigidamente:\n"
    "1. CODICE: mostra sempre codice funzionante e COMPLETO, con il linguaggio corretto nel fence "
    "(```c, ```python, ```bash, ecc.). Non troncare mai il codice: scrivilo sempre fino all'ultima riga.\n"
    "2. SPIEGAZIONI: brevi e tecniche. Vai dritto al punto.\n"
    "3. STRUTTURA: usa ## per sezioni (Problema, Soluzione, Note), `inline code` per funzioni/variabili/file.\n"
    "4. ERRORI: indica prima la causa radice, poi la correzione con snippet completo.\n"
    "5. BEST PRACTICE: segnala con > [!NOTE] eventuali anti-pattern.\n"
    "6. Se il codice non e' nel vault, usa le best practice del settore senza inventare API.\n"
    "7. Non usare emoji. Solo informazioni tecniche utili.\n"
    "8. Non scrivere mai '[...resto del codice...]', '[omissis]' o commenti simili: "
    "il codice deve essere sempre integrale e pronto alla compilazione/esecuzione.\n"
    "Linguaggi supportati: Python, C, C++, Bash. "
    "Crea codice compatibile con sistemi Linux/Unix, specificatamente per Arch Linux, "
    "utilizzando tutte le librerie necessarie per il funzionamento del codice. "
    "IMPORTANTISSIMO: verifica sempre il codice almeno due volte prima di scrivere la risposta. "
)

LANG_SPECIFIC_PROMPTS: dict[str, str] = {
    "python_language": "Scrivi programmi in Python. Verifica e testa attentamente il codice prima di inviarlo.",
    "c_language": (
        "Scrivi programmi in C. Verifica e testa attentamente il codice prima di inviarlo.\n"
        "REGOLE SPECIFICHE PER CODICE C:\n"
        "- Fai molta attenzione ad usare le librerie necessarie "
        "- Aggiungi SEMPRE '#define _POSIX_C_SOURCE 200809L' come primissima riga del file "
        "(prima di qualsiasi #include) in qualsiasi sorgente C che usa: "
        "nanosleep, clock_gettime, gmtime_r, localtime_r, strdup, getline, pread, pwrite "
        "o qualsiasi altra API POSIX/XSI.\n"
        "Obbligatorio: la compilazione avviene con '-std=c17' che nasconde le estensioni POSIX per default.\n"
        "- Includi sempre tutti gli header necessari: <time.h> per clock_gettime/nanosleep, "
        "<termios.h> per tcgetattr/tcsetattr, <unistd.h> per read/write/STDIN_FILENO, "
        "<limits.h> per LONG_MAX, <errno.h> per errno/EINTR/ERANGE.\n"
        "- Riporta sempre il comando di compilazione esatto nel commento in cima al file, "
        "es: '// gcc -std=c17 -Wall -Wextra -O2 file.c -o file' (aggiungi -lm se usi <math.h>).\n"
        "- Controlla SEMPRE il valore di ritorno di malloc/realloc. "
        "Non passare mai il puntatore originale direttamente a realloc: usa sempre un temporaneo.\n"
        "- In modalita' raw terminal (VMIN/VTIME): usa sempre VMIN=1 VTIME=0 (bloccante) "
        "e scrivi a schermo con write() + '\\r\\n', mai printf senza \\r."
    ),
    "cpp_language": "Scrivi programmi in C++. Verifica e testa attentamente il codice prima di inviarlo.",
}

LANG_EXT_MAP: dict[str, set[str]] = {
    "python": {".py", ".pyw"},
    "c":      {".c", ".h"},
    "cpp":    {".cpp", ".hpp", ".cc"},
    "bash":   {".sh", ".bash"},
}

LANG_CODE_PROMPTS: dict[str, str] = {
    "python": (
        "MODALITÀ PYTHON: Scrivi codice Python 3.10+ completo e funzionante. "
        "Usa type hints, docstring brevi, best practice moderne. "
        "Non troncare mai il codice. Includi tutti gli import necessari."
    ),
    "c": (
        "MODALITÀ C: Scrivi codice C17 completo e compilabile. "
        "REGOLE:\n"
        "- '#define _POSIX_C_SOURCE 200809L' prima di ogni #include se usi API POSIX.\n"
        "- Includi SEMPRE tutti gli header necessari.\n"
        "- Controlla SEMPRE il return di malloc/realloc. Usa puntatore temporaneo con realloc.\n"
        "- Mostra il comando di compilazione esatto in un commento in cima.\n"
        "- Gestisci errno e EINTR dove rilevante. Mai usare funzioni deprecate."
    ),
    "cpp": (
        "MODALITÀ C++: Scrivi codice C++17/20 moderno. "
        "Usa smart pointer, RAII, std::string_view, auto, range-for. "
        "Evita raw new/delete. Preferisci std::array/std::vector."
    ),
    "bash": (
        "MODALITÀ BASH: Scrivi script robusti con 'set -euo pipefail'. "
        "Usa double-quote per le variabili. Gestisci gli errori. "
        "Preferisci built-in a comandi esterni."
    ),
}

CODE_EXTENSIONS: dict[str, str] = {
    ".py": "python", ".pyw": "python",
    ".c":  "c",      ".h":   "c",
    ".cpp": "cpp",    ".hpp": "cpp", ".cc": "cpp",
    ".sh": "sh",     ".bash": "sh",
}

# Emoji strip table — built once
_STRIP_EMOJI_TABLE: dict[int, None] = {
    i: None for i in range(256, 0x110000)
    if not unicodedata.category(chr(i)).startswith(("L", "N", "P", "Z"))
}


# ═══════════════════════════════════════════════════════════
#  STATE (replaces module-level globals)
# ═══════════════════════════════════════════════════════════

class AppState:
    """Mutable application state, thread-safe via internal lock."""

    __slots__ = (
        "system_prompt", "mode", "lang", "materie_attive",
        "user_context", "reindex_log", "reindex_running",
        "graph_log", "graph_running",
    )

    def __init__(self) -> None:
        self.system_prompt: str = SYSTEM_PROMPT_DEFAULT
        self.mode: str = "default"
        self.lang: str = "auto"
        self.materie_attive: dict[str, str] = {}
        self.user_context: str = ""
        self.reindex_log: list[str] = []
        self.reindex_running: bool = False
        self.graph_log: list[str] = []
        self.graph_running: bool = False

    @property
    def materia(self) -> str:
        return ", ".join(self.materie_attive.keys()) if self.materie_attive else ""

    @property
    def materia_context(self) -> str:
        return "\n\n---\n\n".join(self.materie_attive.values())

    def set_mode(self, mode: str) -> bool:
        if mode not in ("default", "codice"):
            return False
        self.system_prompt = SYSTEM_PROMPT_DEFAULT if mode == "default" else SYSTEM_PROMPT_CODICE
        self.mode = mode
        return True

    def reset(self) -> None:
        self.system_prompt = SYSTEM_PROMPT_DEFAULT
        self.mode = "default"
        self.materie_attive = {}


# ═══════════════════════════════════════════════════════════
#  CONTEXT (dependency injection container)
# ═══════════════════════════════════════════════════════════

@dataclass
class EmbeddingCache:
    """Simple LRU-style embedding cache."""
    _cache: dict[str, np.ndarray] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    max_size: int = 512

    def get(self, text: str) -> Optional[np.ndarray]:
        with self._lock:
            return self._cache.get(text)

    def put(self, text: str, emb: np.ndarray) -> None:
        with self._lock:
            if len(self._cache) >= self.max_size:
                try:
                    self._cache.pop(next(iter(self._cache)))
                except StopIteration:
                    pass
            self._cache[text] = emb


@dataclass
class VaultRagContext:
    """
    Central DI container. All shared state and lazily-created
    singletons are held here and passed to functions that need them.
    """
    # ── Paths & config ──────────────────────────────
    vault_path: Path
    db_path: Path
    history_file: Path
    state_file: Path
    corrections_file: Path
    error_memory_file: Path
    graph_file: Path
    chat_dir: str = "_chat"
    context_dir: str = "_context"
    chunk_size: int = 500
    chunk_overlap: int = 50
    n_results: int = 2
    max_history: int = 20
    dup_threshold: float = 0.97
    ocr_langs: str = "ita+eng"
    graph_chunk_sample: int = 3
    graph_max_files: int = 200
    graph_max_workers: int = 8
    max_corrections: int = 500
    correction_boost: float = 2.0
    error_context_n: int = 5
    code_embed_model: str = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
    embed_model: str = "all-MiniLM-L6-v2"
    extensions: tuple[str, ...] = (
        "*.md", "*.txt", "*.pdf", "*.docx", "*.epub",
        "*.odt", "*.ods", "*.html", "*.htm",
    )

    # ── LLM config ──────────────────────────────────
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "qwen-plus"
    max_tokens: int = 8192

    # ── State ───────────────────────────────────────
    state: AppState = field(default_factory=AppState)

    # ── Lazy singletons ─────────────────────────────
    _llm_client: Optional[OpenAI] = field(default=None, repr=False)
    _embedder: Optional[SentenceTransformer] = field(default=None, repr=False)
    _code_embedder: Optional[SentenceTransformer] = field(default=None, repr=False)
    _chroma: Optional[chromadb.PersistentClient] = field(default=None, repr=False)
    _chroma_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _embedder_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _embed_cache: EmbeddingCache = field(default_factory=EmbeddingCache, repr=False)
    _graph_cache: dict = field(default_factory=lambda: {"G": None, "mtime": 0.0}, repr=False)
    _graph_cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _corrections_emb_cache: dict = field(default_factory=dict, repr=False)
    _corrections_emb_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _vault_files_cache: dict = field(default_factory=lambda: {"files": None, "ts": 0.0}, repr=False)
    _vault_files_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── Collections (lazy) ──────────────────────────
    _collection: Optional[Any] = field(default=None, repr=False)
    _corrections_collection: Optional[Any] = field(default=None, repr=False)

    # ── LLM client ──────────────────────────────────
    def get_llm(self) -> OpenAI:
        if self._llm_client is None:
            self._llm_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._llm_client

    def reset_llm_client(self) -> None:
        self._llm_client = None

    # ── Embedder ────────────────────────────────────
    def get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    self._embedder = SentenceTransformer(self.embed_model)
        return self._embedder

    def get_code_embedder(self) -> SentenceTransformer:
        if self._code_embedder is None:
            try:
                self._code_embedder = SentenceTransformer(self.code_embed_model)
                log.info("Code embedder loaded: %s", self.code_embed_model)
            except Exception as exc:
                log.warning("Code embedder unavailable (%s), using fallback", exc)
                self._code_embedder = self.get_embedder()
        return self._code_embedder

    # ── ChromaDB ────────────────────────────────────
    def get_chroma(self) -> chromadb.PersistentClient:
        if self._chroma is None:
            with self._chroma_lock:
                if self._chroma is None:
                    self._chroma = chromadb.PersistentClient(path=str(self.db_path))
        return self._chroma

    def get_collection(self) -> Any:
        if self._collection is None:
            self._collection = self.get_chroma().get_or_create_collection("obsidian_vault")
        return self._collection

    def get_corrections_collection(self) -> Any:
        if self._corrections_collection is None:
            self._corrections_collection = self.get_chroma().get_or_create_collection("rag_corrections")
        return self._corrections_collection

    # ── Embedding cache ─────────────────────────────
    def cached_encode(self, text: str) -> np.ndarray:
        emb = self._embed_cache.get(text)
        if emb is not None:
            return emb
        emb = self.get_embedder().encode(text, convert_to_numpy=True)
        self._embed_cache.put(text, emb)
        return emb


# ═══════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════

def strip_emoji(text: str) -> str:
    return text.translate(_STRIP_EMOJI_TABLE)


def vault_files(ctx: VaultRagContext) -> list[Path]:
    """Return vault files, excluding chat dir. Cached 30s."""
    with ctx._vault_files_lock:
        now = time.monotonic()
        if ctx._vault_files_cache["files"] is not None and now - ctx._vault_files_cache["ts"] < 30:
            return ctx._vault_files_cache["files"]
        seen: set[Path] = set()
        result: list[Path] = []
        for ext in ctx.extensions:
            for f in ctx.vault_path.rglob(ext):
                if ctx.chat_dir not in f.parts and f not in seen:
                    seen.add(f)
                    result.append(f)
        ctx._vault_files_cache["files"] = result
        ctx._vault_files_cache["ts"] = now
    return result


def invalidate_vault_cache(ctx: VaultRagContext) -> None:
    with ctx._vault_files_lock:
        ctx._vault_files_cache["files"] = None


def extract_text(file: Path, ocr_langs: str = "ita+eng") -> str:
    """Extract text from any supported format. Returns empty string on error."""
    try:
        import ebooklib
        import fitz
        from bs4 import BeautifulSoup
        from docx import Document
        from ebooklib import epub
        from odf import teletype
        from odf.opendocument import load as odf_load

        suf = file.suffix.lower()
        if suf in (".md", ".txt"):
            return file.read_text(errors="ignore")
        if suf == ".pdf":
            pages: list[str] = []
            with fitz.open(str(file)) as doc:
                for page in doc:
                    text = page.get_text()
                    if not text.strip():
                        try:
                            import io
                            import pytesseract
                            from PIL import Image
                            pix = page.get_pixmap(dpi=200)
                            img = Image.open(io.BytesIO(pix.tobytes("png")))
                            text = pytesseract.image_to_string(img, lang=ocr_langs)
                        except Exception:
                            pass
                    pages.append(text)
            return "\n".join(pages)
        if suf == ".docx":
            doc = Document(str(file))
            return "\n".join(p.text for p in doc.paragraphs)
        if suf == ".epub":
            book = epub.read_epub(str(file))
            return "\n".join(
                BeautifulSoup(item.get_content(), "html.parser").get_text()
                for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
            )
        if suf in (".odt", ".ods"):
            return teletype.extractText(odf_load(str(file)).body)
        if suf in (".html", ".htm"):
            return BeautifulSoup(file.read_text(errors="ignore"), "html.parser").get_text()
    except Exception as exc:
        log.warning("extract_text(%s): %s", file.name, exc)
    return ""


# ═══════════════════════════════════════════════════════════
#  CHUNKING
# ═══════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    step = chunk_size - chunk_overlap
    return [text[i: i + chunk_size] for i in range(0, len(text), step)]


def _chunk_python(source: str, filename: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """AST-based semantic chunking for Python."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunk_text(source, chunk_size, chunk_overlap)

    lines = source.splitlines(keepends=True)
    chunks: list[str] = []

    nodes = sorted(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))),
        key=lambda n: n.lineno,
    )

    covered_lines: set[int] = set()
    for node in nodes:
        start = node.lineno - 1
        end = node.end_lineno
        body = "".join(lines[start:end])
        kind = "class" if isinstance(node, ast.ClassDef) else "def"
        prefix = f"# file: {filename} | {kind}: {node.name}\n"
        if len(body) > chunk_size * 3:
            for sc in chunk_text(body, chunk_size, chunk_overlap):
                chunks.append(prefix + sc)
        else:
            chunks.append(prefix + body)
        covered_lines.update(range(start, end))

    uncovered = "".join(
        line for i, line in enumerate(lines) if i not in covered_lines
    ).strip()
    if uncovered:
        prefix = f"# file: {filename} | modulo: top-level\n"
        for sc in chunk_text(uncovered, chunk_size, chunk_overlap):
            chunks.append(prefix + sc)

    return chunks or chunk_text(source, chunk_size, chunk_overlap)


def _chunk_generic_code(source: str, filename: str, lang: str,
                        chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Regex-based semantic chunking for C/C++/Bash."""
    if lang in ("c", "cpp", "h", "hpp"):
        pattern = _RE_TRIPLE
    elif lang == "sh":
        pattern = _RE_BASH_FN
    else:
        return chunk_text(source, chunk_size, chunk_overlap)

    boundaries = [m.start() for m in pattern.finditer(source)]
    if not boundaries:
        return chunk_text(source, chunk_size, chunk_overlap)

    chunks: list[str] = []
    if boundaries[0] > 0:
        preamble = source[:boundaries[0]].strip()
        if preamble:
            prefix = f"# file: {filename} | preamble\n"
            for sc in chunk_text(preamble, chunk_size, chunk_overlap):
                chunks.append(prefix + sc)

    for idx, start_pos in enumerate(boundaries):
        end_pos = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(source)
        body = source[start_pos:end_pos].strip()
        first_line = body.splitlines()[0] if body else ""
        prefix = f"# file: {filename} | function: {first_line[:60]}\n"
        if len(body) > chunk_size * 3:
            for sc in chunk_text(body, chunk_size, chunk_overlap):
                chunks.append(prefix + sc)
        else:
            chunks.append(prefix + body)

    return chunks or chunk_text(source, chunk_size, chunk_overlap)


def chunk_file(file: Path, text: str, mode: str = "default",
               chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Dispatch chunking based on file extension and mode."""
    if mode != "codice":
        return chunk_text(text, chunk_size, chunk_overlap)
    lang = CODE_EXTENSIONS.get(file.suffix.lower())
    if not lang:
        return chunk_text(text, chunk_size, chunk_overlap)
    if lang == "python":
        return _chunk_python(text, file.name, chunk_size, chunk_overlap)
    return _chunk_generic_code(text, file.name, lang, chunk_size, chunk_overlap)


# ═══════════════════════════════════════════════════════════
#  COSINE SIMILARITY
# ═══════════════════════════════════════════════════════════

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ═══════════════════════════════════════════════════════════
#  JSON PERSISTENCE
# ═══════════════════════════════════════════════════════════

def _load_json(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("load_json(%s): %s", path, exc)
    return default


def _save_json(path: Path, data: Any) -> bool:
    """Atomic write via rename."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return True
    except Exception as exc:
        log.warning("save_json(%s): %s", path, exc)
        return False


def load_history(path: Path) -> list[dict]:
    return _load_json(path, [])


def save_history(path: Path, history: list[dict], max_history: int = 20) -> None:
    _save_json(path, history[-(max_history * 2):])


def load_state_file(path: Path) -> dict:
    return _load_json(path, {})


def save_state_file(path: Path, data: dict) -> None:
    _save_json(path, data)


# ═══════════════════════════════════════════════════════════
#  LLM CALL
# ═══════════════════════════════════════════════════════════

def call_llm(ctx: VaultRagContext, messages: list[dict]) -> str:
    """Call the LLM with automatic continuation on truncation."""
    full_response = ""
    current_messages = list(messages)
    max_continuations = 3

    for attempt in range(max_continuations + 1):
        response = ctx.get_llm().chat.completions.create(
            model=ctx.model, messages=current_messages, max_tokens=ctx.max_tokens,
        )
        choice = response.choices[0]
        chunk = choice.message.content or ""
        finish_reason = choice.finish_reason
        full_response += chunk

        if finish_reason != "length" or attempt == max_continuations:
            if attempt > 0:
                log.info("Response completed after %d continuation(s).", attempt)
            break

        log.info("Response truncated (finish_reason=length), continuing... [%d/%d]",
                 attempt + 1, max_continuations)
        current_messages = current_messages + [
            {"role": "assistant", "content": full_response},
            {"role": "user", "content":
                "Continua esattamente da dove ti sei fermato, senza ripetere nulla di quanto scritto."},
        ]

    return strip_emoji(full_response)


# ═══════════════════════════════════════════════════════════
#  VAULT INTERACTION LOG
# ═══════════════════════════════════════════════════════════

def save_interaction(ctx: VaultRagContext, question: str, answer: str, sources: list[str]) -> None:
    try:
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M")
        ds = now.strftime("%Y-%m-%d")
        ts_short = now.strftime("%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9\-]", "-", question[:50])[:40].strip("-")
        chat_dir = ctx.vault_path / ctx.chat_dir
        chat_dir.mkdir(exist_ok=True)
        links = "\n".join(f"- [[{Path(s).stem}]]" for s in sources)
        (chat_dir / f"{ds}-{ts_short}-{slug}.md").write_text(
            f"---\ndata: {ts}\ntipo: chat\n---\n\n"
            f"## Domanda\n{question}\n\n## Risposta\n{answer}\n\n## Fonti\n{links}\n",
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning("save_interaction: %s", exc)


# ═══════════════════════════════════════════════════════════
#  GRAPH RAG
# ═══════════════════════════════════════════════════════════

def load_graph(ctx: VaultRagContext) -> nx.DiGraph:
    if ctx.graph_file.exists():
        try:
            return nx.node_link_graph(
                json.loads(ctx.graph_file.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            log.warning("load_graph: %s", exc)
    return nx.DiGraph()


def load_graph_cached(ctx: VaultRagContext) -> nx.DiGraph:
    with ctx._graph_cache_lock:
        mtime = ctx.graph_file.stat().st_mtime if ctx.graph_file.exists() else 0.0
        if ctx._graph_cache["G"] is None or ctx._graph_cache["mtime"] != mtime:
            ctx._graph_cache["G"] = load_graph(ctx)
            ctx._graph_cache["mtime"] = mtime
        return ctx._graph_cache["G"]


def invalidate_graph_cache(ctx: VaultRagContext) -> None:
    with ctx._graph_cache_lock:
        ctx._graph_cache["G"] = None


def save_graph(ctx: VaultRagContext, G: nx.DiGraph) -> None:
    try:
        ctx.graph_file.write_text(
            json.dumps(nx.node_link_data(G), ensure_ascii=False),
            encoding="utf-8",
        )
        invalidate_graph_cache(ctx)
    except Exception as exc:
        log.warning("save_graph: %s", exc)


def extract_triples_llm(ctx: VaultRagContext, text_sample: str, filename: str) -> list[tuple]:
    prompt = (
        "Estrai le relazioni principali dal testo seguente come lista di triplette.\n"
        "Formato richiesto (una per riga, SOLO questo formato, nient'altro):\n"
        "SOGGETTO | RELAZIONE | OGGETTO\n\n"
        "Regole:\n"
        "- Max 15 triplette\n"
        "- Soggetto e Oggetto: termini chiave brevi (1-4 parole), no frasi\n"
        "- Relazione: verbo o frase breve (es: usa, dipende da, definisce, contiene)\n"
        "- Solo informazioni presenti nel testo\n"
        "- Niente spiegazioni, solo le righe nel formato richiesto\n\n"
        f"File: {filename}\n\nTesto:\n{text_sample[:1200]}"
    )
    try:
        raw = call_llm(ctx, [{"role": "user", "content": prompt}])
        triples: list[tuple] = []
        for line in raw.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3 and all(parts) and all(len(p) < 60 for p in parts):
                s, r, o = parts
                triples.append((s.lower(), r.lower(), o.lower()))
        return triples
    except Exception as exc:
        log.warning("extract_triples_llm(%s): %s", filename, exc)
        return []


def _build_graph_for_file(args: tuple) -> Optional[tuple]:
    ctx, file, already, solo_nuovi = args
    fname = file.name
    if solo_nuovi and fname in already:
        return None
    try:
        col = ctx.get_collection()
        res = col.get(where={"path": str(file)}, include=["documents"])
        if not res["documents"]:
            return None
        docs = res["documents"]
        step = max(1, len(docs) // ctx.graph_chunk_sample)
        sample = "\n\n".join(
            docs[i] for i in range(0, len(docs), step)[:ctx.graph_chunk_sample]
        )
    except Exception:
        return None

    triples = extract_triples_llm(ctx, sample, fname)
    if not triples:
        return ("SKIP", fname, 0)
    return ("OK", fname, len(triples), triples)


def build_graph_from_vault(ctx: VaultRagContext, solo_nuovi: bool = True) -> None:
    if ctx.state.graph_running:
        return
    ctx.state.graph_log = []
    ctx.state.graph_running = True

    def run() -> None:
        try:
            G = load_graph(ctx) if solo_nuovi else nx.DiGraph()
            col = ctx.get_collection()
            if col.count() == 0:
                ctx.state.graph_log.append("ERR: Vault non indicizzato. Esegui prima il reindex.")
                return

            files = vault_files(ctx)[:ctx.graph_max_files]
            ctx.state.graph_log.append(f"Processamento {len(files)} file per il grafo...")
            aggiunti = 0
            already = (
                {fname for _, d in G.nodes(data=True) for fname in d.get("sources", [])}
                if solo_nuovi else set()
            )

            tasks = [(ctx, f, already, solo_nuovi) for f in files]
            with ThreadPoolExecutor(max_workers=ctx.graph_max_workers) as pool:
                for result in as_completed(pool.map(_build_graph_for_file, tasks)):
                    r = result.result()
                    if r is None:
                        continue
                    if r[0] == "SKIP":
                        ctx.state.graph_log.append(f"SKIP: {r[1]} (nessuna tripletta)")
                        continue
                    fname = r[1]
                    triples = r[3]
                    for s, rel, o in triples:
                        for node in (s, o):
                            if not G.has_node(node):
                                G.add_node(node, sources=[], weight=1)
                        G.nodes[s]["weight"] = G.nodes[s].get("weight", 1) + 1
                        srcs = G.nodes[s].get("sources", [])
                        if fname not in srcs:
                            srcs.append(fname)
                        G.nodes[s]["sources"] = srcs
                        if G.has_edge(s, o):
                            G[s][o]["weight"] = G[s][o].get("weight", 1) + 1
                            rels = G[s][o].get("relations", [])
                            if rel not in rels:
                                rels.append(rel)
                            G[s][o]["relations"] = rels
                        else:
                            G.add_edge(s, o, relations=[rel], weight=1, file=fname)
                    aggiunti += len(triples)
                    ctx.state.graph_log.append(f"OK: {fname} ({len(triples)} relazioni)")

            save_graph(ctx, G)
            ctx.state.graph_log.append(
                f"Grafo completato. Nodi: {G.number_of_nodes()}, "
                f"Archi: {G.number_of_edges()}, Nuove relazioni: {aggiunti}"
            )
        except Exception as exc:
            ctx.state.graph_log.append(f"Errore critico: {exc}")
            log.exception("build_graph_from_vault")
        finally:
            ctx.state.graph_running = False

    threading.Thread(target=run, daemon=True, name="graph-builder").start()


def graph_expand_query(ctx: VaultRagContext, question: str, G: nx.DiGraph,
                       top_n: int = 5) -> tuple[list, list, str]:
    if G.number_of_nodes() == 0:
        return [], [], ""
    q_tokens = set(_RE_WORD.findall(question.lower()))
    scored: list[tuple[int, str]] = []
    for node in G.nodes():
        overlap = len(q_tokens & set(_RE_WORD.findall(node)))
        if overlap:
            scored.append((overlap * G.nodes[node].get("weight", 1), node))
    scored.sort(reverse=True)
    seeds = [n for _, n in scored[:top_n]]
    if not seeds:
        return [], [], ""
    expanded = set(seeds)
    for n in seeds:
        expanded.update(G.successors(n))
        expanded.update(G.predecessors(n))
    sources = list(dict.fromkeys(
        s for n in expanded for s in G.nodes[n].get("sources", [])
    ))[:6]
    lines: list[str] = []
    for node in seeds:
        for _, nb, d in G.out_edges(node, data=True):
            lines.append(f"- {node} {d.get('relations', ['→'])[0]} {nb}")
        for pr, _, d in G.in_edges(node, data=True):
            lines.append(f"- {pr} {d.get('relations', ['→'])[0]} {node}")
    return list(expanded), sources, "\n".join(lines[:20])


# ═══════════════════════════════════════════════════════════
#  RAG QUERY
# ═══════════════════════════════════════════════════════════

def _parse_cartelle_from_context(ctx_text: str) -> list[str]:
    return [m.group(1).strip() for m in _RE_CART.finditer(ctx_text)]


def _vector_query(ctx: VaultRagContext, question: str,
                  q_embed: Optional[list[float]] = None,
                  n: Optional[int] = None) -> tuple[list, list]:
    """Vector search with optional language and folder filtering."""
    collection = ctx.get_collection()
    if n is None:
        n = ctx.n_results
    if q_embed is None:
        q_embed = ctx.cached_encode(question).tolist()

    state = ctx.state
    lang = state.lang
    ext_filter = LANG_EXT_MAP.get(lang) if lang != "auto" else None
    materia_ctx = state.materia_context

    need_filter = bool(ext_filter) or bool(materia_ctx)
    fetch_n = min(50, collection.count()) if need_filter else min(n, collection.count())

    res = collection.query(query_embeddings=[q_embed], n_results=fetch_n)
    all_docs = res["documents"][0]
    all_metas = res["metadatas"][0]

    if not need_filter:
        return all_docs, all_metas

    docs: list[str] = []
    metas: list[dict] = []
    for doc, meta in zip(all_docs, all_metas):
        if ext_filter and meta.get("type", "") not in ext_filter:
            continue
        if materia_ctx:
            cartelle = _parse_cartelle_from_context(materia_ctx)
            if cartelle:
                file_validi = {
                    str(f) for c in cartelle for f in vault_files(ctx) if c in str(f)
                }
                if file_validi and meta["path"] not in file_validi:
                    continue
        docs.append(doc)
        metas.append(meta)
        if len(docs) >= n:
            break

    if docs:
        return docs, metas

    # Fallback to global search if filter yields no results
    log.info("_vector_query: filter (%s%s) no results, fallback global",
             lang or "", " + materia" if materia_ctx else "")
    return all_docs[:n], all_metas[:n]


def query_hybrid(ctx: VaultRagContext, question: str) -> tuple:
    """Hybrid RAG: vector + graph expansion + corrections."""
    try:
        col = ctx.get_collection()
        if col.count() == 0:
            return None, None, "Vault non indicizzato.", ""

        q_embed = ctx.cached_encode(question).tolist()
        docs, metas = _vector_query(ctx, question, q_embed=q_embed)

        # Corrections
        correction_docs = query_corrections(ctx, question, q_embed=q_embed, n=2)
        if correction_docs:
            corr_metas = [
                {"file": "[correzione]", "path": "", "chunk": 0,
                 "total": 1, "type": "correction"}
            ] * len(correction_docs)
            docs = correction_docs + docs
            metas = corr_metas + metas

        # Graph expansion
        G = load_graph_cached(ctx)
        _, graph_sources, relation_text = graph_expand_query(ctx, question, G)

        graph_docs: list[str] = []
        vector_files = {m["file"] for m in metas}
        for fname in graph_sources:
            if fname in vector_files or len(graph_docs) >= 2:
                continue
            try:
                sub = col.query(
                    query_embeddings=[q_embed], n_results=1, where={"file": fname}
                )
                if sub["documents"][0]:
                    graph_docs.append(sub["documents"][0][0])
            except Exception:
                pass

        return docs + graph_docs, metas, None, relation_text

    except Exception as exc:
        log.exception("query_hybrid crashed")
        return None, None, f"Errore interno RAG: {exc}", ""


# ═══════════════════════════════════════════════════════════
#  CORRECTIONS / FEEDBACK
# ═══════════════════════════════════════════════════════════

def load_corrections(path: Path) -> list[dict]:
    return _load_json(path, [])


def save_corrections(path: Path, corrections: list[dict], max_corrections: int = 500) -> None:
    _save_json(path, corrections[-max_corrections:])


def load_error_memory(path: Path) -> list[dict]:
    return _load_json(path, [])


def save_error_memory(path: Path, memory: list[dict], max_corrections: int = 500) -> None:
    _save_json(path, memory[-(max_corrections * 2):])


def _get_correction_embedding(ctx: VaultRagContext, cid: str, question: str) -> np.ndarray:
    with ctx._corrections_emb_lock:
        if cid not in ctx._corrections_emb_cache:
            ctx._corrections_emb_cache[cid] = ctx.get_embedder().encode(
                question, convert_to_numpy=True
            )
        return ctx._corrections_emb_cache[cid]


def add_correction(ctx: VaultRagContext, question: str, wrong_answer: str,
                   correct_answer: str, sources: list[str]) -> None:
    ts = datetime.now().isoformat()
    cid = f"correction::{hash(question + ts)}"

    corrections = load_corrections(ctx.corrections_file)
    corrections.append({
        "id": cid, "ts": ts, "question": question,
        "wrong_answer": wrong_answer, "correct_answer": correct_answer, "sources": sources,
    })
    save_corrections(ctx.corrections_file, corrections, ctx.max_corrections)

    correction_text = (
        f"DOMANDA: {question}\n"
        f"RISPOSTA CORRETTA: {correct_answer}\n"
        f"NOTA: questa è la risposta verificata dall'utente."
    )
    try:
        emb = ctx.cached_encode(correction_text).tolist()
        ctx.get_corrections_collection().add(
            ids=[cid], embeddings=[emb], documents=[correction_text],
            metadatas=[{
                "type": "correction", "question": question[:200],
                "ts": ts, "sources": json.dumps(sources),
            }],
        )
        log.info("Correction indexed: %s", cid)
    except Exception as exc:
        log.warning("add_correction ChromaDB: %s", exc)

    error_memory = load_error_memory(ctx.error_memory_file)
    error_memory.append({
        "ts": ts, "question": question,
        "wrong_answer": wrong_answer[:300], "correct_answer": correct_answer[:600],
    })
    save_error_memory(ctx.error_memory_file, error_memory, ctx.max_corrections)

    with ctx._corrections_emb_lock:
        ctx._corrections_emb_cache.pop(cid, None)


def build_error_context(ctx: VaultRagContext, question: str,
                        q_emb: Optional[np.ndarray] = None) -> str:
    """Find semantically similar corrections and inject into prompt."""
    try:
        corrections = load_corrections(ctx.corrections_file)
        if not corrections:
            return ""
        if q_emb is None:
            q_emb = ctx.cached_encode(question)
        scored: list[tuple[float, dict]] = []
        for c in corrections[-100:]:
            try:
                c_emb = _get_correction_embedding(ctx, c["id"], c["question"])
                sim = cosine_sim(q_emb, c_emb)
                if sim > 0.55:
                    scored.append((sim, c))
            except Exception:
                continue
        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[:3]
        if not top:
            return ""
        lines = ["CORREZIONI DA ERRORI PRECEDENTI (segui queste indicazioni con priorità):"]
        for _, c in top:
            lines.append(
                f"- Domanda simile: \"{c['question'][:80]}\"\n"
                f"  Risposta corretta: {c['correct_answer'][:300]}"
            )
        return "\n".join(lines) + "\n\n"
    except Exception as exc:
        log.warning("build_error_context: %s", exc)
        return ""


def query_corrections(ctx: VaultRagContext, question: str,
                      q_embed: Optional[list[float]] = None, n: int = 2) -> list[str]:
    """ChromaDB search for relevant corrections."""
    try:
        col = ctx.get_corrections_collection()
        if col.count() == 0:
            return []
        if q_embed is None:
            q_embed = ctx.cached_encode(question).tolist()
        res = col.query(query_embeddings=[q_embed], n_results=min(n, col.count()))
        return res["documents"][0]
    except Exception as exc:
        log.warning("query_corrections: %s", exc)
        return []


# ═══════════════════════════════════════════════════════════
#  SYSTEM PROMPT BUILDER
# ═══════════════════════════════════════════════════════════

def build_system_prompt(materie_attive: dict[str, str]) -> str:
    nomi = list(materie_attive.keys())
    if not nomi:
        return SYSTEM_PROMPT_DEFAULT
    base = f"Sei un assistente specializzato in {', '.join(nomi)}. "
    extra = [LANG_SPECIFIC_PROMPTS[n] for n in nomi if n in LANG_SPECIFIC_PROMPTS]
    if extra:
        base += " ".join(extra) + " "
    base += (
        "Rispondi in base ai file del vault con un focus su queste materie. "
        "Se la risposta non è nei file, dillo. Non inventare. Rispondi in italiano."
    )
    return base


# ═══════════════════════════════════════════════════════════
#  DUPLICATE DETECTION
# ═══════════════════════════════════════════════════════════

def find_duplicates(ctx: VaultRagContext) -> list[dict]:
    col = ctx.get_collection()
    files = vault_files(ctx)
    embs: dict[str, np.ndarray] = {}
    for file in files:
        try:
            res = col.get(
                where={"$and": [{"path": {"$eq": str(file)}}, {"chunk": {"$eq": 0}}]},
                include=["embeddings"],
            )
            if res["embeddings"]:
                v = np.array(res["embeddings"][0], dtype=np.float32)
                norm = np.linalg.norm(v)
                if norm > 0:
                    embs[str(file)] = v / norm
        except Exception:
            continue

    paths = list(embs.keys())
    found: list[dict] = []
    if paths:
        mat = np.stack([embs[p] for p in paths])
        sim_matrix = mat @ mat.T
        n = len(paths)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim >= ctx.dup_threshold:
                    found.append({
                        "file1": Path(paths[i]).name,
                        "file2": Path(paths[j]).name,
                        "sim": round(sim, 3),
                    })
    return found
