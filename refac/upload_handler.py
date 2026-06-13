#!/usr/bin/env python3
"""
VaultRAG — Upload Handler Blueprint
RAG temporaneo su file caricati al volo (nessuna persistenza).

Refactored: accepts VaultRagContext instead of raw globals dict.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

import numpy as np
from flask import Blueprint, jsonify, request

from rag_core import (
    VaultRagContext,
    call_llm,
    chunk_text,
    cosine_sim,
    extract_text,
    strip_emoji,
)

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXT = {
    ".md", ".txt", ".pdf", ".docx", ".epub",
    ".odt", ".ods", ".html", ".htm",
}
ALLOWED_MIME = {
    "text/plain", "text/markdown", "text/html",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/epub+zip",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.oasis.opendocument.spreadsheet",
}
UPLOAD_CHUNK_SIZE = 500
UPLOAD_CHUNK_OVERLAP = 50
UPLOAD_N_RESULTS = 3
UPLOAD_MAX_CONTINUATIONS = 3

log = logging.getLogger("vaultrag.upload")

# ═══════════════════════════════════════════════════════════
#  BLUEPRINT
# ═══════════════════════════════════════════════════════════
upload_bp = Blueprint("upload", __name__)

# Dependency injection: set at startup
_ctx: Optional[VaultRagContext] = None


def init(ctx: VaultRagContext) -> None:
    """Inject the VaultRagContext into this module."""
    global _ctx
    _ctx = ctx


# ═══════════════════════════════════════════════════════════
#  SAFETY
# ═══════════════════════════════════════════════════════════

def _sanitize_filename(name: str) -> str:
    """Remove path traversal and dangerous characters."""
    name = Path(name).name
    name = re.sub(r"[^\w\.\-\s]", "_", name)
    name = re.sub(r"\.{2,}", ".", name)
    name = name.strip("._- ")
    return name or "uploaded_file"


def _validate_file(file_storage: Any) -> tuple:
    """
    Validate uploaded file.
    Returns (ok, error_message, sanitized_name, extension)
    """
    if not file_storage or not file_storage.filename:
        return False, "Nessun file selezionato.", "", ""

    raw_name = file_storage.filename
    ext = Path(raw_name).suffix.lower()

    if ext not in ALLOWED_EXT:
        return False, f"Formato non supportato: {ext}. Consentiti: {', '.join(sorted(ALLOWED_EXT))}", "", ext

    # MIME type check (best-effort, non-blocking)
    mime = file_storage.content_type or ""
    if mime and mime.split(";")[0].strip() not in ALLOWED_MIME:
        log.warning("Unexpected MIME type: %s for %s", mime, raw_name)

    # Size check
    file_storage.seek(0, 2)
    size = file_storage.tell()
    file_storage.seek(0)

    if size > MAX_FILE_SIZE:
        mb = size / (1024 * 1024)
        return False, f"File troppo grande: {mb:.1f} MB (max 50 MB).", "", ext

    safe_name = _sanitize_filename(raw_name)
    return True, "", safe_name, ext


# ═══════════════════════════════════════════════════════════
#  LLM WRAPPER (uses context)
# ═══════════════════════════════════════════════════════════

def _call_llm_upload(ctx: VaultRagContext, messages: list[dict]) -> str:
    """Call LLM with automatic continuation for upload flow."""
    return call_llm(ctx, messages)


# ═══════════════════════════════════════════════════════════
#  ENDPOINT
# ═══════════════════════════════════════════════════════════

@upload_bp.route("/api/upload", methods=["POST"])
def api_upload() -> Any:
    """
    Temporary RAG on uploaded file.
    Does not write to disk (except temp dir with cleanup).
    Does not touch ChromaDB, vault, or history.
    """
    if _ctx is None:
        return jsonify({"error": "Upload handler non inizializzato."}), 500

    t0 = time.time()

    # ── 1. Validation ───────────────────────────────────
    file_storage = request.files.get("file")
    ok, err, safe_name, ext = _validate_file(file_storage)
    if not ok:
        return jsonify({"error": err}), 400

    question = (request.form.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Inserire una domanda."}), 400

    # ── 2. Save to temp dir ─────────────────────────────
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="vaultrag_upload_")
        os.close(fd)
        file_storage.save(tmp_path)

        # ── 3. Extract text ──────────────────────────────
        text = extract_text(Path(tmp_path), ocr_langs=_ctx.ocr_langs)
        if not text.strip():
            return jsonify({"error": "Nessun testo estratto dal file."}), 400

        log.info("Upload: %s — %d characters extracted", safe_name, len(text))

        # ── 4. Chunking ──────────────────────────────────
        chunks = chunk_text(text, UPLOAD_CHUNK_SIZE, UPLOAD_CHUNK_OVERLAP)
        if not chunks:
            return jsonify({"error": "File troppo piccolo per il retrieval."}), 400

        # ── 5. Embedding chunks ──────────────────────────
        t_emb = time.time()
        q_emb = _ctx.get_embedder().encode(question, convert_to_numpy=True)
        c_embs = [_ctx.get_embedder().encode(c, convert_to_numpy=True) for c in chunks]
        emb_time = time.time() - t_emb

        # ── 6. Ranking by similarity ────────────────────
        scores = [(float(cosine_sim(q_emb, ce)), i) for i, ce in enumerate(c_embs)]
        scores.sort(reverse=True)
        top = scores[:UPLOAD_N_RESULTS]
        best_score = top[0][0] if top else 0

        context_text = "\n\n---\n\n".join(chunks[i] for _, i in top)

        # ── 7. LLM ───────────────────────────────────────
        t_llm = time.time()
        system_prompt = (
            "Rispondi in italiano basandoti SOLO sul testo fornito dal file uploadato. "
            "Sii conciso. Se la risposta non e' nel testo, dillo chiaramente. "
            "Formatta in markdown: **grassetto** per termini chiave, "
            "`codice` per comandi, ``` per blocchi codice. Non usare emoji."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":
                f"FILE: {safe_name}\n\n"
                f"TESTO DEL FILE (chunk piu' rilevanti):\n\n{context_text}\n\n"
                f"DOMANDA: {question}"},
        ]
        answer = _call_llm_upload(_ctx, messages)
        llm_time = time.time() - t_llm

        total_time = time.time() - t0

        log.info("Upload: %s — Retrieval: %.1fs | LLM: %.1fs | Totale: %.1fs | "
                 "Chunk: %d | Best score: %.3f",
                 safe_name, emb_time, llm_time, total_time, len(chunks), best_score)

        return jsonify({
            "answer":      answer,
            "file":        safe_name,
            "chunks":      len(chunks),
            "top_chunks":  UPLOAD_N_RESULTS,
            "best_score":  round(best_score, 3),
            "timing": {
                "embedding": round(emb_time, 2),
                "llm":       round(llm_time, 2),
                "total":     round(total_time, 2),
            },
        })

    except Exception as exc:
        log.exception("Upload error for %s", safe_name)
        return jsonify({"error": f"Errore interno: {exc}"}), 500

    finally:
        # ── 8. Cleanup temp file (guaranteed) ────────────
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
