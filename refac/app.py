#!/usr/bin/env python3
"""
VaultRAG Web App — Flask backend
Refactored: routes only, all RAG logic in rag_core.py, DI via VaultRagContext.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template_string, request

from rag_core import (
    AppState,
    VaultRagContext,
    build_error_context,
    build_graph_from_vault,
    build_system_prompt,
    call_llm,
    chunk_file,
    extract_text,
    find_duplicates,
    invalidate_graph_cache,
    invalidate_vault_cache,
    load_error_memory,
    load_graph_cached,
    load_history,
    load_state_file,
    save_error_memory,
    save_history,
    save_interaction,
    save_state_file,
    strip_emoji,
    vault_files,
    # Corrections
    add_correction,
    load_corrections,
    query_corrections,
    save_corrections,
)

# ── ENV ───────────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

# ── LOGGING ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vaultrag")

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
VAULT_PATH = Path(os.getenv("VAULT_PATH", "~/obsidian_notes/Notes")).expanduser()
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DB_PATH = Path.home() / ".vault_rag_db"
CHAT_DIR = "_chat"
CONTEXT_DIR = "_context"
USER_CONTEXT_FILE = VAULT_PATH / "_context/user_context.md"
HISTORY_FILE = Path.home() / ".vault_rag_history.json"
STATE_FILE = Path.home() / ".vault_rag_state.json"
CORRECTIONS_FILE = Path.home() / ".vault_rag_corrections.json"
ERROR_MEMORY_FILE = Path.home() / ".vault_rag_error_memory.json"
GRAPH_FILE = Path.home() / ".vault_rag_graph.json"

if not API_KEY:
    log.warning("OPENROUTER_API_KEY non impostata.")

# ═══════════════════════════════════════════════════════════
#  CONTEXT (single DI container for the whole app)
# ═══════════════════════════════════════════════════════════

ctx = VaultRagContext(
    vault_path=VAULT_PATH,
    db_path=DB_PATH,
    history_file=HISTORY_FILE,
    state_file=STATE_FILE,
    corrections_file=CORRECTIONS_FILE,
    error_memory_file=ERROR_MEMORY_FILE,
    graph_file=GRAPH_FILE,
    chat_dir=CHAT_DIR,
    context_dir=CONTEXT_DIR,
    api_key=API_KEY,
    model="qwen-plus",
    max_tokens=8192,
    state=AppState(),
)

# ═══════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════
app = Flask(__name__)

# ── Upload Handler Blueprint ──────────────────────────────
from upload_handler import upload_bp, init as upload_init
app.register_blueprint(upload_bp)
upload_init(ctx)

# ── Load user context at startup ──────────────────────────
def _load_user_context() -> None:
    if USER_CONTEXT_FILE.exists():
        try:
            ctx.state.user_context = USER_CONTEXT_FILE.read_text(encoding="utf-8")
            log.info("user_context.md loaded (%d chars)", len(ctx.state.user_context))
        except Exception as exc:
            log.warning("load_user_context: %s", exc)
    else:
        log.info("user_context.md not found, ignored.")

_load_user_context()

# ── HTML template ─────────────────────────────────────────
_FRONTEND_PATH = Path(__file__).parent / "frontend.html"
_HTML_TEMPLATE = _FRONTEND_PATH.read_text(encoding="utf-8")


# ═══════════════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/")
def index() -> str:
    return render_template_string(_HTML_TEMPLATE)


@app.route("/api/stats")
def api_stats() -> Any:
    col = ctx.get_collection()
    hist = load_history(ctx.history_file)
    return jsonify({
        "chunks":        col.count(),
        "files":         len(vault_files(ctx)),
        "history":       len(hist) // 2,
        "model":         ctx.model,
        "lang":          ctx.state.lang,
        "materia":       ctx.state.materia or "nessuna",
        "materie_lista": list(ctx.state.materie_attive.keys()),
        "mode":          ctx.state.mode,
        "user_context":  bool(ctx.state.user_context),
    })


# ── Language filter ──────────────────────────────────────

@app.route("/api/lang", methods=["GET", "POST"])
def api_lang() -> Any:
    if request.method == "GET":
        return jsonify({"lang": ctx.state.lang})
    target = (request.json or {}).get("lang", "auto").strip()
    valid = ("auto", "python", "c", "cpp", "bash")
    if target in valid:
        ctx.state.lang = target
        return jsonify({"ok": True, "lang": ctx.state.lang})
    return jsonify({"error": f"Linguaggio non valido. Usa: {', '.join(valid)}"}), 400


# ── Chat ──────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def api_chat() -> Any:
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Domanda vuota"}), 400

    t0 = time.time()
    docs, metas, err, relation_text = _query_hybrid(question)
    t1 = time.time()

    if err:
        return jsonify({"error": err}), 400
    if not docs:
        return jsonify({"answer": "Nessun file rilevante trovato.", "sources": [], "graph_used": False})

    q_emb = ctx.cached_encode(question)
    user_ctx = f"CONTESTO UTENTE:\n{ctx.state.user_context}\n\n" if ctx.state.user_context else ""
    materia_ctx = f"CONTESTO MATERIE ATTIVE:\n{ctx.state.materia_context}\n\n" if ctx.state.materia_context else ""
    graph_ctx = f"RELAZIONI DAL KNOWLEDGE GRAPH:\n{relation_text}\n\n" if relation_text else ""
    error_ctx = build_error_context(ctx, question, q_emb=q_emb)
    lang_ctx = ""
    if ctx.state.lang != "auto":
        from rag_core import LANG_CODE_PROMPTS
        lang_ctx = LANG_CODE_PROMPTS.get(ctx.state.lang, "")
    lang_prefix = f"{lang_ctx}\n\n" if lang_ctx else ""
    context_text = "\n\n---\n\n".join(docs)

    messages = [
        {"role": "system", "content": ctx.state.system_prompt},
        {"role": "user",   "content":
            f"{lang_prefix}{error_ctx}{user_ctx}{materia_ctx}{graph_ctx}"
            f"FILE DAL MIO VAULT:\n\n{context_text}\n\nDOMANDA: {question}"},
    ]

    t2 = time.time()
    try:
        answer = call_llm(ctx, messages)
    except Exception as exc:
        log.exception("api_chat LLM error")
        return jsonify({"error": str(exc)}), 500
    t3 = time.time()

    log.info("Timing — Retrieval: %.1fs | LLM: %.1fs | Totale: %.1fs",
             t1 - t0, t3 - t2, t3 - t0)

    history = load_history(ctx.history_file)
    history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    save_history(ctx.history_file, history, ctx.max_history)
    save_interaction(ctx, question, answer, [m["path"] for m in metas])

    return jsonify({
        "answer":     answer,
        "sources":    [{"file": m["file"], "chunk": m["chunk"], "total": m["total"] - 1} for m in metas],
        "graph_used": bool(relation_text),
    })


def _query_hybrid(question: str) -> tuple:
    """Wrapper around rag_core.query_hybrid."""
    from rag_core import query_hybrid as _qh
    return _qh(ctx, question)


# ── Open file ─────────────────────────────────────────────

@app.route("/api/open", methods=["POST"])
def api_open() -> Any:
    nome = (request.json or {}).get("nome", "").strip()
    domanda = (request.json or {}).get("domanda", "riassumi questo file").strip()

    matches = [f for f in vault_files(ctx) if nome.lower() in f.name.lower()]
    if not matches:
        chat_dir = ctx.vault_path / ctx.chat_dir
        if chat_dir.exists():
            matches = [f for f in chat_dir.glob("**/*") if nome.lower() in f.name.lower()]
    if not matches:
        return jsonify({"error": f"File non trovato: {nome}"}), 404

    file = matches[0]
    collection = ctx.get_collection()
    result = collection.get(where={"path": str(file)}, include=["documents", "embeddings", "metadatas"])
    if not result["ids"]:
        return jsonify({"error": "File non indicizzato. Esegui reindex."}), 404

    q_emb = ctx.cached_encode(domanda)
    from rag_core import cosine_sim
    scores = [(cosine_sim(q_emb, np.array(emb)), i) for i, emb in enumerate(result["embeddings"])]
    scores.sort(reverse=True)
    top = scores[:ctx.n_results]
    context_text = "\n\n---\n\n".join(result["documents"][i] for _, i in top)

    user_ctx = f"CONTESTO UTENTE:\n{ctx.state.user_context}\n\n" if ctx.state.user_context else ""
    try:
        answer = call_llm(ctx, [
            {"role": "system", "content": ctx.state.system_prompt},
            {"role": "user",   "content": f"{user_ctx}FILE:\n{context_text}\n\nDOMANDA: {domanda}"},
        ])
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    history = load_history(ctx.history_file)
    history += [
        {"role": "user",      "content": f"apri {nome} | {domanda}"},
        {"role": "assistant", "content": answer},
    ]
    save_history(ctx.history_file, history, ctx.max_history)
    return jsonify({"answer": answer, "file": file.name, "chunks": [i for _, i in top]})


# ── Search ────────────────────────────────────────────────

@app.route("/api/search", methods=["POST"])
def api_search() -> Any:
    termine = (request.json or {}).get("termine", "").strip()
    if not termine:
        return jsonify({"by_name": [], "by_content": []})
    by_name = [{"file": f.name, "cartella": f.parent.name}
               for f in vault_files(ctx) if termine.lower() in f.name.lower()]
    col = ctx.get_collection()
    q_emb = ctx.cached_encode(termine).tolist()
    res = col.query(query_embeddings=[q_emb], n_results=min(8, col.count()))
    by_cont = list(dict.fromkeys(m["file"] for m in res["metadatas"][0]))
    return jsonify({"by_name": by_name, "by_content": by_cont})


# ── File info ─────────────────────────────────────────────

@app.route("/api/info", methods=["POST"])
def api_info() -> Any:
    nome = (request.json or {}).get("nome", "").strip()
    matches = [f for f in vault_files(ctx) if nome.lower() in f.name.lower()]
    if not matches:
        return jsonify({"error": f"File non trovato: {nome}"}), 404
    file = matches[0]
    stat = file.stat()
    col = ctx.get_collection()
    chunks = col.get(where={"path": str(file)})["ids"]
    info: dict[str, Any] = {
        "file":          file.name,
        "cartella":      str(file.parent.relative_to(ctx.vault_path)),
        "tipo":          file.suffix,
        "dimensione_kb": round(stat.st_size / 1024, 1),
        "modificato":    datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        "chunk":         len(chunks),
    }
    if file.suffix == ".pdf":
        try:
            import fitz
            info["pagine"] = len(fitz.open(str(file)))
        except Exception:
            pass
    return jsonify(info)


# ── Duplicates ────────────────────────────────────────────

@app.route("/api/duplicati")
def api_duplicati() -> Any:
    found = find_duplicates(ctx)
    return jsonify({"duplicati": found, "soglia": ctx.dup_threshold})


# ── History ───────────────────────────────────────────────

@app.route("/api/history")
def api_history() -> Any:
    n = int(request.args.get("n", 10))
    h = load_history(ctx.history_file)
    scambi = [{"q": h[i]["content"], "a": h[i+1]["content"]} for i in range(0, len(h) - 1, 2)]
    return jsonify({"history": scambi[-n:]})


@app.route("/api/reset", methods=["POST"])
def api_reset() -> Any:
    if ctx.history_file.exists():
        ctx.history_file.unlink()
    return jsonify({"ok": True})


# ── Feedback / Corrections ────────────────────────────────

@app.route("/api/feedback", methods=["POST"])
def api_feedback() -> Any:
    data = request.json or {}
    question = data.get("question", "").strip()
    wrong_answer = data.get("wrong_answer", "").strip()
    correct_answer = data.get("correct_answer", "").strip()
    sources = data.get("sources", [])
    if not question or not correct_answer:
        return jsonify({"error": "Campi 'question' e 'correct_answer' obbligatori."}), 400
    try:
        add_correction(ctx, question, wrong_answer, correct_answer, sources)
        return jsonify({"ok": True, "msg": "Correzione salvata e indicizzata. Il RAG imparerà da questo errore."})
    except Exception as exc:
        log.exception("api_feedback")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/feedback/list")
def api_feedback_list() -> Any:
    n = int(request.args.get("n", 20))
    corrections = load_corrections(ctx.corrections_file)
    return jsonify({"corrections": corrections[-n:][::-1], "total": len(corrections)})


@app.route("/api/feedback/delete", methods=["POST"])
def api_feedback_delete() -> Any:
    cid = (request.json or {}).get("id", "").strip()
    corrections = load_corrections(ctx.corrections_file)
    before = len(corrections)
    corrections = [c for c in corrections if c.get("id") != cid]
    save_corrections(ctx.corrections_file, corrections, ctx.max_corrections)
    with ctx._corrections_emb_lock:
        ctx._corrections_emb_cache.pop(cid, None)
    try:
        ctx.get_corrections_collection().delete(ids=[cid])
    except Exception:
        pass
    return jsonify({"ok": True, "removed": before - len(corrections)})


@app.route("/api/feedback/reset", methods=["POST"])
def api_feedback_reset() -> Any:
    for f in (ctx.corrections_file, ctx.error_memory_file):
        if f.exists():
            f.unlink()
    with ctx._corrections_emb_lock:
        ctx._corrections_emb_cache.clear()
    try:
        ctx.get_chroma().delete_collection("rag_corrections")
    except Exception:
        pass
    return jsonify({"ok": True})


# ── Mode ──────────────────────────────────────────────────

@app.route("/api/mode", methods=["GET", "POST"])
def api_mode() -> Any:
    if request.method == "GET":
        return jsonify({"mode": ctx.state.mode})
    target = (request.json or {}).get("mode", "").strip()
    if ctx.state.set_mode(target):
        return jsonify({"ok": True, "mode": ctx.state.mode, "msg": f"Modalita' {ctx.state.mode} attivata."})
    return jsonify({"error": "Modalita' non valida. Usa 'default' o 'codice'."}), 400


# ── Model Switcher API ────────────────────────────────────

@app.route("/api/model", methods=["GET"])
def api_model_list() -> Any:
    from model_switcher import list_available_models
    return jsonify({"models": list_available_models()})


@app.route("/api/model/current", methods=["GET"])
def api_model_current() -> Any:
    from model_switcher import get_current_model
    return jsonify(get_current_model(ctx))


@app.route("/api/model/change", methods=["POST"])
def api_model_change() -> Any:
    from model_switcher import change_model
    data = request.json or {}
    model_name = data.get("model", "").strip()
    if not model_name:
        return jsonify({"error": "Parametro 'model' obbligatorio"}), 400

    api_key = data.get("api_key")
    base_url = data.get("base_url")

    return jsonify(change_model(ctx, model_name, api_key=api_key, base_url=base_url))


@app.route("/api/model/test", methods=["POST"])
def api_model_test() -> Any:
    import time
    start = time.time()
    try:
        response = ctx.get_llm().chat.completions.create(
            model=ctx.model,
            messages=[{"role": "user", "content": "Rispondi con una sola parola: ok"}],
            max_tokens=10,
        )
        elapsed = time.time() - start
        text = response.choices[0].message.content or ""
        return jsonify({
            "status": "ok",
            "response": text.strip(),
            "latency_seconds": round(elapsed, 2),
            "model": ctx.model,
        })
    except Exception as exc:
        elapsed = time.time() - start
        log.error(f"Test modello fallito: {exc}")
        return jsonify({
            "status": "error",
            "error": str(exc),
            "model": ctx.model,
            "latency_seconds": round(elapsed, 2),
        }), 500


# ── Export ────────────────────────────────────────────────

@app.route("/api/esporta", methods=["POST"])
def api_esporta() -> Any:
    history = load_history(ctx.history_file)
    if not history:
        return jsonify({"error": "Nessuna conversazione da esportare."}), 400
    try:
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M")
        ds = now.strftime("%Y-%m-%d-%H%M")
        chat_dir = ctx.vault_path / ctx.chat_dir
        chat_dir.mkdir(exist_ok=True)
        filename = chat_dir / f"export-{ds}.md"
        scambi = [(history[i], history[i+1]) for i in range(0, len(history) - 1, 2)]
        lines = [f"---\ndata: {ts}\ntipo: export\n---\n"]
        for idx, (q, a) in enumerate(scambi, 1):
            lines.append(
                f"## [{idx}] {q['content'][:60]}\n\n"
                f"**Domanda:** {q['content']}\n\n**Risposta:** {a['content']}\n\n---\n"
            )
        filename.write_text("\n".join(lines), encoding="utf-8")
        return jsonify({"ok": True, "file": filename.name})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/esportazioni")
def api_esportazioni() -> Any:
    chat_dir = ctx.vault_path / ctx.chat_dir
    if not chat_dir.exists():
        return jsonify({"esportazioni": []})
    exports = sorted(chat_dir.glob("export-*.md"), reverse=True)
    return jsonify({"esportazioni": [f.name for f in exports]})


# ── File list ─────────────────────────────────────────────

@app.route("/api/lista-file")
def api_lista_file() -> Any:
    files = vault_files(ctx)
    return jsonify({"files": sorted(f.name for f in files), "totale": len(files)})


@app.route("/api/cartelle")
def api_cartelle() -> Any:
    dirs = sorted(set(f.parent.name for f in vault_files(ctx)))
    return jsonify({"cartelle": dirs})


@app.route("/api/numero-file")
def api_numero_file() -> Any:
    col = ctx.get_collection()
    files = vault_files(ctx)
    st = load_state_file(ctx.state_file)
    nuovi = sum(1 for f in files if st.get(str(f)) != str(f.stat().st_mtime))
    return jsonify({"vault": len(files), "chunk_db": col.count(), "non_indicizzati": nuovi})


# ── Reindex ───────────────────────────────────────────────

@app.route("/api/reindex", methods=["POST"])
def api_reindex() -> Any:
    if ctx.state.reindex_running:
        return jsonify({"error": "Reindex gia in corso."}), 400
    solo_nuovi = (request.json or {}).get("solo_nuovi", True)
    ctx.state.reindex_log = []
    ctx.state.reindex_running = True

    def run() -> None:
        try:
            col = ctx.get_collection()
            if not solo_nuovi:
                ctx.get_chroma().delete_collection("obsidian_vault")
                col = ctx.get_collection()
                ctx.state.reindex_log.append("Vecchio indice eliminato.")
            files = vault_files(ctx)
            ctx.state.reindex_log.append(f"Trovati {len(files)} file.")
            nuovi = 0
            st = load_state_file(ctx.state_file)
            for file in files:
                mtime = str(file.stat().st_mtime)
                chunk0_id = f"{file}::chunk0"
                if solo_nuovi:
                    already_indexed = bool(col.get(ids=[chunk0_id])["ids"])
                    if already_indexed and st.get(str(file)) == mtime:
                        continue
                    if already_indexed:
                        try:
                            old = col.get(where={"path": str(file)})["ids"]
                            if old:
                                col.delete(ids=old)
                        except Exception:
                            pass
                text = extract_text(file).strip()
                if not text:
                    continue
                chunks = chunk_file(file, text, mode=ctx.state.mode,
                                    chunk_size=ctx.chunk_size, chunk_overlap=ctx.chunk_overlap)
                try:
                    embeds = ctx.get_embedder().encode(
                        chunks, batch_size=64, show_progress_bar=False, convert_to_numpy=True,
                    )
                    chunk_ids = [f"{file}::chunk{n}" for n in range(len(chunks))]
                    metadatas = [
                        {"file": file.name, "path": str(file), "type": file.suffix,
                         "chunk": n, "total": len(chunks)}
                        for n in range(len(chunks))
                    ]
                    col.add(ids=chunk_ids, embeddings=embeds.tolist(), documents=chunks, metadatas=metadatas)
                    nuovi += len(chunks)
                    ctx.state.reindex_log.append(f"OK: {file.name} ({len(chunks)} chunk)")
                    st[str(file)] = mtime
                except Exception as exc:
                    ctx.state.reindex_log.append(f"ERR: {file.name}: {exc}")
            save_state_file(ctx.state_file, st)
            invalidate_vault_cache(ctx)
            ctx.state.reindex_log.append(f"Completato. Nuovi chunk: {nuovi}. Totale DB: {col.count()}")
        except Exception as exc:
            ctx.state.reindex_log.append(f"Errore: {exc}")
            log.exception("reindex")
        finally:
            ctx.state.reindex_running = False

    threading.Thread(target=run, daemon=True, name="reindexer").start()
    return jsonify({"ok": True})


@app.route("/api/reindex-log")
def api_reindex_log() -> Any:
    return jsonify({"log": ctx.state.reindex_log, "running": ctx.state.reindex_running})


# ── Graph routes ──────────────────────────────────────────

@app.route("/api/graph/build", methods=["POST"])
def api_graph_build() -> Any:
    if ctx.state.graph_running:
        return jsonify({"error": "Build grafo gia in corso."}), 400
    solo_nuovi = (request.json or {}).get("solo_nuovi", True)
    build_graph_from_vault(ctx, solo_nuovi=solo_nuovi)
    return jsonify({"ok": True})


@app.route("/api/graph/log")
def api_graph_log() -> Any:
    return jsonify({"log": ctx.state.graph_log, "running": ctx.state.graph_running})


@app.route("/api/graph/stats")
def api_graph_stats() -> Any:
    G = load_graph_cached(ctx)
    if G.number_of_nodes() == 0:
        return jsonify({"nodes": 0, "edges": 0, "components": 0, "top_nodes": [], "exists": False})
    try:
        import networkx as nx
        components = nx.number_weakly_connected_components(G)
    except Exception:
        components = 0
    top = sorted(G.nodes(data=True), key=lambda x: x[1].get("weight", 1), reverse=True)[:10]
    top_nodes = [{"name": n, "weight": d.get("weight", 1), "sources": d.get("sources", [])} for n, d in top]
    return jsonify({
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "components": components, "top_nodes": top_nodes, "exists": True,
    })


@app.route("/api/graph/search", methods=["POST"])
def api_graph_search() -> Any:
    term = ((request.json or {}).get("term", "")).strip().lower()
    if not term:
        return jsonify({"results": []})
    G = load_graph_cached(ctx)
    results = [
        {
            "node":    node,
            "weight":  data.get("weight", 1),
            "sources": data.get("sources", []),
            "out": [(o, G[node][o].get("relations", ["→"])[0]) for o in G.successors(node)][:8],
            "in":  [(p, G[p][node].get("relations", ["→"])[0]) for p in G.predecessors(node)][:8],
        }
        for node, data in G.nodes(data=True) if term in node
    ]
    results.sort(key=lambda x: x["weight"], reverse=True)
    return jsonify({"results": results[:15]})


@app.route("/api/graph/delete", methods=["POST"])
def api_graph_delete() -> Any:
    if ctx.graph_file.exists():
        ctx.graph_file.unlink()
    invalidate_graph_cache(ctx)
    return jsonify({"ok": True})


@app.route("/api/graph/data")
def api_graph_data() -> Any:
    G = load_graph_cached(ctx)
    limit = int(request.args.get("limit", 150))
    if G.number_of_nodes() == 0:
        return jsonify({"nodes": [], "links": []})
    top = sorted(G.nodes(data=True), key=lambda x: x[1].get("weight", 1), reverse=True)[:limit]
    ids = {n for n, _ in top}
    nodes = [{"id": n, "weight": d.get("weight", 1), "sources": d.get("sources", [])} for n, d in top]
    links = [
        {"source": s, "target": o, "relation": d.get("relations", [""])[0], "weight": d.get("weight", 1)}
        for s, o, d in G.edges(data=True) if s in ids and o in ids
    ]
    return jsonify({"nodes": nodes, "links": links})


# ── User context ──────────────────────────────────────────

@app.route("/api/user-context", methods=["GET", "POST"])
def api_user_context() -> Any:
    if request.method == "GET":
        return jsonify({
            "attivo":  bool(ctx.state.user_context),
            "len":     len(ctx.state.user_context),
            "preview": ctx.state.user_context[:200] if ctx.state.user_context else "",
        })
    action = (request.json or {}).get("action", "")
    if action == "reload":
        _load_user_context()
        return jsonify({"ok": True, "len": len(ctx.state.user_context)})
    if action == "clear":
        ctx.state.user_context = ""
        return jsonify({"ok": True})
    return jsonify({"error": "Azione non riconosciuta."}), 400


# ── Materia (multi-materia) ───────────────────────────────

@app.route("/api/materia", methods=["GET", "POST"])
def api_materia() -> Any:
    if request.method == "GET":
        return jsonify({
            "materia":       ctx.state.materia,
            "materie_lista": list(ctx.state.materie_attive.keys()),
            "prompt":        ctx.state.system_prompt,
            "context_len":   len(ctx.state.materia_context),
        })

    data = request.json or {}
    action = data.get("action", "")

    if action == "reset":
        ctx.state.reset()
        ctx.state.system_prompt = build_system_prompt(ctx.state.materie_attive)
        return jsonify({"ok": True, "msg": "Prompt, modalita' e contesto ripristinati."})

    if action == "attiva":
        nome = data.get("nome", "").strip()
        context_file = ctx.vault_path / ctx.context_dir / f"{nome}.md"
        if not context_file.exists():
            return jsonify({"error": f"Contesto '{nome}' non trovato. Usa 'genera' prima."}), 404
        ctx.state.materie_attive[nome] = context_file.read_text(encoding="utf-8")
        ctx.state.system_prompt = build_system_prompt(ctx.state.materie_attive)
        nomi = list(ctx.state.materie_attive.keys())
        return jsonify({
            "ok": True, "msg": f"Materia aggiunta: {nome}. Attive: {', '.join(nomi)}",
            "context_len": len(ctx.state.materia_context), "materie": nomi,
        })

    if action == "disattiva":
        nome = data.get("nome", "").strip()
        removed = ctx.state.materie_attive.pop(nome, None)
        if not removed:
            return jsonify({"error": f"Materia '{nome}' non era attiva."}), 404
        ctx.state.system_prompt = build_system_prompt(ctx.state.materie_attive)
        nomi = list(ctx.state.materie_attive.keys())
        return jsonify({"ok": True, "msg": f"Materia disattivata: {nome}", "materie": nomi})

    if action == "genera":
        nome = data.get("nome", "").strip()
        col = ctx.get_collection()
        if col.count() == 0:
            return jsonify({"error": "Vault non indicizzato."}), 400
        q_emb = ctx.cached_encode(nome).tolist()
        res = col.query(query_embeddings=[q_emb], n_results=min(10, col.count()))
        file_scores = {}
        for meta in res["metadatas"][0]:
            file_scores.setdefault(meta["file"], meta["path"])
        file_list = list(file_scores.keys())[:6]
        samples = []
        for path in list(file_scores.values())[:3]:
            try:
                r = col.get(where={"path": path}, include=["documents"])
                if r["documents"]:
                    samples.append(r["documents"][0][:300])
            except Exception:
                continue
        try:
            contesto = call_llm(ctx, [
                {"role": "system", "content": "Genera file di contesto markdown per vault Obsidian. Rispondi SOLO con il contenuto markdown."},
                {"role": "user",   "content": (
                    f"Materia: '{nome}'\nFile rilevanti: {', '.join(file_list)}\n"
                    f"Contenuto campione:\n{''.join(samples)}\n\n"
                    "Genera un file con: titolo, concetti chiave, file principali con [[link]], "
                    "elenco cartelle nel formato '- NomeCartella/' per il filtro RAG. Conciso."
                )},
            ])
            ctx_dir = ctx.vault_path / ctx.context_dir
            ctx_dir.mkdir(exist_ok=True)
            (ctx_dir / f"{nome}.md").write_text(contesto, encoding="utf-8")
            return jsonify({"ok": True, "file": f"{ctx.context_dir}/{nome}.md", "file_trovati": file_list})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    if action == "aggiorna":
        nome = data.get("nome", "").strip()
        context_file = ctx.vault_path / ctx.context_dir / f"{nome}.md"
        if not context_file.exists():
            return jsonify({"error": f"Contesto '{nome}' non trovato."}), 404
        col = ctx.get_collection()
        q_emb = ctx.cached_encode(nome).tolist()
        res = col.query(query_embeddings=[q_emb], n_results=10)
        files = list(dict.fromkeys(m["file"] for m in res["metadatas"][0]))
        links = "\n".join(f"- [[{Path(f).stem}]]" for f in files)
        new_content = context_file.read_text(encoding="utf-8") + f"\n\n## File aggiornati automaticamente\n{links}\n"
        context_file.write_text(new_content, encoding="utf-8")
        if nome in ctx.state.materie_attive:
            ctx.state.materie_attive[nome] = new_content
        return jsonify({"ok": True, "aggiunti": len(files), "files": files})

    if action == "lista":
        ctx_dir = ctx.vault_path / ctx.context_dir
        if not ctx_dir.exists():
            return jsonify({"contesti": []})
        return jsonify({"contesti": sorted(f.stem for f in ctx_dir.glob("*.md"))})

    return jsonify({"error": "Azione non riconosciuta."}), 400


# ═══════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("VaultRAG Web — http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
