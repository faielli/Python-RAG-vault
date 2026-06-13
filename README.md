# VaultRAG

Sistema RAG (Retrieval-Augmented Generation) ibrido — vettoriale + knowledge graph — per un vault Obsidian, pensato per studio e uso scolastico generale (appunti, libri, dispense).

Backend Flask in Python (`app.py`, `rag_core.py`), frontend single-page HTML/JS servito inline da Flask (`frontend.html`).

**Avvio:**
```bash
python app.py
```
App disponibile su `http://localhost:5000`.

---

## 1. Architettura

Il progetto è organizzato in moduli con dependency injection tramite `VaultRagContext` (dataclass condivisa, niente più variabili globali sparse):

| File | Ruolo |
|---|---|
| `app.py` | Entry point Flask: configurazione, routing principale, frontend |
| `rag_core.py` | Logica core: estrazione testo, chunking, embedding, ChromaDB, knowledge graph, chiamate LLM |
| `upload_handler.py` | Blueprint Flask per RAG temporaneo su file caricati al volo (nessuna persistenza) |
| `model_switcher.py` | Cambio modello/LLM a runtime senza riavvio |
| `frontend.html` | Interfaccia utente (single-page app) |
| `test_rag_core.py` | Test unitari per `rag_core.py` |

---

## 2. Configurazione

I parametri principali sono definiti come campi di default nella dataclass `VaultRagContext` (`rag_core.py`) e nelle costanti in cima ad `app.py`.

### Percorsi e dati persistenti

| Costante | Default | Descrizione |
|---|---|---|
| `VAULT_PATH` | `/home/fede/obsidian_notes/Notes` | Percorso root del vault Obsidian |
| `DB_PATH` | `~/.vault_rag_db` | Directory ChromaDB su disco |
| `HISTORY_FILE` | `~/.vault_rag_history.json` | Cronologia conversazioni (JSON) |
| `STATE_FILE` | `~/.vault_rag_state.json` | Mappa `path -> mtime` per indicizzazione incrementale |
| `GRAPH_FILE` | `~/.vault_rag_graph.json` | Knowledge graph serializzato (formato `node_link` NetworkX) |
| `CHAT_DIR` | `_chat` | Sottocartella del vault esclusa dall'indicizzazione (log conversazioni) |
| `context_dir` | `_context` | Sottocartella con i file di contesto per materia |

> **Nota:** `VAULT_PATH` contiene un percorso assoluto specifico (`/home/fede/...`). Per uso su altre macchine, sostituirlo o renderlo configurabile via variabile d'ambiente.

### LLM e API

| Campo | Default | Descrizione |
|---|---|---|
| `api_key` | da `OPENROUTER_API_KEY` (env) | API key per il provider LLM |
| `base_url` | `https://openrouter.ai/api/v1` | Endpoint compatibile OpenAI (OpenRouter) |
| `model` | `qwen-plus` | Modello LLM di default |
| `max_tokens` | `8192` | Limite token per risposta LLM |

Il modello può essere cambiato a runtime senza riavviare l'app, vedi [§4.5 Gestione Modello](#45-gestione-modello).

### Retrieval e indicizzazione

| Campo | Default | Descrizione |
|---|---|---|
| `n_results` | `2` | Numero di chunk restituiti dal retriever al prompt LLM |
| `chunk_size` | `500` | Dimensione di ogni chunk in caratteri |
| `chunk_overlap` | `50` | Sovrapposizione tra chunk adiacenti in caratteri |
| `extensions` | `*.md, *.txt, *.pdf, *.docx, *.epub, *.odt, *.ods, *.html, *.htm` | Tipi di file indicizzati dal vault |
| `ocr_langs` | `ita+eng` | Lingue Tesseract per OCR su PDF scansionati |
| `embed_model` | `all-MiniLM-L6-v2` | Modello SentenceTransformers per embedding testuali (384 dim) |
| `code_embed_model` | `flax-sentence-embeddings/st-codesearch-distilroberta-base` | Modello di embedding specializzato per codice |

> Per testi prevalentemente in italiano si può valutare `multilingual-e5-large` come alternativa a `all-MiniLM-L6-v2`.

### Conversazione, duplicati, correzioni

| Campo | Default | Descrizione |
|---|---|---|
| `max_history` | `20` | Numero massimo di scambi mantenuti in cronologia |
| `dup_threshold` | `0.97` | Soglia di cosine similarity per rilevamento file duplicati |
| `max_corrections` | `500` | Numero massimo di correzioni memorizzate |
| `correction_boost` | `2.0` | Fattore di boost applicato ai contenuti corretti |
| `error_context_n` | `5` | Numero di chunk di contesto usati per la verifica errori |

### Knowledge graph

| Campo | Default | Descrizione |
|---|---|---|
| `graph_chunk_sample` | `3` | Chunk per file campionati per l'estrazione di triplette tramite LLM |
| `graph_max_files` | `200` | Numero massimo di file processati durante la build del grafo |
| `graph_max_workers` | `8` | Worker paralleli per la build del grafo |

---

## 3. Funzioni principali (`rag_core.py`)

### 3.1 Utility

**`strip_emoji(text: str) -> str`**
Rimuove emoji e caratteri Unicode non testuali dall'output dell'LLM prima di restituirlo al frontend.

**`get_collection(ctx) -> chromadb.Collection`**
Apre (o crea) la collection ChromaDB persistente in `DB_PATH`.

**`vault_files(ctx) -> list[Path]`**
Scansiona ricorsivamente `VAULT_PATH` per i file con estensioni in `extensions`, escludendo `_chat`.

**`extract_text(file: Path, ocr_langs: str) -> str`**
Estrae testo grezzo in base all'estensione:
- `.md`, `.txt` → lettura diretta
- `.pdf` → PyMuPDF (`fitz`); se una pagina è priva di testo, fallback a OCR Tesseract (200 DPI)
- `.docx` → `python-docx`, tutti i paragrafi
- `.epub` → `ebooklib` + BeautifulSoup su ogni documento HTML interno
- `.odt`, `.ods` → `odfpy` (`teletype.extractText`)
- `.html`, `.htm` → BeautifulSoup, testo senza tag

Ritorna stringa vuota in caso di errore.

**`chunk_text(text: str, chunk_size, chunk_overlap) -> list[str]`**
Suddivisione a finestra fissa con overlap:
```
chunk[0] = text[0:500]
chunk[1] = text[450:950]
chunk[2] = text[900:1400]
```
(con i default `chunk_size=500`, `chunk_overlap=50`)

**`load_history()` / `save_history(history)`**
Carica/salva la cronologia da `HISTORY_FILE`. `save_history` mantiene solo gli ultimi `max_history * 2` messaggi.

**`load_state()` / `save_state(state)`**
Carica/salva la mappa `{path: mtime}` per l'indicizzazione incrementale: se il `mtime` non è cambiato, il file viene saltato al reindex.

**`save_interaction(question, answer, sources)`**
Salva ogni scambio come file `.md` in `VAULT_PATH/_chat/`, con nome `YYYY-MM-DD-<slug-domanda>.md`, frontmatter YAML e sezioni Domanda / Risposta / Fonti (con wikilink).

**`call_llm(ctx, messages) -> str`**
Invia i messaggi al modello LLM configurato (API compatibile OpenAI) e restituisce la risposta dopo `strip_emoji`.

### 3.2 Knowledge Graph

**`load_graph(ctx) -> nx.DiGraph`**
Carica il grafo da `GRAPH_FILE` (formato `node_link`). Se mancante o corrotto, ritorna un grafo diretto vuoto.

**`save_graph(ctx, G)`**
Serializza il grafo in JSON `node_link` su `GRAPH_FILE`.

**`extract_triples_llm(text_sample, filename) -> list[tuple]`**
Chiede all'LLM di estrarre fino a 15 triplette `SOGGETTO | RELAZIONE | OGGETTO`, valida (3 parti, ognuna < 60 caratteri) e normalizza in minuscolo.

**`build_graph_from_vault(ctx, solo_nuovi: bool = True)`**
Costruisce/aggiorna il knowledge graph in un thread daemon:
1. Se `solo_nuovi=True` carica il grafo esistente, altrimenti riparte da zero
2. Verifica che il vault sia indicizzato (ChromaDB non vuoto)
3. Per ogni file (max `graph_max_files`): recupera i chunk, campiona `graph_chunk_sample` chunk, chiama `extract_triples_llm`
4. Aggiunge nodi (con `weight` e `sources`) e archi (con `relations` e `weight`, incrementati se già esistenti)
5. Salva il grafo su disco

Log disponibili via `/api/graph/log`.

**`graph_expand_query(question, G, top_n=5) -> tuple`**
1. Tokenizza la domanda (parole ≥ 3 caratteri)
2. Per ogni nodo calcola l'overlap pesato con i token della query
3. Seleziona i `top_n` nodi seed con score più alto
4. Espande a 1-hop (predecessori + successori)
5. Raccoglie i file sorgente associati (deduplicati, max 6)
6. Costruisce un testo relazionale con le relazioni dei seed (max 20 righe)

Ritorna `(entita_espanse, file_sorgente, testo_relazionale)`.

### 3.3 Retrieval

**`query_rag(ctx, question) -> tuple[list, list, str | None]`**
RAG vettoriale puro: embedda la domanda con `embed_model`, interroga ChromaDB per i `n_results` chunk più simili. Se è attivo un contesto materia, filtra per le cartelle indicate; se il filtro non produce risultati, ricade sulla ricerca globale.

**`query_hybrid(ctx, question) -> tuple[list, list, str | None, str]`**
RAG ibrido = `query_rag` + graph expansion:
1. Esegue `query_rag`
2. Chiama `graph_expand_query` per individuare file suggeriti dal grafo
3. Per ogni file suggerito non già presente nei risultati vettoriali, recupera il chunk più rilevante con sotto-query filtrata (max 2 file aggiuntivi)
4. Concatena i risultati vettoriali con quelli graph-expanded

---

## 4. API Routes

### 4.1 Chat e Query

**`GET /`**
Serve il frontend HTML/JS (single-page app).

**`POST /api/chat`**
Endpoint principale della chat RAG ibrida.
- Body: `{"question": "testo"}`
- Processo: `query_hybrid` → costruzione prompt (contesto materia + relazioni grafo + chunk vettoriali) → `call_llm` → salvataggio in history e in `_chat/`
- Response: `{"answer", "sources": [{"file","chunk","total"}], "graph_used": bool}`

**`POST /api/open`**
Risponde a una domanda su un file specifico, bypassando la ricerca globale.
- Body: `{"nome": "nome_file", "domanda": "testo"}`
- Cerca il file per match parziale case-insensitive (vault, poi `_chat/`), recupera tutti i chunk, calcola la similarità con la domanda e passa i top `n_results` all'LLM.

**`POST /api/search`**
Ricerca per nome file e per contenuto semantico.
- Body: `{"termine": "testo"}`
- `by_name`: match parziale case-insensitive sul nome file (file + cartella)
- `by_content`: query vettoriale top-8, nomi file deduplicati

**`POST /api/info`**
Metadati di un file: nome, cartella relativa, tipo, dimensione (KB), data modifica, numero di chunk indicizzati (per i PDF anche il numero di pagine).

**`GET /api/duplicati`**
Confronta l'embedding del primo chunk di ogni file; se cosine similarity ≥ `dup_threshold` (default 0.97), segnala come duplicato.
- Response: `{"duplicati": [{"file1","file2","sim"}], "soglia": 0.97}`

**`GET /api/history?n=10`**
Ultimi `n` scambi domanda/risposta dalla cronologia.

**`POST /api/reset`**
Azzera `HISTORY_FILE`.

**`POST /api/esporta`**
Esporta la conversazione corrente in `_chat/export-YYYY-MM-DD-HHMM.md`, con ogni scambio numerato in Markdown.

**`GET /api/esportazioni`**
Lista i file di export in `_chat/`, ordinati per data decrescente.

**`GET /api/lista-file`**
Lista tutti i file del vault (ordinati per nome) con conteggio totale.

**`GET /api/cartelle`**
Lista le cartelle uniche del vault (cartella padre di ogni file).

**`GET /api/numero-file`**
Statistiche dell'indice:
- `vault`: numero totale di file nel vault
- `chunk_db`: numero totale di chunk in ChromaDB
- `non_indicizzati`: file il cui `mtime` non corrisponde a `STATE_FILE`

**`GET /api/stats`**
Statistiche generali: chunk totali, file totali, numero di scambi in cronologia, modello LLM attivo, materia attiva.

### 4.2 Indicizzazione

**`POST /api/reindex`**
Avvia il reindex del vault in un thread daemon.
- Body: `{"solo_nuovi": true|false}`
  - `true` (default): indicizza solo i file non ancora presenti in ChromaDB (controllo su id `{file}::chunk0`)
  - `false`: reindex completo — elimina e ricrea l'intera collection ChromaDB

Per ogni file: `extract_text` → `chunk_text` → embedding in batch (`batch_size=64`) → insert in ChromaDB con id `{path}::chunkN` e metadati `{file, path, type, chunk, total}` → aggiornamento `STATE_FILE`.

Response immediata: `{"ok": true}` (processo in background).

**`GET /api/reindex-log`**
Polling dello stato del reindex.
Response: `{"log": ["OK: file.pdf (12 chunk)", ...], "running": bool}`

### 4.3 Knowledge Graph

**`POST /api/graph/build`**
Avvia la build del knowledge graph in background.
- Body: `{"solo_nuovi": true|false}` — `true`: aggiunge solo i file non ancora processati; `false`: ricostruisce da zero

**`GET /api/graph/log`**
Polling dello stato della build.
Response: `{"log": [...], "running": bool}`

**`GET /api/graph/stats`**
Statistiche del grafo: `nodes`, `edges`, `components` (componenti debolmente connesse), `top_nodes` (top 10 per `weight` con file sorgente).

**`POST /api/graph/search`**
Ricerca nodi per sottostringa. Per ogni nodo trovato: nome, peso, file sorgente, archi uscenti (successori + relazione), archi entranti (predecessori). Max 15 risultati, ordinati per peso decrescente.

**`GET /api/graph/data?limit=150`**
Dati del grafo per visualizzazione D3.js: i `limit` nodi con peso più alto e tutti gli archi tra essi.
Response: `{"nodes": [{"id","weight","sources"}], "links": [{"source","target","relation","weight"}]}`

**`POST /api/graph/delete`**
Elimina `GRAPH_FILE` dal disco.

### 4.4 Gestione Materia / Contesto

**`GET /api/materia`**
Stato del contesto materia corrente: nome materia attiva, system prompt corrente, lunghezza del testo di contesto.

**`POST /api/materia` — `action: "attiva"`**
- Body: `{"action": "attiva", "nome": "reti"}`
- Carica `_context/reti.md` dal vault, sostituisce il system prompt con uno specializzato. Da questo momento le query sono filtrate per le cartelle elencate nel file di contesto.

**`POST /api/materia` — `action: "reset"`**
Ripristina il system prompt al default e azzera materia/contesto attivi. Le query tornano a cercare in tutto il vault.

**`POST /api/materia` — `action: "genera"`**
- Body: `{"action": "genera", "nome": "nome_materia"}`
- Interroga ChromaDB col nome materia, recupera i file più rilevanti, campiona il contenuto e chiede all'LLM di generare un file `.md` strutturato in `_context/nome_materia.md`.

**`POST /api/materia` — `action: "aggiorna"`**
Aggiorna un file di contesto esistente: cerca file recenti pertinenti e li appende in una sezione `## File aggiornati automaticamente`.

**`POST /api/materia` — `action: "lista"`**
Elenca tutti i file `.md` in `_context/` (senza estensione).

### 4.5 Gestione Modello

**`GET /api/model`**
Configurazione corrente: modello attivo, `base_url`, se l'API key è impostata.

**`GET /api/model/list`**
Lista dei modelli predefiniti disponibili (vedi `model_switcher.AVAILABLE_MODELS`).

**`POST /api/model/change`**
- Body: `{"model_name": "...", "api_key": "...", "base_url": "..."}` (`api_key` e `base_url` opzionali)
- `model_name` può essere un preset (es. `"gemini-free"`) o un nome diretto di modello (es. `"google/gemma-4-31b-it:free"`)
- Se cambia `base_url` o `api_key`, il client LLM viene ricreato

**`POST /api/model/test`**
Esegue una chiamata di test al modello corrente e restituisce latenza e risposta.

### 4.6 Upload temporaneo (`upload_handler.py`)

**`POST /api/upload`**
RAG temporaneo su un file caricato al volo (nessuna persistenza, nessun accesso a ChromaDB/vault/history).
- Multipart form: `file` (uno dei formati supportati, max 50 MB) + `question` (testo)
- Processo: validazione → salvataggio in file temporaneo → `extract_text` → `chunk_text` → embedding di domanda e chunk → ranking per cosine similarity → top-3 chunk → `call_llm` → cleanup del file temporaneo (garantito anche in caso di errore)
- Response: `{"answer", "file", "chunks", "top_chunks", "best_score", "timing": {"embedding","llm","total"}}`

---

## 5. Struttura dei dati persistenti

| File / Directory | Formato | Contenuto |
|---|---|---|
| `~/.vault_rag_db/` | ChromaDB | Vettori embedding + chunk + metadati |
| `~/.vault_rag_history.json` | JSON array | Lista messaggi `{role, content}` |
| `~/.vault_rag_state.json` | JSON object | `{"path/file": "mtime_string"}` |
| `~/.vault_rag_graph.json` | JSON `node_link` | Knowledge graph serializzato (NetworkX) |
| `VAULT_PATH/_context/*.md` | Markdown | File di contesto per materia |
| `VAULT_PATH/_chat/*.md` | Markdown | Log conversazioni ed export |

---

## 6. Stato condiviso (`VaultRagContext`)

Tutto lo stato runtime è centralizzato nella dataclass `VaultRagContext` (`rag_core.py`), iniettata nei vari moduli (`app.py`, `upload_handler.py`, `model_switcher.py`) tramite dependency injection — non sono più presenti variabili globali sparse nel modulo.

Principali sotto-componenti:

| Campo | Descrizione |
|---|---|
| `state` (`AppState`) | System prompt corrente, materia attiva, contesto materia, log e flag di reindex/graph build |
| `_llm_client`, `_embedder`, `_code_embedder`, `_chroma` | Singleton lazy (client OpenAI, modelli embedding, client ChromaDB) con relativi lock thread-safe |

> In un deployment multi-worker, lo stato runtime (`AppState`) andrebbe spostato su uno store condiviso (es. Redis), perché non persiste tra i processi.

---

## 7. Pipeline RAG ibrida — schema

Flusso completo di una query su `/api/chat`:

1. La domanda viene embeddata con `embed_model` (default `all-MiniLM-L6-v2`, 384 dim)
2. ChromaDB esegue la ricerca per cosine similarity e restituisce `n_results` chunk
3. Se è attiva una materia, i risultati vengono filtrati per le cartelle del relativo file di contesto
4. In parallelo, `graph_expand_query` tokenizza la domanda e trova nodi del grafo con overlap semantico
5. I file suggeriti dal grafo (non già nei risultati vettoriali) vengono aggiunti con sotto-query filtrate (max 2)
6. Il prompt finale contiene: system prompt + contesto materia + relazioni grafo + chunk vettoriali
7. L'LLM configurato (default `qwen-plus`, cambiabile a runtime) genera la risposta
8. La coppia domanda/risposta viene salvata in history e come file `.md` in `_chat/`

---

## 8. Manutenzione

**Aggiungere nuovi file al vault**
1. Copia il file nella cartella appropriata del vault
2. `POST /api/reindex` con `{"solo_nuovi": true}` per indicizzare solo i nuovi file
3. Aggiorna il file `_context/materia.md` corrispondente con il nuovo `[[wikilink]]` e note descrittive
4. Se il file introduce nuovi argomenti, aggiungili a `## Concetti chiave`

**Ricostruire l'indice da zero**
`POST /api/reindex` con `{"solo_nuovi": false}` — elimina e ricrea la collection ChromaDB indicizzando tutti i file.

**Aggiornare il knowledge graph**
`POST /api/graph/build` con `{"solo_nuovi": true}` dopo ogni reindex, per aggiornare le relazioni semantiche.

**Cambiare materia attiva**
- UI: tab *Materia* → inserisci nome → *Attiva*
- API: `POST /api/materia` con `{"action": "attiva", "nome": "nome_materia"}`
- Per tornare alla ricerca globale: `POST /api/materia` con `{"action": "reset"}`

**Cambiare modello LLM**
- `GET /api/model/list` per vedere i preset disponibili
- `POST /api/model/change` con `{"model_name": "gemini-free"}` (o un nome di modello diretto)
