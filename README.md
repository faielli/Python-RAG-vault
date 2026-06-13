# RAG-vault
Retrieval-Augmented Generation (RAG) for school books, documents and general school usage.



documentazione VaultRAG
Manuale Tecnico
app.py - Flask RAG backend + frontend single-file

1. Panoramica
VaultRAG e' un'applicazione web Flask che implementa un sistema RAG ibrido (vettoriale + knowledge graph) sul vault Obsidian. Il backend e' Python, il frontend e' una single-page app HTML/JS servita inline da Flask.

Avvio:  python app.py  ->  http://localhost:5000

2. Configurazione (CONFIG)
Tutte le costanti sono definite in cima al file e modificabili direttamente.

Costante
Default
Descrizione
VAULT_PATH
/home/fede/obsidian_notes/Notes
Percorso root del vault Obsidian
API_KEY
sk-...
API key per Qwen via DashScope (Alibaba)
MODEL
qwen-plus
Modello LLM usato per la generazione
N_RESULTS
3
Numero di chunk restituiti dal retriever al prompt LLM
DB_PATH
~/.vault_rag_db
Directory ChromaDB su disco
COLLECTION
obsidian_vault
Nome della collection ChromaDB
EXTENSIONS
*.md, *.txt, *.pdf...
Tipi di file indicizzati dal vault
CHAT_DIR
_chat
Cartella esclusa dall'indicizzazione (log chat)
HISTORY_FILE
~/.vault_rag_history.json
File JSON della cronologia conversazioni
STATE_FILE
~/.vault_rag_state.json
Mappa path->mtime per indicizzazione incrementale
CHUNK_SIZE
500
Dimensione chunk in caratteri
CHUNK_OVERLAP
50
Sovrapposizione tra chunk adiacenti in caratteri
MAX_HISTORY
10
Numero massimo di scambi mantenuti in memoria
DUP_THRESHOLD
0.97
Soglia cosine similarity per rilevamento duplicati
OCR_LANGS
ita+eng
Lingue Tesseract per OCR su PDF scansionati
GRAPH_FILE
~/.vault_rag_graph.json
File JSON di persistenza del knowledge graph
GRAPH_CHUNK_SAMPLE
3
Chunk per file campionati per estrazione triplette LLM
GRAPH_MAX_FILES
200
File massimi processati per build del knowledge graph

Modello di embedding: all-MiniLM-L6-v2 (SentenceTransformers) - 384 dimensioni. Per testi prevalentemente italiani considera multilingual-e5-large.

LLM endpoint: Alibaba DashScope - compatibile con API OpenAI.
3. Funzioni Python
3.1 Utility
strip_emoji(text: str) -> str
Rimuove tutti i caratteri Unicode non ASCII e non alfanumerici/punteggiatura dal testo. Usata per pulire l'output dell'LLM prima di restituirlo al frontend.

get_collection() -> chromadb.Collection
Apre (o crea se non esiste) la collection ChromaDB persistente in DB_PATH. Chiamata ad ogni operazione sul DB vettoriale - non mantiene una connessione globale persistente.

vault_files() -> list[Path]
Scansiona ricorsivamente VAULT_PATH per tutti i file con le estensioni in EXTENSIONS, escludendo qualsiasi file la cui path contenga _chat. Restituisce una lista di oggetti Path.

extract_text(file: Path) -> str
Estrae il testo grezzo da un file in base all'estensione:
- .md, .txt - lettura diretta con errors="ignore"
- .pdf - PyMuPDF (fitz); se una pagina e' vuota tenta OCR con Tesseract a 200 DPI nelle lingue OCR_LANGS
- .docx - python-docx, estrae tutti i paragrafi
- .epub - ebooklib + BeautifulSoup, estrae testo da tutti i documenti HTML interni
- .odt, .ods - odfpy (teletype.extractText)
- .html, .htm - BeautifulSoup, testo grezzo senza tag
Restituisce stringa vuota in caso di errore.

chunk_text(text: str) -> list[str]
Divide il testo in chunk di CHUNK_SIZE caratteri con overlap di CHUNK_OVERLAP caratteri. Chunking a dimensione fissa (non semantico).
chunk[0] = text[0 : 500]
chunk[1] = text[450 : 950]
chunk[2] = text[900 : 1400]

load_history() / save_history(history)
Carica/salva la cronologia conversazione da HISTORY_FILE in formato JSON. save_history mantiene solo gli ultimi MAX_HISTORY*2 messaggi (domande + risposte).

load_state() / save_state(state)
Carica/salva il dizionario {path: mtime} usato per l'indicizzazione incrementale. Se il mtime di un file non e' cambiato, il file viene saltato durante il reindex.

save_interaction(question, answer, sources)
Salva ogni coppia domanda/risposta come file .md in VAULT_PATH/_chat/. Nome file: YYYY-MM-DD-<slug-domanda>.md. Include frontmatter YAML, sezioni Domanda, Risposta e Fonti con wikilink.

call_llm(messages: list[dict]) -> str
Invia una lista di messaggi al modello LLM configurato via OpenAI-compatible API e restituisce il testo della risposta passato per strip_emoji.
3.2 Knowledge Graph
load_graph() -> nx.DiGraph
Carica il knowledge graph da GRAPH_FILE (formato node_link di NetworkX). Se il file non esiste o e' corrotto restituisce un grafo diretto vuoto.

save_graph(G: nx.DiGraph)
Serializza il grafo in formato node_link JSON e lo scrive su GRAPH_FILE.

extract_triples_llm(text_sample: str, filename: str) -> list[tuple]
Invia un campione di testo all'LLM chiedendo di estrarre fino a 15 triplette nel formato SOGGETTO | RELAZIONE | OGGETTO. Ogni riga viene parsata, validata (3 parti, ognuna < 60 caratteri) e normalizzata in minuscolo. Restituisce lista di tuple (soggetto, relazione, oggetto).

build_graph_from_vault(solo_nuovi: bool = True)
Costruisce o aggiorna il knowledge graph in un thread daemon separato. Processo:
- Se solo_nuovi=True carica il grafo esistente, altrimenti parte da zero
- Verifica che il vault sia indicizzato (ChromaDB non vuoto)
- Per ogni file (max GRAPH_MAX_FILES): recupera chunk dal DB, campiona GRAPH_CHUNK_SAMPLE chunk, chiama extract_triples_llm
- Per ogni tripletta: aggiunge nodi con weight e sources, aggiunge archi con relations e weight (incrementa se gia' esistente)
- Salva il grafo aggiornato su disco
I log sono scritti nella lista globale graph_log e consultabili via /api/graph/log.

graph_expand_query(question: str, G: nx.DiGraph, top_n: int = 5) -> tuple
Dato una domanda in linguaggio naturale:
- Tokenizza la domanda in parole di almeno 3 caratteri
- Per ogni nodo del grafo calcola overlap tra token del nodo e della query, pesato per il weight del nodo
- Seleziona i top_n nodi seed con score piu' alto
- Espande a 1-hop: aggiunge successori e predecessori di ogni seed
- Raccoglie i file sorgente associati (deduplicati, max 6)
- Costruisce testo relazionale con relazioni in/out dei seed (max 20 righe)
Restituisce (entita_espanse, file_sorgente, testo_relazionale).
3.3 Retrieval
query_rag(question: str) -> tuple[list, list, str|None]
RAG vettoriale puro. Embeds la domanda con all-MiniLM-L6-v2 e interroga ChromaDB per i N_RESULTS chunk piu' simili. Se e' attivo un contesto materia, filtra i risultati per le cartelle specificate nel file .md di contesto. Se il filtro non produce risultati, torna alla ricerca globale. Restituisce (documenti, metadati, errore).

query_hybrid(question: str) -> tuple[list, list, str|None, str]
RAG ibrido = vettoriale + graph expansion. Esegue la stessa ricerca di query_rag, poi:
- Chiama graph_expand_query per trovare file suggeriti dal grafo
- Per ogni file suggerito non gia' presente nei risultati vettoriali, recupera il chunk piu' rilevante con sotto-query filtrata per file (max 2 file aggiuntivi)
- Concatena risultati vettoriali con quelli graph-expanded
Restituisce (documenti, metadati, errore, testo_relazionale).


4. API Routes
4.1 Chat e Query
GET /
Serve il frontend HTML/JS come single-page app tramite render_template_string(HTML).

POST /api/chat
Endpoint principale della chat RAG ibrida.
Body: {"question": "testo"}
Processo: chiama query_hybrid, costruisce il prompt iniettando contesto materia + relazioni grafo + chunk vettoriali, chiama call_llm, salva in history e in _chat/.
Response: {"answer", "sources": [{"file","chunk","total"}], "graph_used": bool}

POST /api/open
Apre un file specifico e risponde a una domanda su di esso, bypassando la ricerca globale.
Body: {"nome": "nome_file", "domanda": "testo"}
Cerca il file per match parziale case-insensitive prima nel vault poi in _chat/, recupera tutti i chunk del file, calcola cosine similarity tra embedding della domanda e ogni chunk, passa i top N_RESULTS all'LLM.

POST /api/search
Ricerca doppia: per nome file e per contenuto semantico.
Body: {"termine": "testo"}
- by_name: match parziale case-insensitive sul nome file, restituisce file e cartella
- by_content: query vettoriale top-8, restituisce lista nomi file deduplicati

POST /api/info
Restituisce metadati di un file: nome, cartella relativa, tipo, dimensione KB, data modifica, numero di chunk indicizzati. Per i PDF aggiunge il numero di pagine.

GET /api/duplicati
Rileva file potenzialmente duplicati confrontando i vettori embedding del primo chunk (chunk 0) di ogni file. Se cosine similarity >= DUP_THRESHOLD (0.97) vengono segnalati come duplicati.
Response: {"duplicati": [{"file1","file2","sim"}], "soglia": 0.97}

GET /api/history?n=10
Restituisce gli ultimi n scambi (coppie domanda/risposta) dalla cronologia.

POST /api/reset
Cancella HISTORY_FILE azzerando la cronologia conversazione.

POST /api/esporta
Esporta tutta la conversazione corrente in un file .md in _chat/ con nome export-YYYY-MM-DD-HHMM.md. Include ogni scambio numerato con domanda e risposta in formato Markdown.

GET /api/esportazioni
Lista tutti i file di export presenti in _chat/, ordinati per data decrescente.

GET /api/lista-file
Lista tutti i file nel vault (ordinati per nome) con conteggio totale.

GET /api/cartelle
Lista tutte le cartelle uniche presenti nel vault (nome della cartella padre di ogni file).

GET /api/numero-file
Statistiche di stato dell'indice:
- vault: numero di file totali nel vault
- chunk_db: numero di chunk totali in ChromaDB
- non_indicizzati: file il cui mtime non corrisponde a quello in STATE_FILE

GET /api/stats
Statistiche generali: chunk totali, file totali, numero scambi in cronologia, modello LLM attivo, materia attiva.
4.2 Indicizzazione
POST /api/reindex
Avvia il reindex del vault in un thread daemon separato.
Body: {"solo_nuovi": true|false}
- solo_nuovi: true (default) - indicizza solo i file non ancora presenti in ChromaDB (controlla l'id {file}::chunk0). File gia' indicizzati vengono saltati.
- solo_nuovi: false - reindex completo: elimina l'intera collection ChromaDB e la ricrea da zero.

Processo per ogni file: estrae testo con extract_text, divide in chunk con chunk_text, calcola embedding in batch (batch_size=64), inserisce in ChromaDB con id {path}::chunkN e metadati {file, path, type, chunk, total}, aggiorna STATE_FILE con il nuovo mtime.
Response immediata: {"ok": true} - il processo gira in background.

GET /api/reindex-log
Polling dello stato del reindex.
Response: {"log": ["OK: file.pdf (12 chunk)", ...], "running": bool}
4.3 Knowledge Graph
POST /api/graph/build
Avvia la build del knowledge graph in background.
Body: {"solo_nuovi": true|false}
- solo_nuovi: true - aggiunge solo file non gia' processati
- solo_nuovi: false - ricostruisce il grafo da zero

GET /api/graph/log
Polling dello stato della build del grafo.
Response: {"log": [...], "running": bool}

GET /api/graph/stats
Statistiche del knowledge graph: nodes, edges, components (componenti debolmente connesse), top_nodes (top 10 nodi per weight con file sorgente).

POST /api/graph/search
Ricerca nodi nel grafo per sottostringa. Restituisce per ogni nodo trovato: nome, peso, file sorgente, archi in uscita (successori con relazione), archi in entrata (predecessori). Max 15 risultati per peso decrescente.

GET /api/graph/data?limit=150
Restituisce i dati del grafo per la visualizzazione D3.js. Prende i limit nodi con peso piu' alto e tutti gli archi tra di essi.
Response: {"nodes": [{"id","weight","sources"}], "links": [{"source","target","relation","weight"}]}

POST /api/graph/delete
Elimina GRAPH_FILE dal disco, cancellando il knowledge graph.
4.4 Gestione Materia / Contesto
GET /api/materia
Restituisce lo stato del contesto materia corrente: nome materia attiva, system prompt corrente, lunghezza del testo di contesto.

POST /api/materia - action: "attiva"
Body: {"action": "attiva", "nome": "reti"}
Legge il file _context/reti.md dal vault e lo carica in current_materia_context. Sostituisce il system prompt con uno specializzato sulla materia. Da questo momento tutte le query vengono filtrate per le cartelle elencate nel file di contesto.

POST /api/materia - action: "reset"
Ripristina il system prompt al default e azzera current_materia e current_materia_context. Le query tornano a cercare in tutto il vault.

POST /api/materia - action: "genera"
Body: {"action": "genera", "nome": "nome_materia"}
Genera automaticamente un file di contesto: interroga ChromaDB con il nome materia come query, recupera i file piu' rilevanti, campiona il contenuto e chiede all'LLM di generare un file .md strutturato. Scrive il file in _context/nome_materia.md.

POST /api/materia - action: "aggiorna"
Aggiorna un file di contesto esistente: interroga ChromaDB per trovare file recenti pertinenti e li appende al file esistente in una sezione "## File aggiornati automaticamente".

POST /api/materia - action: "lista"
Elenca tutti i file .md presenti in _context/ (senza estensione).


5. Struttura dei Dati Persistenti

File
Formato
Contenuto
~/.vault_rag_db/
ChromaDB
Vettori embedding + chunk + metadati
~/.vault_rag_history.json
JSON array
Lista messaggi {role, content}
~/.vault_rag_state.json
JSON object
{'/path/file': 'mtime_string'}
~/.vault_rag_graph.json
JSON node_link
Grafo NetworkX serializzato
VAULT_PATH/_context/*.md
Markdown
File di contesto per materia
VAULT_PATH/_chat/*.md
Markdown
Log conversazioni e export

6. Variabili Globali di Stato
Queste variabili sono globali al processo Flask - non persistono al riavvio (eccetto i dati su disco). In un deployment multi-worker andrebbero spostate su Redis o simili.

Variabile
Descrizione
current_system_prompt
System prompt LLM corrente (modificato da /api/materia). Default: assistente generico sul vault.
current_materia
Nome della materia attiva (stringa vuota se nessuna).
current_materia_context
Contenuto del file .md di contesto attivo, usato per filtrare il retrieval.
reindex_log
Lista di stringhe con il log dell'ultimo reindex. Condivisa col frontend via polling su /api/reindex-log.
reindex_running
Flag booleano mutex per il thread di reindex. Impedisce avvii concorrenti.
graph_log
Lista di stringhe con il log dell'ultima build del grafo.
graph_running
Flag booleano mutex per il thread del knowledge graph.

7. Pipeline RAG Ibrida - Schema

Flusso completo di una query su /api/chat:

- 1. La domanda viene embeddata con all-MiniLM-L6-v2 (384 dim)
- 2. ChromaDB esegue ricerca per cosine similarity e restituisce N_RESULTS chunk
- 3. Se current_materia e' attivo, i risultati vengono filtrati per cartelle specificate nel file .md di contesto
- 4. Parallelamente, graph_expand_query tokenizza la domanda e trova nodi nel grafo con overlap semantico
- 5. I file suggeriti dal grafo (non gia' nei risultati vettoriali) vengono aggiunti con sotto-query vettoriali filtrate (max 2)
- 6. Il prompt finale contiene: system prompt + contesto materia + relazioni grafo + chunk vettoriali
- 7. L'LLM (qwen-plus) genera la risposta
- 8. La coppia domanda/risposta viene salvata in history e come file .md in _chat/

8. Manutenzione e Aggiornamento
Aggiungere nuovi file al vault
- Aggiungi il file nella cartella appropriata del vault
- Avvia /api/reindex con solo_nuovi: true per indicizzare solo i nuovi file
- Aggiorna il file _context/materia.md corrispondente con il nuovo [[wikilink]] e note descrittive
- Se il file introduce nuovi argomenti, aggiungili a ## Concetti chiave

Ricostruire l'indice da zero
POST /api/reindex con solo_nuovi: false. Elimina la collection ChromaDB e la ricrea indicizzando tutti i file.

Aggiornare il knowledge graph
POST /api/graph/build con solo_nuovi: true dopo ogni reindex per aggiornare le relazioni semantiche.

Cambiare materia attiva
- Dalla UI: tab Materia -> inserisci nome -> clicca Attiva
- Via API: POST /api/materia con {"action": "attiva", "nome": "nome_materia"}
- Per tornare alla ricerca globale: POST /api/materia con {"action": "reset"}

VaultRAG - Manuale Tecnico
