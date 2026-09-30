# Changelog

Wszystkie istotne zmiany w projekcie będą dokumentowane w tym pliku.

Format bazuje na [Keep a Changelog](https://keepachangelog.com/pl/1.0.0/),
a projekt stosuje [Semantic Versioning](https://semver.org/lang/pl/).

## [1.3.3] - 2026-09-30

### Zmieniono
- **Wersja projektu**: 1.3.2 -> 1.3.3
- **Powrot do NumPy 2.x**: usuniety opcjonalny extra `mineru` z `pyproject.toml`. uv rozwiazuje extras razem z baza, wiec extra przypinal numpy 1.26.x dla wszystkich instalacji na Pythonie 3.11/3.12. `uv.lock` znowu zawiera tylko numpy 2.3.5 (zaleznosci takie jak w 1.3.1)
- MinerU instaluje sie recznie (poza lockiem): `uv pip install "magic-pdf[cpu]==0.6.1" "numpy<2" setuptools` — opis w README i w komunikacie `ImportError` parsera. Kolejne `uv sync` przywraca numpy 2.x i usuwa te pakiety
- Usuniete testy sprawdzajace extra `mineru` w `pyproject.toml`/`uv.lock`

## [1.3.2] - 2026-09-29

### Bezpieczenstwo
- **Path traversal w serwowaniu frontendu (SPA)**: catch-all route w `app/main.py` skladal sciezke przez `os.path.join` bez sprawdzenia, czy wynik zostaje w katalogu `app/static` — `GET //proc/self/environ` lub `/..%2f..%2fetc/passwd` zwracaly dowolny plik z serwera (w tym zmienne srodowiskowe z `OPENAI_API_KEY`). Nowy helper `resolve_static_path()` w pelni rozwiazuje sciezke (`..`, symlinki, petle symlinkow na Pythonie < 3.13) i serwuje plik tylko, gdy lezy wewnatrz rootu; wszystko inne dostaje 404. Rejestracja tras wydzielona do `register_spa_routes()` (testowalna bez buildu frontendu)

### Naprawiono
- **Puste cytowania po rerankingu (regresja z 1.3.1)**: `CrossEncoderReranker.rerank()` tworzyl nowy `SearchResult` bez `doc_id`, `page` i `chunk_index`, wiec przy domyslnej konfiguracji (reranking wlaczony) kazde zrodlo mialo pusty `doc_id`/`filename` i strone 0, a LLM dostawal "Page 0" w kontekscie. Reranker kopiuje teraz wynik przez `dataclasses.replace()` i zmienia tylko `score` i `metadata`
- **Osobny, nieaktualny indeks BM25 w QueryService**: `QueryService` tworzyl i ladowal wlasny `BM25Index`, niezalezny od indeksu `DocumentService`. Nowe dokumenty byly niewidoczne dla keyword search do restartu, a usuniete dokumenty nadal trafialy do kontekstu LLM i cytowan. Oba serwisy wspoldziela teraz jeden indeks (jawnie w `lifespan()` oraz domyslnie z `document_service.bm25_index`); `QueryService` laduje indeks z dysku tylko, gdy tworzy go sam
- **Parser MinerU zwracal pusty dokument**: `RAGAnythingParser` czytal plaska liste z `pipe_mk_uni_format()` tak, jakby byla lista stron z `para_blocks`, wiec kazdy PDF dawal 0 blokow tekstu, a upload konczyl sie bledem "Cannot add empty embeddings list". Konwersja korzysta teraz z `pipe.pdf_mid_data["pdf_info"]` (numery stron z `page_idx`, tytuly, rownania, obrazy i tabele z podpisami/przypisami); naprawiona tez podwojna sciezka `images/images/`, przez ktora gubione byly wszystkie obrazy
- **Brak zaleznosci MinerU**: `magic-pdf` nie byl zadeklarowany, a podpowiedz `mineru[core]` wskazywala zly pakiet. Nowy opcjonalny extra `mineru` (`magic-pdf[cpu]==0.6.1`, Python < 3.13) — instalacja: `uv sync --extra dev --extra mineru`. `SystemExit` z magic-pdf przy bledzie importu modeli nie zabija juz serwera

### Zmieniono
- **Wersja projektu**: 1.3.1 -> 1.3.2
- `uv.lock`: na Pythonie 3.11/3.12 numpy przypiety do 1.26.x dla wszystkich instalacji (paddleocr/imgaug z extra `mineru` nie dzialaja z NumPy 2; uv rozwiazuje extras razem z baza). Na Pythonie >= 3.13 bez zmian (numpy 2.3.x)
- `QueryService` nie wola juz `load()` na jawnie przekazanym `bm25_index` — za zaladowanie indeksu odpowiada jego wlasciciel
- Testy MinerU w `tests/test_parser_properties.py` sa pomijane, gdy `magic_pdf` nie jest zainstalowany
- Nowe testy regresyjne: `tests/test_spa_static_files.py`, `tests/test_query_citations.py`, `tests/test_shared_bm25_index.py`, `tests/test_rag_anything_parser.py` oraz rozszerzone `tests/retrieval/test_reranker.py`

## [1.3.1] - 2026-02-25

### Bezpieczenstwo
- **JSON zamiast insecure deserialization w BM25 (CWE-502)**: Zamiana niebezpiecznej deserializacji na `json.dump/load` w `bm25_index.py` — eliminacja ryzyka RCE. Backward-compatible: automatyczna jednorazowa migracja ze starego formatu do .json

### Naprawiono
- **Zunifikowany SearchResult**: Eliminacja 3 osobnych klas `SearchResult` (vector_store, reranker, models/search) — jedna definicja w `app/models/search.py` z property `relevance_score` jako alias `score`. Usuniecie ~30 linii glue code konwersji z `query_service.py`
- **Sync operacje w async kontekscie**: Opakowanie synchronicznych wywolan ChromaDB (`collection.add/query/get/delete/count`) i file I/O (`write_bytes`) w `asyncio.to_thread()` — zapobiega blokowaniu event loop

### Zmieniono
- **Wersja projektu**: 1.3.0 -> 1.3.1
- Import `SearchResult` we wszystkich modulach teraz z `app.models.search` (single source of truth)
- Sciezka persystencji BM25: `bm25_index.json` (nowy format)

## [1.3.0] - 2026-02-25

### Bezpieczenstwo
- **SecretStr fix w klientach OpenAI**: Naprawienie krytycznego bugu - vlm_client.py, multimodal_embedder.py i openai_pdf_parser.py przekazywaly obiekt `SecretStr` zamiast `.get_secret_value()` do OpenAI SDK
- **Security headers middleware**: Dodanie naglowkow X-Content-Type-Options, X-Frame-Options, X-XSS-Protection do wszystkich odpowiedzi API

### Naprawiono
- **ChromaDB distance→similarity**: Poprawienie blednego wzoru `1 - distance/2` na poprawny `1 - distance` (cosine distance → cosine similarity)
- **BM25 tokenizacja**: Dodanie usuwania interpunkcji, filtrowania stop words (30+ slow) i minimalnej dlugosci tokenu (2 znaki)
- **Reranker normalizacja**: Zmiana arbitralnego fallback `0.5` na `1.0` gdy wszystkie scores sa rowne (rownie relevantne = max score)
- **RRF duplikacja kodu**: Wyodrebnienie wspolnej funkcji `reciprocal_rank_fusion` do `app/retrieval/rrf.py`, usuniecie duplikatu z query_service.py
- **Magic numbers**: Wyekstrahowanie stalych MAX_HYDE_TOKENS, MAX_MULTI_QUERY_TOKENS, MAX_CACHE_SIZE, CACHE_EVICTION_KEEP w QueryExpander

### Dodano
- `app/retrieval/rrf.py`: Wspolna implementacja Reciprocal Rank Fusion uzywana przez hybrid_search i query_service

### Zmieniono
- **Wersja projektu**: 1.2.2 -> 1.3.0
- Usuniecie nieuzywanych importow `TYPE_CHECKING` z vector_store.py i reranker.py

## [1.2.2] - 2026-02-24

### Naprawiono
- **EmbeddedChunk modality**: Dodanie brakujacego pola `modality="text"` we wszystkich fixture'ach tworzacych `EmbeddedChunk` (test_vector_store.py, test_vector_store_properties.py)
- **Test API ChromaDB conflict**: Naprawienie konfliktu embedding function ChromaDB przez mockowanie lifespan w test_api_properties.py (zamiast inicjalizacji prawdziwych serwisow)
- **Mock settings niekompletne**: Dodanie brakujacych atrybutow `text_collection`, `bm25_k1`, `bm25_b`, `enable_hybrid_search` oraz poprawnego `openai_api_key` (SecretStr mock) w test_document_service_properties.py
- **Mock chunker chunk_with_structure**: Dodanie mockowania `chunk_with_structure` obok `chunk_document` w test_document_service_properties.py (produkcyjny kod uzywa `chunk_with_structure`)
- **Parser numpy incompatibility**: Oznaczenie 5 testow parsera jako `xfail` z powodu pre-existing niezgodnosci numpy 2.0 z imgaug/paddleocr (`np.sctypes` removed)

### Zmieniono
- **Wersja projektu**: 1.2.1 -> 1.2.2

## [1.2.1] - 2026-02-24

### Bezpieczenstwo
- **CORS configurable origins**: Zamiana hardkodowanego `allow_origins=["*"]` na konfigurowalne `settings.cors_origins` (domyslnie localhost:3000, localhost:8000)
- **Path traversal protection**: Dodanie walidacji `resolve().relative_to()` w `FileStorageService.get_storage_path()` zapobiegajacej atakowi path traversal
- **SecretStr dla API key**: Zmiana `openai_api_key` z `str | None` na `SecretStr | None` -- klucz API nie jest widoczny w logach/repr
- **raise from**: Dodanie `from e` do wszystkich `raise HTTPException` w blokach `except` (app/main.py, app/storage/file_storage.py)

### Dodano
- `app/models/search.py`: Zunifikowany model `SearchResult` (dataclass) jako single source of truth dla wynikow wyszukiwania
- `tests/test_security.py`: Testy bezpieczenstwa (path traversal, CORS, SecretStr)
- `tests/test_search_result_model.py`: Testy modelu SearchResult (unit + property-based z Hypothesis)
- `cors_origins` field w Settings (konfigurowany przez `CORS_ORIGINS` env var)

### Naprawiono
- BM25 tokenizacja: Dodanie usuwania interpunkcji i filtrowania krotkich tokenow (min 2 znaki)
- `zip()` z `strict=True` w krytycznych miejscach (bm25_index, document_service, query_service)
- Niekompletne mock fixtures w testach (brakujace atrybuty settings, chunk_with_structure)
- Test `hyde_expand` -- aktualizacja oczekiwan do `list[str]` (zgodnie z aktualnym API)

### Zmieniono
- **Wersja projektu**: 1.2.0 -> 1.2.1
- Magic numbers w `query_expansion.py` wyekstrahowane do stalych klasy (`MAX_CACHE_SIZE`, `CACHE_EVICTION_KEEP`)
- Ulepszona instrukcja HyDE prompt (bardziej specyficzna, zachowuje jezyk pytania)

## [1.2.0] - 2025-12-29

### Dodano
- **Deployment na GCP Cloud Run**: Pełna konfiguracja deploymentu w projekcie halobotics
- `cloudbuild.yaml`: Automatyczny build i deploy przez Cloud Build
- `deploy.sh`: Interaktywny skrypt deploymentu z dwoma metodami (Cloud Build / Local Docker)
- `.gcloudignore`: Optymalizacja uploadu do Cloud Build
- `DEPLOYMENT.md`: Kompleksowa dokumentacja deploymentu z:
  - Przewodnik krok po kroku
  - Konfiguracja Cloud Run (4GiB RAM, 2 vCPU, autoscaling 0-10)
  - Integracja Secret Manager dla OpenAI API Key
  - Monitoring i debugging
  - Troubleshooting i best practices
  - Opcje persistent storage (GCS, Cloud SQL, Vertex AI)
  - Continuous deployment setup

### Zmieniono
- **Wersja projektu**: 1.1.0 → 1.2.0
- Dockerfile: Przygotowany dla Cloud Run (port 8080, zmienne środowiskowe)
- Multi-stage build: Frontend (Node 18) + Backend (Python 3.11)

### Infrastruktura
- **Region**: europe-central2 (Warsaw)
- **Service**: pdf-rag-system
- **Endpoint**: HTTPS z auto-managed certificate
- **Authentication**: Unauthenticated (można zmienić dla produkcji)
- **Secrets**: OpenAI API Key w Secret Manager
- **Storage**: Ephemeral (wymaga persistent storage dla produkcji)

### Koszty (szacunkowe)
- ~$12/miesiąc dla 100k requestów (+ koszty OpenAI API)
- Pay-per-use model (skalowanie do zera)
- Pierwsze 2M requestów/miesiąc gratis (free tier)

## [1.1.0] - 2025-12-18

### Dodano
- Multi-query reranking: Reranking wykonywany dla każdego wariantu query expansion osobno
- Reciprocal Rank Fusion (RRF): Algorytm łączący wyniki z wielu rerankingów
- BGE-reranker-v2-m3: Multilingual cross-encoder model jako domyślny reranker
- Funkcja `_reciprocal_rank_fusion()` w QueryService (app/services/query_service.py:398-453)
- Szczegółowa dokumentacja problemu retrieval w ROZWIAZANIE_RETRIEVAL.md
- Skrypty diagnostyczne:
  - debug_full_pipeline.py - Debug krok po kroku całego pipeline
  - debug_rrf_fusion.py - Debug RRF fusion szczegółowo
  - debug_reranking.py - Analiza rankingu przed i po rerankingu
  - test_bge_reranker.py - Test BGE-reranker-v2-m3
  - test_final_comparison.py - Porównanie wszystkich metod
  - test_rrf_reranking.py - Test nowej implementacji RRF
  - test_increased_topk.py - Test zwiększonego top_k
  - test_without_reranking.py - Test bez rerankingu

### Zmieniono
- **Domyślny reranker model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` → `BAAI/bge-reranker-v2-m3`
- **reranking_top_k**: 20 → 40 (większa pula kandydatów)
- **final_top_k**: 5 → 10 (więcej wyników finalnych)
- Reranking respektuje parametr `request.top_k` (query_service.py:241)
- Logika rerankingu w QueryService - dodano warunkową ścieżkę dla multi-query (query_service.py:243-269)

### Naprawiono
- Problem z niepełnymi odpowiedziami dla pytań o szczegóły (np. "Co jest przedmiotem ubezpieczenia?")
- MS-MARCO cross-encoder usuwał właściwe chunki z listami punktowanymi
- Reranking ignorował warianty z query expansion (HyDE, Multi-Query)
- Parametr `top_k` z request był ignorowany przez reranking
- Chunk z kluczowymi informacjami wypada z top-10 po rerankingu

### Wydajność
- Reranking dla pojedynczego query: ~2-3s
- Reranking z RRF (5 wariantów): ~15-23s
- BEZ rerankingu: ~11s
- Trade-off: BGE-reranker-v2-m3 wolniejszy (~2x) ale znacznie lepsza jakość

### Wyniki
- **Przed (MS-MARCO)**: 1-2/4 kluczowe frazy w odpowiedzi
- **Po (BGE-reranker-v2-m3 + RRF)**: 4/4 kluczowe frazy w odpowiedzi
- Pełne odpowiedzi z wszystkimi szczegółami
- Chunk "Pojazd (silnikowy...)" teraz konsekwentnie w top-10

## [1.0.0] - 2025-12-17

### Dodano
- Podstawowy system RAG z Adobe PDF Extract API
- Query expansion (HyDE, Multi-Query, Hybrid)
- Hybrid search (vector + BM25)
- Cross-encoder reranking (MS-MARCO)
- Semantic chunking
- OpenAI embeddings (text-embedding-3-small)
- ChromaDB vector store
- FastAPI backend
- React frontend

### Pierwsze wydanie produkcyjne
