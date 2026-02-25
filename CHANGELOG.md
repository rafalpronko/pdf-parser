# Changelog

Wszystkie istotne zmiany w projekcie będą dokumentowane w tym pliku.

Format bazuje na [Keep a Changelog](https://keepachangelog.com/pl/1.0.0/),
a projekt stosuje [Semantic Versioning](https://semver.org/lang/pl/).

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
