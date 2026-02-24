# Changelog

Wszystkie istotne zmiany w projekcie będą dokumentowane w tym pliku.

Format bazuje na [Keep a Changelog](https://keepachangelog.com/pl/1.0.0/),
a projekt stosuje [Semantic Versioning](https://semver.org/lang/pl/).

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
