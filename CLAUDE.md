# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Zasady Ogólne

- **Zawsze używaj `uv` do uruchamiania skryptów Pythonowych** (np. `uv run pytest`, `uv run uvicorn`)
- **Po każdej zmianie kodu aktualizuj wersję projektu** (w `pyproject.toml`) oraz ewentualny plik CHANGELOG
- **Nie dodawaj do commit message**: generycznych wiadomości o Claude Code ani linków do https://claude.ai/code
- **Odpowiadaj w języku polskim** zgodnie z ustawieniami użytkownika

## Komendy Deweloperskie

### Uruchamianie Serwera
```bash
# Uruchom serwer FastAPI z hot-reload
uv run uvicorn app.main:app --reload

# Alternatywnie użyj skryptu startowego
./start_server.sh
```

API będzie dostępne na `http://localhost:8000`. Dokumentacja:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Testowanie
```bash
# Uruchom wszystkie testy
uv run pytest

# Uruchom testy z coverage
uv run pytest --cov=app --cov-report=html

# Uruchom konkretny plik testowy
uv run pytest tests/test_models.py

# Uruchom tylko testy property-based (Hypothesis)
uv run pytest tests/ -k "property"

# Uruchom testy z verbose output
uv run pytest -v

# Uruchom pojedynczy test
uv run pytest tests/test_models.py::test_query_request_validation -v
```

### Code Quality
```bash
# Formatowanie kodu (zawsze przed commitem)
uv run ruff format .

# Linting
uv run ruff check .

# Napraw automatycznie błędy lintowania
uv run ruff check --fix .

# Type checking (opcjonalnie)
uv run mypy app/
```

### Zarządzanie Zależnościami
```bash
# Instalacja zależności developerskich
uv pip install -e ".[dev]"

# Dodanie nowej zależności (edytuj pyproject.toml, potem):
uv pip install -e .
```

## Architektura Systemu

### Wielowarstwowa Architektura RAG

System implementuje **Retrieval-Augmented Generation (RAG)** z rozszerzonymi funkcjami:

```
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                      │
│  - endpoints w app/main.py                                   │
│  - walidacja Pydantic (models/)                              │
│  - middleware: CORS, request ID, error handling              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                             │
│  - DocumentService: przetwarzanie dokumentów PDF             │
│  - QueryService: pipeline RAG z enhancement'ami              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Processing Layer                           │
│  - Parsing: RAG-Anything (MinerU) dla PDF                    │
│  - Chunking: semantic/fixed/sentence-window                  │
│  - Embedding: OpenAI text-embedding-3-small                  │
│  - Multimodal: CLIP vision encoder + VLM (gpt-4-vision)      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Retrieval Layer (Enhanced)                   │
│  - Hybrid Search: Vector (ChromaDB) + BM25                   │
│  - Query Expansion: HyDE / Multi-Query                       │
│  - Cross-Encoder Reranking: ms-marco-MiniLM                  │
│  - RRF (Reciprocal Rank Fusion)                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  - Vector Store: ChromaDB (3 kolekcje)                       │
│  - File Storage: ./data/uploads                              │
│  - BM25 Index: ./data/vectordb/bm25_index.pkl                │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Przetwarzania Dokumentu

```
PDF Upload
    ↓
Walidacja (format, rozmiar)
    ↓
Parsing (MinerU RAG-Anything)
    ↓ (tekst + obrazy + tabele)
Chunking Strategia
    ├── semantic (domyślna - z użyciem sentence embeddings)
    ├── fixed (stały rozmiar z overlap)
    └── sentence-window (okna zdań)
    ↓
Embedding Generation
    ├── Text: text-embedding-3-small
    └── Visual: CLIP-ViT-Large
    ↓
Storage w ChromaDB
    ├── text_chunks (tekst)
    ├── visual_chunks (obrazy)
    └── multimodal_chunks (połączone)
```

### Pipeline Zapytania (RAG)

```
User Query
    ↓
Query Expansion (opcjonalnie)
    ├── HyDE: generuj hipotetyczny dokument
    └── Multi-Query: generuj 3 warianty zapytania
    ↓
Embedding Query
    ↓
Hybrid Search
    ├── Vector Search (ChromaDB similarity)
    └── BM25 Keyword Search
    ↓
RRF Fusion (połącz wyniki)
    ↓
Cross-Encoder Reranking (opcjonalnie)
    ↓ (top 20 → top 5)
Context Construction
    ↓
LLM Generation (gpt-4o-mini)
    ↓
Response + Citations
```

## Kluczowe Moduły

### app/config.py
- **Settings**: konfiguracja z walidacją Pydantic
- Wczytuje zmienne z `.env`
- Fail-fast walidacja przy starcie
- Kluczowe sekcje:
  - LLM/VLM (OpenAI)
  - RAG Enhancements (hybrid search, reranking, query expansion)
  - Chunking Strategy (semantic/fixed/sentence-window)
  - Performance (GPU, batch size, caching)

### app/services/query_service.py
- **QueryService**: główna logika RAG pipeline
- Integracja:
  - Query expansion (HyDE lub multi-query)
  - Hybrid search (vector + BM25)
  - Cross-encoder reranking
- Obsługuje puste wyniki i multi-source synthesis

### app/services/document_service.py
- **DocumentService**: orkiestracja przetwarzania dokumentów
- Workflow:
  1. Walidacja pliku (rozmiar, format)
  2. Parsing (RAG-Anything)
  3. Chunking (wybrana strategia)
  4. Embedding (batch processing)
  5. Storage (ChromaDB + file system)
  6. BM25 index update

### app/retrieval/
Zaawansowane moduły retrieval:

- **hybrid_search.py**: łączy vector search + BM25 keyword search
  - RRF (Reciprocal Rank Fusion) dla mergowania wyników
  - Konfigurowalne wagi (vector_weight, keyword_weight)

- **query_expansion.py**: rozszerza zapytania dla lepszego retrievalu
  - HyDE (Hypothetical Document Embeddings)
  - Multi-Query (generuje warianty zapytania)
  - Cache z TTL

- **reranker.py**: cross-encoder dla dokładniejszego rankingu
  - Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - GPU acceleration (jeśli dostępne)
  - Batch processing + caching

- **bm25_index.py**: keyword search z BM25
  - Parametry k1, b konfigurowalne
  - Persist/load funkcjonalność

### app/processing/
- **chunker.py**: strategie podziału dokumentów
  - Semantic chunking (embedding-based)
  - Fixed size chunking
  - Sentence-window chunking
  - Structure-aware (headers, paragraphs)

- **multimodal/**: obsługa treści wizualnych
  - MultimodalChunker: łączy tekst + obrazy
  - MultimodalEmbedder: CLIP embeddings dla obrazów
  - VLM integration (GPT-4V)

### app/storage/vector_store.py
- ChromaDB wrapper z 3 kolekcjami
- Batch upsert dla wydajności
- Metadata filtering
- Podobieństwo kosinusowe dla retrieval

## Konfiguracja Środowiska (.env)

Wymagane zmienne:
```env
# LLM
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# RAG Enhancements (zalecane włączone)
ENABLE_HYBRID_SEARCH=true
ENABLE_QUERY_EXPANSION=true
ENABLE_RERANKING=true
EXPANSION_METHOD=multi-query  # lub 'hyde'

# Chunking
CHUNKING_STRATEGY=semantic  # lub 'fixed', 'sentence-window'
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Multimodal (opcjonalne)
ENABLE_MULTIMODAL_CHUNKING=true
ENABLE_VLM=true
VLM_PROVIDER=openai
VISION_ENCODER=openai/clip-vit-large-patch14
```

Zobacz `.env.example` dla pełnej listy.

## Testowanie Property-Based (Hypothesis)

System używa **Hypothesis** dla testów property-based. Przykłady w:
- `tests/test_chunker_properties.py`: właściwości chunków
- `tests/test_config_properties.py`: walidacja konfiguracji
- `tests/test_query_service_properties.py`: właściwości RAG pipeline

Property-based testy weryfikują niezmienniki:
- Chunki nie przekraczają max rozmiaru
- Overlap jest zgodny z konfiguracją
- Embeddingi mają poprawne wymiary
- Wyniki są deterministyczne dla tych samych inputów

## Rozszerzanie Systemu

### Dodanie Nowego Parsera
Zaimplementuj `BaseParser` interface w `app/parsers/`:
```python
from app.parsers import BaseParser

class MyParser(BaseParser):
    def parse(self, file_content: bytes) -> ParsedDocument:
        # implementacja
        pass
```

### Dodanie Nowej Strategii Chunkingu
Rozszerz `app/processing/chunker.py`:
```python
def _chunk_by_custom_strategy(
    self, text: str, metadata: dict
) -> list[dict[str, Any]]:
    # implementacja
    pass
```

Dodaj do `chunking_strategy` w config.py walidatorze.

### Dodanie Nowego Retrieval Enhancement
Utwórz moduł w `app/retrieval/` i zintegruj z `QueryService`.

## Struktura Danych

### DocumentMetadata (models/document.py)
```python
filename: str
content_type: str
tags: list[str]
description: str | None
created_at: datetime
```

### Chunk (models/chunk.py)
```python
doc_id: str
chunk_index: int
content: str
metadata: dict  # zawiera page, section, embedding_model, etc.
```

### QueryRequest (models/query.py)
```python
question: str
top_k: int = 5
temperature: float = 0.7
include_sources: bool = True
```

## Performance & Best Practices

1. **Batch Processing**: embeddingi generowane w batch'ach (domyślnie 100)
2. **Async/Await**: wszystkie I/O operacje async
3. **Connection Pooling**: HTTP połączenia do OpenAI są reużywane
4. **GPU Acceleration**: reranking używa GPU jeśli dostępne
5. **Caching**:
   - Query expansion cache (TTL: 3600s)
   - Reranking scores cache
6. **Hybrid Search**: vector (70%) + keyword (30%) dla balansu precyzji/recall

## Debug & Troubleshooting

### Włącz Debug Logging
```bash
LOG_LEVEL=DEBUG uv run uvicorn app.main:app --reload
```

### Sprawdź ChromaDB Collections
```python
from app.storage.vector_store import VectorStore
store = VectorStore(persist_directory="./data/vectordb")
print(f"Total chunks: {store.collection.count()}")
```

### Testuj Query Pipeline
```bash
# Użyj interactive Python shell
uv run python
>>> from app.services.query_service import QueryService
>>> service = QueryService()
>>> # testuj poszczególne komponenty
```

## Logowanie

System używa structured logging z request ID:
- `app/logging_config.py`: konfiguracja loggera
- Każde żądanie ma unikalny request_id
- Context variables dla request ID w logach
- Poziomy: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Frontend (Opcjonalny)

Chat UI w React:
```bash
./start_frontend.sh
# Dostępne na http://localhost:3000
```

Features:
- Interaktywny chat interface
- Lista dokumentów
- Source citations z linkami
- Processing time monitoring
