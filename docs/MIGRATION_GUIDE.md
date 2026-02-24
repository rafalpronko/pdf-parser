# Migration Guide - RAG System Integration

## 🚨 Breaking Changes

### 1. MinerU jest teraz wymagany

**Przed:**
- System używał PyPDF jako fallback
- MinerU był opcjonalny

**Teraz:**
- MinerU jest **wymagany**
- Brak fallbacku do PyPDF
- System rzuci błąd jeśli MinerU nie jest dostępny

**Akcja wymagana:**
```bash
uv pip install magic-pdf
```

### 2. Nowe dependencies

Dodano nowe biblioteki dla RAG enhancements:

```bash
# Już w pyproject.toml:
- sentence-transformers>=2.2.0  # Dla rerankingu
- rank-bm25>=0.2.2              # Dla keyword search
```

**Akcja wymagana:**
```bash
uv sync
```

## ✅ Nowe Funkcjonalności (Backward Compatible)

### 1. Hybrid Search

Automatycznie włączone jeśli `ENABLE_HYBRID_SEARCH=true` (domyślnie true).

**Nie wymaga zmian w kodzie** - działa automatycznie.

### 2. Query Expansion

Automatycznie włączone jeśli `ENABLE_QUERY_EXPANSION=true` (domyślnie true).

**Nie wymaga zmian w kodzie** - działa automatycznie.

### 3. Reranking

Automatycznie włączone jeśli `ENABLE_RERANKING=true` (domyślnie true).

**Nie wymaga zmian w kodzie** - działa automatycznie.

## 📝 Aktualizacja .env

Dodaj nowe zmienne (opcjonalne, mają sensowne defaulty):

```bash
# Hybrid Search
ENABLE_HYBRID_SEARCH=true
VECTOR_WEIGHT=0.7
KEYWORD_WEIGHT=0.3
BM25_K1=1.5
BM25_B=0.75

# Query Expansion
ENABLE_QUERY_EXPANSION=true
EXPANSION_METHOD=multi-query  # hyde, multi-query, none
NUM_QUERY_VARIATIONS=3
EXPANSION_CACHE_TTL=3600

# Reranking
ENABLE_RERANKING=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKING_TOP_K=20
FINAL_TOP_K=5

# Performance
ENABLE_GPU=true
RERANKING_BATCH_SIZE=32
CACHE_RERANKING_SCORES=true
```

## 🔄 Migracja Istniejących Danych

### BM25 Index

Jeśli masz już dokumenty w vector store:

1. **Opcja A: Automatyczna odbudowa** (zalecane)
   - System automatycznie zbuduje BM25 index przy pierwszym query
   - Może być wolniejsze przy pierwszym użyciu

2. **Opcja B: Ręczna odbudowa**
   ```python
   from app.services.document_service import DocumentService
   from app.retrieval.bm25_index import BM25Index
   
   # Pobierz wszystkie chunki z vector store
   chunks = await vector_store.get_all_chunks()
   
   # Odbuduj BM25 index
   bm25_index = BM25Index(persist_path="./data/vectordb/bm25_index.pkl")
   bm25_index.rebuild_from_chunks(chunks)
   bm25_index.save()
   ```

### Vector Store

**Nie wymaga migracji** - istniejące dane działają bez zmian.

## 🧪 Testowanie Po Migracji

### 1. Sprawdź czy MinerU działa

```bash
uv run python -c "from magic_pdf.pipe.UNIPipe import UNIPipe; print('✅ MinerU OK')"
```

### 2. Sprawdź czy app się uruchamia

```bash
uv run python -c "from app.main import app; print('✅ App OK')"
```

### 3. Uruchom testy

```bash
uv run pytest tests/retrieval/ -v
```

Powinno być: **27 passed**

### 4. Sprawdź health endpoint

```bash
# Uruchom serwer
uvicorn app.main:app --reload

# W innym terminalu
curl http://localhost:8000/health
```

Powinno zwrócić:
```json
{
  "status": "healthy",
  "service": "RAG-Anything Multimodal System",
  "capabilities": {
    "multimodal": true,
    "rag_anything": true,
    ...
  }
}
```

## 🐛 Troubleshooting

### MinerU ImportError

```
ImportError: MinerU is required but not available
```

**Rozwiązanie:**
```bash
uv pip install magic-pdf
```

### sentence-transformers ImportError

```
ImportError: sentence-transformers not installed
```

**Rozwiązanie:**
```bash
uv pip install sentence-transformers
```

### BM25 index corrupted

```
WARNING: Corruption detected: doc_ids and texts length mismatch
```

**Rozwiązanie:**
System automatycznie przebuduje index. Jeśli problem persystuje:
```bash
rm ./data/vectordb/bm25_index.pkl
# System utworzy nowy index przy następnym użyciu
```

### Reranker model download fails

```
ERROR: Failed to load cross-encoder model
```

**Rozwiązanie:**
System automatycznie użyje fallback (oryginalne scores). Sprawdź połączenie internetowe lub:
```bash
# Pobierz model ręcznie
uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

## 📊 Weryfikacja Działania

### Test 1: Upload dokumentu

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@test.pdf"
```

Sprawdź logi - powinny zawierać:
```
INFO | Adding document 'xxx' to BM25 index
INFO | BM25 index saved successfully
```

### Test 2: Query z enhancements

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "test query", "top_k": 5}'
```

Sprawdź logi - powinny zawierać:
```
INFO | QueryService initialized with enhancements: hybrid_search=True, query_expansion=True, reranking=True
DEBUG | Expanding query
DEBUG | Using hybrid search
DEBUG | Reranking X results
```

## ✅ Checklist Migracji

- [ ] Zainstalowano MinerU: `uv pip install magic-pdf`
- [ ] Zaktualizowano dependencies: `uv sync`
- [ ] Dodano nowe zmienne do .env (opcjonalne)
- [ ] Uruchomiono testy: `uv run pytest tests/retrieval/ -v`
- [ ] Sprawdzono import app: `uv run python -c "from app.main import app"`
- [ ] Przetestowano upload dokumentu
- [ ] Przetestowano query
- [ ] Sprawdzono logi pod kątem błędów

## 🎉 Gotowe!

Po wykonaniu tych kroków system powinien działać z pełną integracją RAG enhancements.

Jeśli napotkasz problemy, sprawdź:
1. `INTEGRATION_SUMMARY.md` - szczegóły techniczne
2. `RAG_ENHANCEMENTS_README.md` - dokumentacja komponentów
3. Logi aplikacji - szczegółowe informacje o błędach
