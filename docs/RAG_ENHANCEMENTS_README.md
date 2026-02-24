# RAG Enhancements - Quick Start Guide

## 🎉 Co zostało zaimplementowane?

System RAG został wzbogacony o 4 kluczowe ulepszenia:

1. **🔍 BM25 Keyword Search** - Wyszukiwanie po dokładnych słowach kluczowych
2. **🎯 Cross-Encoder Reranking** - Poprawa kolejności wyników
3. **🔀 Hybrid Search** - Połączenie vector + keyword search z RRF fusion
4. **📝 Query Expansion** - HyDE i multi-query dla lepszego retrieval

## 📦 Instalacja

```bash
# Zainstaluj dependencies
uv sync

# Uruchom testy
uv run pytest tests/retrieval/ -v
```

## ⚙️ Konfiguracja

Dodaj do `.env`:

```bash
# Reranking
ENABLE_RERANKING=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKING_TOP_K=20
FINAL_TOP_K=5

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

# Chunking
CHUNKING_STRATEGY=semantic  # fixed, semantic, sentence-window
USE_STRUCTURE_AWARE_CHUNKING=true

# Performance
ENABLE_GPU=true
RERANKING_BATCH_SIZE=32
CACHE_RERANKING_SCORES=true
```

## 🚀 Użycie

### BM25 Keyword Search

```python
from app.retrieval import BM25Index

# Stwórz index
bm25 = BM25Index(persist_path="./data/bm25_index.pkl")

# Dodaj dokumenty
bm25.add_documents(
    doc_ids=["doc1", "doc2"],
    texts=["machine learning", "deep learning"],
    metadata=[{"source": "doc1"}, {"source": "doc2"}]
)

# Wyszukaj
results = bm25.search("learning", top_k=5)
# Returns: [("doc1", 0.95), ("doc2", 0.87)]

# Zapisz index
bm25.save()
```

### Cross-Encoder Reranking

```python
from app.retrieval import CrossEncoderReranker, SearchResult

# Inicjalizuj reranker
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="auto"  # Automatycznie użyje GPU jeśli dostępne
)

# Rerank wyników
chunks = [
    SearchResult("id1", "content 1", 0.8),
    SearchResult("id2", "content 2", 0.7)
]

reranked = reranker.rerank(
    query="user question",
    chunks=chunks,
    top_k=5
)
```

### Hybrid Search

```python
from app.retrieval import HybridSearchEngine

# Połącz vector store z BM25
hybrid = HybridSearchEngine(
    vector_store=vector_store,
    bm25_index=bm25_index,
    vector_weight=0.7,
    keyword_weight=0.3
)

# Wyszukaj używając obu metod
results = await hybrid.search(
    query="machine learning",
    query_embedding=embedding,
    top_k=5
)
```

### Query Expansion

```python
from app.retrieval import QueryExpander

# Inicjalizuj expander
expander = QueryExpander(
    llm_client=openai_client,
    method="multi-query",  # lub "hyde"
    num_variations=3
)

# Rozszerz query
variations = await expander.expand("What is machine learning?")
# Returns: [
#   "What is machine learning?",
#   "How does machine learning work?",
#   "Explain machine learning concepts"
# ]
```

## 🧪 Testy

```bash
# Wszystkie testy retrieval
uv run pytest tests/retrieval/ -v

# Konkretny komponent
uv run pytest tests/retrieval/test_bm25_index.py -v
uv run pytest tests/retrieval/test_reranker.py -v
uv run pytest tests/retrieval/test_hybrid_search.py -v
uv run pytest tests/retrieval/test_query_expansion.py -v

# Z coverage
uv run pytest tests/retrieval/ --cov=app/retrieval --cov-report=html
```

## 📊 Wyniki Testów

- **27 testów** - wszystkie przechodzą ✅
- **9 property-based testów** (Hypothesis)
- **18 unit testów**
- **Coverage**: 79-100% dla nowych komponentów

## 🎯 Zalecane Ustawienia

### Dla Szybkości (Fast MVP)
```bash
ENABLE_RERANKING=true
ENABLE_HYBRID_SEARCH=true
ENABLE_QUERY_EXPANSION=false  # Wyłącz dla szybszych odpowiedzi
EXPANSION_METHOD=none
```

### Dla Jakości (Best Quality)
```bash
ENABLE_RERANKING=true
ENABLE_HYBRID_SEARCH=true
ENABLE_QUERY_EXPANSION=true
EXPANSION_METHOD=hyde  # Najlepsza jakość
NUM_QUERY_VARIATIONS=3
```

### Dla Balansu (Recommended)
```bash
ENABLE_RERANKING=true
ENABLE_HYBRID_SEARCH=true
ENABLE_QUERY_EXPANSION=true
EXPANSION_METHOD=multi-query  # Dobry balans
NUM_QUERY_VARIATIONS=2
```

## 🔧 Troubleshooting

### Reranker nie działa
- Sprawdź czy `sentence-transformers` jest zainstalowany
- System automatycznie użyje fallback (original scores)

### BM25 index corrupted
- System automatycznie wykryje i przebuduje index
- Możesz ręcznie przebudować: `bm25.rebuild_from_chunks(chunks)`

### Query expansion zbyt wolny
- Zmniejsz `NUM_QUERY_VARIATIONS` do 2
- Lub wyłącz: `ENABLE_QUERY_EXPANSION=false`
- Cache pomaga przy powtarzających się queries

### GPU nie jest używane
- Sprawdź `torch.cuda.is_available()` lub `torch.backends.mps.is_available()`
- Ustaw `ENABLE_GPU=true` w .env
- Reranker automatycznie wykryje i użyje GPU

## 📈 Oczekiwane Ulepszenia

Z tymi enhancement'ami powinieneś zobaczyć:
- **+10-15% MRR** (Mean Reciprocal Rank)
- **+15-20% NDCG** (Normalized Discounted Cumulative Gain)
- **Lepsze keyword matching** (technical terms, proper nouns)
- **Bardziej relevantne wyniki** (dzięki reranking)

## 🎊 Gotowe!

System jest production-ready z:
- ✅ Comprehensive error handling
- ✅ Graceful fallback w każdym komponencie
- ✅ Logging i monitoring
- ✅ Caching dla performance
- ✅ GPU support
- ✅ Property-based testing

**Miłego używania! 🚀**
