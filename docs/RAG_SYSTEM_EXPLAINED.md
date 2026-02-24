# Jak Działa RAG System - Kompletne Wyjaśnienie

## 🎯 Przegląd Systemu

Twój RAG (Retrieval-Augmented Generation) system to zaawansowany pipeline, który:
1. **Przetwarza dokumenty** (PDF, obrazy, tekst)
2. **Przechowuje je** w bazie wektorowej
3. **Wyszukuje relevantne fragmenty** na podstawie zapytań użytkownika
4. **Generuje odpowiedzi** używając LLM z kontekstem

## 📊 Architektura - Główne Komponenty

```
┌─────────────────────────────────────────────────────────────┐
│                    UŻYTKOWNIK                                │
│              (wysyła dokument lub zapytanie)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   API ENDPOINTS                              │
│  • POST /documents/upload  - upload dokumentu                │
│  • POST /query            - zapytanie do systemu             │
│  • GET  /health           - status systemu                   │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│ DOCUMENT SERVICE │    │  QUERY SERVICE   │
│  (przetwarzanie) │    │   (wyszukiwanie) │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         │                       │
    ┌────▼────┐            ┌────▼────┐
    │ PARSERS │            │RETRIEVAL│
    │         │            │ ENGINE  │
    └────┬────┘            └────┬────┘
         │                      │
         ▼                      ▼
┌─────────────────────────────────────┐
│      VECTOR DATABASE (ChromaDB)      │
│   • Text embeddings                  │
│   • Visual embeddings                │
│   • Metadata                         │
└─────────────────────────────────────┘
```

## 🔄 Flow 1: Przetwarzanie Dokumentu

### Krok po kroku:

```
1. UPLOAD DOKUMENTU
   ↓
   User wysyła PDF → POST /documents/upload
   
2. PARSING
   ↓
   RAGAnythingParser:
   • Wykrywa typ dokumentu
   • Ekstraktuje tekst (pypdf lub MinerU)
   • Wykrywa obrazy i tabele
   • Zachowuje strukturę (headings, paragraphs)
   
3. CHUNKING (Semantic Chunking)
   ↓
   SemanticChunker:
   • Dzieli dokument na semantyczne fragmenty
   • Respektuje granice (paragrafy, sekcje)
   • Opcjonalnie: sentence-window (małe chunki + kontekst)
   • Dodaje metadata (pozycja, typ, struktura)
   
4. EMBEDDING
   ↓
   OpenAI Embedder:
   • Generuje embeddingi dla każdego chunka
   • Model: text-embedding-3-small
   • Wymiar: 1536
   
5. INDEXING
   ↓
   Równolegle:
   
   A) Vector Store (ChromaDB):
      • Zapisuje embeddingi
      • Przechowuje metadata
      • Umożliwia semantic search
   
   B) BM25 Index:
      • Tokenizuje tekst
      • Buduje inverted index
      • Umożliwia keyword search
      • Zapisuje na dysk (persistence)
   
6. POTWIERDZENIE
   ↓
   Zwraca document_id i status
```

### Przykład:

```python
# User uploaduje dokument
POST /documents/upload
Content-Type: multipart/form-data
file: research_paper.pdf

# System przetwarza:
{
  "document_id": "doc_abc123",
  "filename": "research_paper.pdf",
  "status": "completed",
  "chunks_created": 45,
  "processing_time": 3.2
}
```

## 🔍 Flow 2: Zapytanie (Query) - Enhanced Retrieval

To jest najważniejsza część! Tutaj dzieje się magia z wszystkimi enhancements:

### Krok po kroku:

```
1. ZAPYTANIE UŻYTKOWNIKA
   ↓
   User: "What are the main findings about climate change?"
   
2. QUERY EXPANSION (opcjonalne, domyślnie włączone)
   ↓
   QueryExpander generuje warianty zapytania:
   
   Metoda: Multi-Query (domyślna)
   • Original: "What are the main findings about climate change?"
   • Variant 1: "What are the key discoveries regarding climate change?"
   • Variant 2: "What are the primary results about global warming?"
   
   LUB Metoda: HyDE
   • Generuje hipotetyczny dokument odpowiadający na pytanie
   • Używa tego jako query
   
   Cache: Zapisuje wyniki (TTL: 1h)
   
3. HYBRID SEARCH (dla każdego wariantu)
   ↓
   Równolegle wykonuje:
   
   A) VECTOR SEARCH (semantic)
      ChromaDB:
      • Embedding zapytania
      • Cosine similarity
      • Top-20 najbardziej podobnych chunków
      • Wynik: [(chunk1, score1), (chunk2, score2), ...]
   
   B) KEYWORD SEARCH (lexical)
      BM25 Index:
      • Tokenizacja zapytania
      • BM25 scoring (k1=1.5, b=0.75)
      • Top-20 najlepiej matchujących chunków
      • Wynik: [(chunk3, score3), (chunk4, score4), ...]
   
4. RECIPROCAL RANK FUSION (RRF)
   ↓
   Łączy wyniki z obu metod:
   
   Formula: RRF_score = Σ 1/(k + rank_i)
   gdzie k=60 (stała)
   
   Weights:
   • Vector: 0.7 (domyślnie)
   • Keyword: 0.3 (domyślnie)
   
   Wynik: Top-20 chunków z najlepszymi combined scores
   
5. RERANKING (Cross-Encoder)
   ↓
   CrossEncoderReranker:
   • Model: ms-marco-MiniLM-L-6-v2
   • Dla każdej pary (query, chunk):
     - Oblicza precyzyjny relevance score
     - Używa attention mechanism
   • Batch processing (32 pary na raz)
   • GPU acceleration (jeśli dostępne)
   • Score normalization [0, 1]
   
   Cache: Zapisuje scores dla par (query, chunk)
   
   Wynik: Top-5 najbardziej relevantnych chunków
   
6. DEDUPLICATION (jeśli query expansion)
   ↓
   Jeśli używano wielu wariantów zapytania:
   • Merge wyników z wszystkich wariantów
   • Usuń duplikaty (ten sam chunk_id)
   • Zachowaj najwyższy score
   
7. CONTEXT PREPARATION
   ↓
   Dla top-5 chunków:
   • Pobierz pełny tekst
   • Pobierz metadata (source, page, position)
   • Formatuj jako kontekst dla LLM
   
8. LLM GENERATION
   ↓
   OpenAI GPT-4o-mini:
   • Prompt: "Based on the following context, answer: {query}"
   • Context: {top 5 chunks}
   • Temperature: 0.7
   • Max tokens: 500
   
9. RESPONSE
   ↓
   Zwraca:
   {
     "answer": "The main findings...",
     "sources": [
       {
         "chunk_id": "chunk_1",
         "document_id": "doc_abc123",
         "text": "...",
         "score": 0.95,
         "page": 3
       },
       ...
     ],
     "metadata": {
       "retrieval_method": "hybrid_reranked",
       "query_expansion": "multi-query",
       "num_candidates": 20,
       "reranking_time": 0.15,
       "total_time": 1.2
     }
   }
```

## 🎛️ Konfiguracja - Co Możesz Kontrolować

### Environment Variables (.env):

```bash
# === LLM Settings ===
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# === Chunking ===
CHUNK_SIZE=512                    # Rozmiar chunka w tokenach
CHUNK_OVERLAP=50                  # Overlap między chunkami
CHUNKING_STRATEGY=semantic        # fixed, semantic, sentence-window

# === Reranking ===
ENABLE_RERANKING=true            # Włącz/wyłącz reranking
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKING_TOP_K=20               # Ile kandydatów do rerankingu
FINAL_TOP_K=5                    # Ile wyników końcowych
ENABLE_GPU=true                  # GPU dla rerankingu

# === Hybrid Search ===
ENABLE_HYBRID_SEARCH=true        # Włącz/wyłącz hybrid search
VECTOR_WEIGHT=0.7                # Waga vector search (0-1)
KEYWORD_WEIGHT=0.3               # Waga keyword search (0-1)
BM25_K1=1.5                      # BM25 term frequency saturation
BM25_B=0.75                      # BM25 length normalization

# === Query Expansion ===
ENABLE_QUERY_EXPANSION=true      # Włącz/wyłącz expansion
EXPANSION_METHOD=multi-query     # hyde, multi-query, none
NUM_QUERY_VARIATIONS=3           # Ile wariantów generować
EXPANSION_CACHE_TTL=3600         # Cache TTL w sekundach

# === Performance ===
RERANKING_BATCH_SIZE=32          # Batch size dla rerankingu
CACHE_RERANKING_SCORES=true      # Cache scores
```

## 🔬 Przykład Działania - Krok po Kroku

### Scenariusz: User pyta o "machine learning algorithms"

```
1. Query Expansion:
   Original: "machine learning algorithms"
   Variant 1: "ML algorithm types"
   Variant 2: "supervised and unsupervised learning methods"

2. Hybrid Search (dla każdego wariantu):
   
   Vector Search (semantic):
   • "neural networks and deep learning" - score: 0.85
   • "classification and regression models" - score: 0.82
   • "training data and model optimization" - score: 0.78
   
   Keyword Search (BM25):
   • "machine learning algorithms include..." - score: 12.5
   • "algorithm selection for ML tasks..." - score: 11.2
   • "popular algorithms: SVM, Random Forest..." - score: 10.8

3. RRF Fusion:
   Combined scores (0.7 * vector + 0.3 * keyword):
   • Chunk A: 0.89 (high semantic + keyword match)
   • Chunk B: 0.85 (high semantic)
   • Chunk C: 0.81 (good keyword match)
   ... (top 20)

4. Reranking:
   Cross-encoder scores (query-chunk pairs):
   • Chunk A: 0.95 ⭐ (najbardziej relevant)
   • Chunk D: 0.92 ⭐
   • Chunk B: 0.88 ⭐
   • Chunk E: 0.85 ⭐
   • Chunk C: 0.82 ⭐

5. Final Result:
   Top 5 chunków → LLM → Answer
```

## 🎯 Dlaczego To Działa Lepiej?

### 1. **Query Expansion** - Rozumie intencję
- User może pytać różnie o to samo
- System generuje warianty i znajduje więcej relevant content
- Przykład: "ML" = "machine learning" = "artificial intelligence algorithms"

### 2. **Hybrid Search** - Łączy semantic + keyword
- **Vector search**: rozumie znaczenie, kontekst, synonimy
- **Keyword search**: znajduje exact matches, terminy techniczne
- **Razem**: najlepsze z obu światów

### 3. **Reranking** - Precyzyjne scorowanie
- Pierwszy pass (hybrid): szybki, znajduje kandydatów
- Drugi pass (reranking): dokładny, wybiera najlepsze
- Cross-encoder: patrzy na całą parę (query, chunk) jednocześnie

### 4. **Semantic Chunking** - Zachowuje kontekst
- Nie tnie w środku zdania
- Respektuje strukturę dokumentu
- Chunki mają sens semantyczny

## 📊 Metryki i Monitoring

System loguje wszystko:

```python
# Przykładowe logi:
INFO: Query received: "machine learning algorithms"
INFO: Query expansion: generated 3 variations
INFO: Vector search: found 20 candidates in 0.05s
INFO: Keyword search: found 18 candidates in 0.02s
INFO: RRF fusion: merged to 25 unique chunks
INFO: Reranking: scored 25 pairs in 0.15s (GPU)
INFO: Final results: 5 chunks, avg score: 0.89
INFO: LLM generation: 1.2s
INFO: Total query time: 1.5s
```

Debug mode (dodaj `?debug=true`):
```json
{
  "metadata": {
    "timings": {
      "query_expansion": 0.08,
      "vector_search": 0.05,
      "keyword_search": 0.02,
      "rrf_fusion": 0.01,
      "reranking": 0.15,
      "llm_generation": 1.2
    },
    "scores": {
      "vector_contribution": 0.65,
      "keyword_contribution": 0.35,
      "reranking_improvement": 0.12
    },
    "candidates": {
      "initial": 25,
      "after_reranking": 5
    }
  }
}
```

## 🔧 Troubleshooting

### Problem: Wyniki nie są relevantne
**Rozwiązanie:**
1. Zwiększ `RERANKING_TOP_K` (więcej kandydatów)
2. Zmień `EXPANSION_METHOD` na `hyde`
3. Dostosuj `VECTOR_WEIGHT` / `KEYWORD_WEIGHT`

### Problem: Zbyt wolne
**Rozwiązanie:**
1. Wyłącz query expansion: `ENABLE_QUERY_EXPANSION=false`
2. Zmniejsz `RERANKING_TOP_K`
3. Włącz GPU: `ENABLE_GPU=true`
4. Zwiększ `RERANKING_BATCH_SIZE`

### Problem: Nie znajduje exact matches
**Rozwiązanie:**
1. Zwiększ `KEYWORD_WEIGHT` (np. 0.5)
2. Sprawdź BM25 index: `GET /health`

## 🎓 Podsumowanie

Twój RAG system to **4-stage pipeline**:

1. **Document Processing** → Parsing + Chunking + Embedding + Indexing
2. **Query Enhancement** → Expansion + Hybrid Search + RRF Fusion
3. **Reranking** → Cross-encoder scoring
4. **Generation** → LLM z kontekstem

**Kluczowe zalety:**
- ✅ Znajduje więcej relevant content (query expansion)
- ✅ Łączy semantic + keyword search (hybrid)
- ✅ Precyzyjne scorowanie (reranking)
- ✅ Zachowuje kontekst (semantic chunking)
- ✅ Szybkie i skalowalne (caching, GPU)
- ✅ Monitorowalne (detailed metrics)

**To jest state-of-the-art RAG system!** 🚀
