# RAG System - Praktyczne Przykłady Użycia

## 🚀 Quick Start

### 1. Uruchomienie Systemu

```bash
# Zainstaluj dependencies
uv sync

# Skonfiguruj .env
cp .env.example .env
# Edytuj .env i dodaj swój OPENAI_API_KEY

# Uruchom serwer
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# Serwer działa na http://localhost:8000
# Dokumentacja API: http://localhost:8000/docs
```

## 📤 Przykład 1: Upload Dokumentu

### Curl:
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@research_paper.pdf"
```

### Python:
```python
import requests

url = "http://localhost:8000/documents/upload"
files = {"file": open("research_paper.pdf", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

### Response:
```json
{
  "document_id": "doc_abc123xyz",
  "filename": "research_paper.pdf",
  "status": "completed",
  "chunks_created": 45,
  "processing_time": 3.2,
  "metadata": {
    "file_size": 2048576,
    "pages": 12,
    "chunking_strategy": "semantic"
  }
}
```

## 🔍 Przykład 2: Podstawowe Query

### Curl:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings about climate change?",
    "top_k": 5
  }'
```

### Python:
```python
import requests

url = "http://localhost:8000/query"
data = {
    "query": "What are the main findings about climate change?",
    "top_k": 5
}

response = requests.post(url, json=data)
result = response.json()

print("Answer:", result["answer"])
print("\nSources:")
for source in result["sources"]:
    print(f"- {source['document_id']} (score: {source['score']:.2f})")
```

### Response:
```json
{
  "answer": "The main findings about climate change include: 1) Global temperatures have risen by 1.1°C since pre-industrial times, 2) Sea levels are rising at an accelerating rate, 3) Extreme weather events are becoming more frequent...",
  "sources": [
    {
      "chunk_id": "chunk_1",
      "document_id": "doc_abc123xyz",
      "text": "Recent studies show that global temperatures...",
      "score": 0.95,
      "metadata": {
        "page": 3,
        "position": 1,
        "document_name": "research_paper.pdf"
      }
    },
    {
      "chunk_id": "chunk_2",
      "document_id": "doc_abc123xyz",
      "text": "Sea level rise has accelerated...",
      "score": 0.92,
      "metadata": {
        "page": 5,
        "position": 2,
        "document_name": "research_paper.pdf"
      }
    }
  ],
  "metadata": {
    "retrieval_method": "hybrid_reranked",
    "query_expansion": "multi-query",
    "total_time": 1.5
  }
}
```

## 🎛️ Przykład 3: Query z Custom Configuration

### Wyłącz Query Expansion (szybsze):
```python
data = {
    "query": "machine learning algorithms",
    "top_k": 5,
    "enable_query_expansion": False  # Wyłącz expansion
}
```

### Użyj tylko Vector Search (bez keyword):
```python
data = {
    "query": "neural networks",
    "top_k": 5,
    "enable_hybrid_search": False  # Tylko vector search
}
```

### Wyłącz Reranking (najszybsze):
```python
data = {
    "query": "deep learning",
    "top_k": 5,
    "enable_reranking": False  # Bez rerankingu
}
```

### Wszystko włączone + Debug:
```python
data = {
    "query": "artificial intelligence",
    "top_k": 5,
    "enable_query_expansion": True,
    "enable_hybrid_search": True,
    "enable_reranking": True,
    "include_sources": True,
    "debug": True  # Dodatkowe metryki
}
```

### Response z Debug:
```json
{
  "answer": "...",
  "sources": [...],
  "metadata": {
    "retrieval_method": "hybrid_reranked",
    "query_expansion": "multi-query",
    "timings": {
      "query_expansion": 0.08,
      "vector_search": 0.05,
      "keyword_search": 0.02,
      "rrf_fusion": 0.01,
      "reranking": 0.15,
      "llm_generation": 1.2,
      "total": 1.51
    },
    "scores": {
      "vector_contribution": 0.65,
      "keyword_contribution": 0.35,
      "reranking_improvement": 0.12
    },
    "candidates": {
      "initial": 25,
      "after_reranking": 5
    },
    "query_variations": [
      "artificial intelligence",
      "AI and machine learning",
      "intelligent systems"
    ]
  }
}
```

## 📊 Przykład 4: Batch Processing

### Upload wielu dokumentów:
```python
import requests
from pathlib import Path

url = "http://localhost:8000/documents/upload"
documents = Path("./documents").glob("*.pdf")

results = []
for doc in documents:
    with open(doc, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
        results.append(response.json())
        print(f"Uploaded: {doc.name} - {response.json()['document_id']}")

print(f"\nTotal uploaded: {len(results)}")
```

### Query z filtrowaniem po dokumencie:
```python
# Zapytaj tylko w konkretnym dokumencie
data = {
    "query": "What is the conclusion?",
    "document_ids": ["doc_abc123xyz"],  # Tylko ten dokument
    "top_k": 5
}
```

## 🔧 Przykład 5: Różne Strategie Wyszukiwania

### Strategia 1: Maksymalna Jakość (wolniejsze)
```python
# .env configuration:
ENABLE_QUERY_EXPANSION=true
EXPANSION_METHOD=hyde
ENABLE_HYBRID_SEARCH=true
ENABLE_RERANKING=true
RERANKING_TOP_K=30
FINAL_TOP_K=5

# Query:
data = {
    "query": "complex technical question",
    "top_k": 5
}
# Czas: ~2-3s, Jakość: ⭐⭐⭐⭐⭐
```

### Strategia 2: Zbalansowana (domyślna)
```python
# .env configuration:
ENABLE_QUERY_EXPANSION=true
EXPANSION_METHOD=multi-query
ENABLE_HYBRID_SEARCH=true
ENABLE_RERANKING=true
RERANKING_TOP_K=20
FINAL_TOP_K=5

# Query:
data = {
    "query": "standard question",
    "top_k": 5
}
# Czas: ~1.5s, Jakość: ⭐⭐⭐⭐
```

### Strategia 3: Maksymalna Szybkość
```python
# .env configuration:
ENABLE_QUERY_EXPANSION=false
ENABLE_HYBRID_SEARCH=false
ENABLE_RERANKING=false

# Query:
data = {
    "query": "simple question",
    "top_k": 5
}
# Czas: ~0.3s, Jakość: ⭐⭐⭐
```

## 🎯 Przykład 6: Use Cases

### Use Case 1: Research Assistant
```python
# Upload research papers
papers = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
for paper in papers:
    upload_document(paper)

# Ask research questions
questions = [
    "What are the main methodologies used?",
    "What are the key findings?",
    "What are the limitations?",
    "What future work is suggested?"
]

for q in questions:
    response = query(q, top_k=5)
    print(f"\nQ: {q}")
    print(f"A: {response['answer']}")
```

### Use Case 2: Technical Documentation Search
```python
# Upload documentation
upload_document("api_docs.pdf")
upload_document("user_guide.pdf")

# Technical queries (keyword search important!)
data = {
    "query": "How to configure authentication?",
    "enable_hybrid_search": True,  # Keyword search dla tech terms
    "keyword_weight": 0.5,  # Zwiększ wagę keyword
    "top_k": 3
}
```

### Use Case 3: Legal Document Analysis
```python
# Upload legal documents
upload_document("contract.pdf")

# Precise queries (reranking important!)
data = {
    "query": "What are the termination clauses?",
    "enable_reranking": True,
    "reranking_top_k": 30,  # Więcej kandydatów
    "final_top_k": 5,
    "include_sources": True  # Ważne dla legal
}
```

## 📈 Przykład 7: Monitoring i Health Check

### Health Check:
```bash
curl http://localhost:8000/health
```

### Response:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "components": {
    "vector_store": {
      "status": "healthy",
      "total_chunks": 1250,
      "collections": ["text_chunks", "visual_chunks"]
    },
    "bm25_index": {
      "status": "healthy",
      "total_documents": 45,
      "index_size_mb": 12.5
    },
    "reranker": {
      "status": "healthy",
      "model": "ms-marco-MiniLM-L-6-v2",
      "device": "cuda:0"
    },
    "llm": {
      "status": "healthy",
      "provider": "openai",
      "model": "gpt-4o-mini"
    }
  },
  "enhancements": {
    "query_expansion": true,
    "hybrid_search": true,
    "reranking": true,
    "semantic_chunking": true
  }
}
```

## 🔍 Przykład 8: Debugging

### Sprawdź co się dzieje wewnątrz:
```python
data = {
    "query": "machine learning",
    "top_k": 5,
    "debug": True
}

response = query(data)

# Analiza timings
timings = response["metadata"]["timings"]
print(f"Query expansion: {timings['query_expansion']}s")
print(f"Vector search: {timings['vector_search']}s")
print(f"Keyword search: {timings['keyword_search']}s")
print(f"Reranking: {timings['reranking']}s")
print(f"LLM: {timings['llm_generation']}s")

# Analiza scores
scores = response["metadata"]["scores"]
print(f"\nVector contribution: {scores['vector_contribution']:.2%}")
print(f"Keyword contribution: {scores['keyword_contribution']:.2%}")
print(f"Reranking improvement: {scores['reranking_improvement']:.2%}")

# Query variations
variations = response["metadata"]["query_variations"]
print(f"\nQuery variations:")
for v in variations:
    print(f"  - {v}")
```

## 🎨 Przykład 9: Frontend Integration

### React Example:
```javascript
import React, { useState } from 'react';

function RAGChat() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleQuery = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          top_k: 5,
          include_sources: true
        })
      });
      const data = await res.json();
      setResponse(data);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  return (
    <div>
      <input 
        value={query} 
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask a question..."
      />
      <button onClick={handleQuery} disabled={loading}>
        {loading ? 'Searching...' : 'Ask'}
      </button>
      
      {response && (
        <div>
          <h3>Answer:</h3>
          <p>{response.answer}</p>
          
          <h4>Sources:</h4>
          {response.sources.map((source, i) => (
            <div key={i}>
              <p><strong>Score: {source.score.toFixed(2)}</strong></p>
              <p>{source.text}</p>
              <small>Page {source.metadata.page}</small>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

## 🔧 Przykład 10: Custom Configuration per Query

### Różne konfiguracje dla różnych typów pytań:

```python
def query_with_strategy(query, strategy="balanced"):
    strategies = {
        "fast": {
            "enable_query_expansion": False,
            "enable_hybrid_search": False,
            "enable_reranking": False,
            "top_k": 3
        },
        "balanced": {
            "enable_query_expansion": True,
            "enable_hybrid_search": True,
            "enable_reranking": True,
            "top_k": 5
        },
        "quality": {
            "enable_query_expansion": True,
            "expansion_method": "hyde",
            "enable_hybrid_search": True,
            "enable_reranking": True,
            "reranking_top_k": 30,
            "top_k": 5
        },
        "keyword_focused": {
            "enable_query_expansion": False,
            "enable_hybrid_search": True,
            "keyword_weight": 0.6,
            "vector_weight": 0.4,
            "enable_reranking": True,
            "top_k": 5
        }
    }
    
    config = strategies.get(strategy, strategies["balanced"])
    config["query"] = query
    
    return requests.post("http://localhost:8000/query", json=config)

# Użycie:
response = query_with_strategy("What is AI?", strategy="quality")
```

## 📊 Przykład 11: Performance Comparison

```python
import time

def benchmark_strategies(query):
    strategies = ["fast", "balanced", "quality"]
    results = {}
    
    for strategy in strategies:
        start = time.time()
        response = query_with_strategy(query, strategy)
        elapsed = time.time() - start
        
        results[strategy] = {
            "time": elapsed,
            "answer_length": len(response.json()["answer"]),
            "num_sources": len(response.json()["sources"])
        }
    
    return results

# Test:
results = benchmark_strategies("What are the main findings?")
for strategy, metrics in results.items():
    print(f"{strategy}: {metrics['time']:.2f}s, "
          f"{metrics['num_sources']} sources")
```

## 🎓 Best Practices

### 1. Chunking Strategy
```python
# Dla technical docs - fixed chunking
CHUNKING_STRATEGY=fixed
CHUNK_SIZE=512

# Dla narrative text - semantic chunking
CHUNKING_STRATEGY=semantic
USE_STRUCTURE_AWARE_CHUNKING=true

# Dla Q&A - sentence-window
CHUNKING_STRATEGY=sentence-window
SENTENCE_WINDOW_SIZE=3
```

### 2. Query Expansion
```python
# Dla broad questions - multi-query
EXPANSION_METHOD=multi-query
NUM_QUERY_VARIATIONS=3

# Dla specific questions - HyDE
EXPANSION_METHOD=hyde

# Dla exact matches - none
EXPANSION_METHOD=none
```

### 3. Hybrid Search Weights
```python
# Dla semantic search - więcej vector
VECTOR_WEIGHT=0.8
KEYWORD_WEIGHT=0.2

# Dla technical terms - więcej keyword
VECTOR_WEIGHT=0.5
KEYWORD_WEIGHT=0.5

# Dla exact matches - głównie keyword
VECTOR_WEIGHT=0.3
KEYWORD_WEIGHT=0.7
```

---

## 🚀 Gotowe do Użycia!

Wszystkie przykłady są gotowe do uruchomienia. Zacznij od:

1. Upload dokumentu
2. Proste query
3. Eksperymentuj z konfiguracją
4. Monitoruj performance
5. Dostosuj do swojego use case

**Happy RAG-ing!** 🎉
