# Analiza RAG Pipeline - Co Się Naprawdę Dzieje

## Pipeline Flow dla Query: "Co jest przedmiotem ubezpieczenia?"

### KROK 1: Query Expansion (Hybrid)
**Kod**: `app/services/query_service.py:160`

```python
queries = await self.query_expander.expand(request.question)
```

**Wynik**: 5 wygenerowanych wariantów:
1. **HyDE**: "Przedmiotem ubezpieczenia jest pojazd silnikowy wraz z wyposażeniem..."
2. **Multi-1**: "Co jest przedmiotem ubezpieczenia?"
3. **Multi-2**: "Jaki pojazd podlega ubezpieczeniu?"
4. **Multi-3**: "Co obejmuje zakres ubezpieczenia?"
5. **Multi-4**: "Jakie elementy są ubezpieczone?"

---

### KROK 2: Embeddingi dla Wszystkich Wariantów
**Kod**: `app/services/query_service.py:166-174`

```python
for i, query in enumerate(queries, 1):
    embedding = await self._embed_query(query)
    query_embeddings.append(embedding)
```

**Wynik**: 5 embeddingów (jeden dla każdego wariantu)

---

### KROK 3: Vector Search dla KAŻDEGO Wariantu
**Kod**: `app/services/query_service.py:186-193`

```python
for i, (query, embedding) in enumerate(zip(queries, query_embeddings), 1):
    results = await self.hybrid_search.search(
        query=query,
        query_embedding=embedding,
        top_k=40,  # reranking_top_k
    )
    all_results.extend(results)
```

**Wynik**:
- Search 1 (HyDE): 40 chunków
- Search 2 (Multi-1): 40 chunków
- Search 3 (Multi-2): 40 chunków
- Search 4 (Multi-3): 40 chunków
- Search 5 (Multi-4): 40 chunków
- **RAZEM**: 200 chunków (z duplikatami)

**Ważne**: Chunk #7 "Pojazd (silnikowy...)" może być:
- Rank #7 dla wariantu HyDE
- Rank #12 dla Multi-1
- Rank #5 dla Multi-2
- itd.

---

### KROK 4: Deduplication
**Kod**: `app/services/query_service.py:207-213`

```python
seen_chunks = set()
unique_results = []
for result in all_results:
    if result.chunk_id not in seen_chunks:
        seen_chunks.add(result.chunk_id)
        unique_results.append(result)
```

**Wynik**: ~50-80 unikalnych chunków (zależy od overlap)

**Chunk #7** jest teraz w tej liście unikalnych chunków!

---

### KROK 5: Reranking - TUTAJ JEST PROBLEM! ⚠️
**Kod**: `app/services/query_service.py:242-245`

```python
reranked = self.reranker.rerank(
    query=request.question,  # ← TYLKO ORYGINALNE PYTANIE!
    chunks=reranker_results,  # ← WSZYSTKIE ~50-80 CHUNKÓW
    top_k=10,
)
```

**PROBLEM**:
- Reranking używa **TYLKO oryginalnego pytania** `"Co jest przedmiotem ubezpieczenia?"`
- **NIE UŻYWA** wygenerowanych wariantów HyDE/Multi-Query!
- Cross-encoder porównuje wszystkie ~50-80 chunków z oryginalnym pytaniem
- Dla pytania "Co jest przedmiotem ubezpieczenia?" cross-encoder preferuje:
  - ✓ "Suma ubezpieczenia odpowiada..." (brzmi jak definicja przedmiotu)
  - ✗ "Pojazd (silnikowy...)" (brzmi jak lista szczegółów)

---

## Dlaczego To Nie Działa?

### 1. Vector Search znajdzie chunk #7:
- HyDE hypothetical doc: "Przedmiotem jest pojazd..."
- Embedding podobny do chunk #7 → rank #7 ✓

### 2. Deduplication zachowuje chunk #7:
- Chunk #7 jest w unique_results ✓

### 3. Reranking USUWA chunk #7:
- Cross-encoder MS-MARCO został wytrenowany na Q&A z internetu
- Dla pytania "Co jest przedmiotem ubezpieczenia?":
  - "Suma ubezpieczenia odpowiada..." → score 1.0000 (wygląda jak odpowiedź)
  - "Pojazd (silnikowy...)" → score < 0.5 (wygląda jak lista/wyliczenie)
- Chunk #7 wypada z top-10! ✗

---

## Rozwiązania

### Opcja A: Rerank dla każdego query wariantu osobno
```python
# Dla każdego wariantu z expansion:
for query in queries:
    reranked = reranker.rerank(query=query, chunks=chunks)
    all_reranked.extend(reranked)

# Potem deduplicate i RRF fusion
```

### Opcja B: Wyłączyć reranking (działa!)
```python
ENABLE_RERANKING=false
```

### Opcja C: Użyć lepszego reranker model
```python
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```
