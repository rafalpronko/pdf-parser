# RAG System - Wizualne Diagramy Przepływu

## 📤 Flow 1: Upload i Przetwarzanie Dokumentu

```mermaid
graph TD
    A[👤 User Upload PDF] --> B[📄 RAGAnythingParser]
    B --> C{Typ Dokumentu?}
    C -->|PDF| D[pypdf/MinerU]
    C -->|Image| E[OCR]
    C -->|Text| F[Direct Parse]
    
    D --> G[📝 Extracted Content]
    E --> G
    F --> G
    
    G --> H[✂️ SemanticChunker]
    H --> I[Detect Boundaries]
    I --> J[Create Chunks]
    J --> K[Add Metadata]
    
    K --> L[🧮 OpenAI Embedder]
    L --> M[Generate Embeddings]
    
    M --> N{Parallel Indexing}
    N --> O[💾 ChromaDB<br/>Vector Store]
    N --> P[🔍 BM25 Index<br/>Keyword Search]
    
    O --> Q[✅ Document Stored]
    P --> Q
    
    style A fill:#e1f5ff
    style Q fill:#c8e6c9
    style N fill:#fff9c4
```

## 🔍 Flow 2: Query Processing (Enhanced Retrieval)

```mermaid
graph TD
    A[👤 User Query:<br/>'machine learning'] --> B[🔄 Query Expansion]
    
    B --> C[Multi-Query Generator]
    C --> D1[Variant 1:<br/>'ML algorithms']
    C --> D2[Variant 2:<br/>'supervised learning']
    C --> D3[Original Query]
    
    D1 --> E[🔀 Hybrid Search]
    D2 --> E
    D3 --> E
    
    E --> F{Parallel Search}
    
    F --> G[🎯 Vector Search<br/>ChromaDB]
    F --> H[📝 Keyword Search<br/>BM25]
    
    G --> I[Top-20 Semantic<br/>Matches]
    H --> J[Top-20 Keyword<br/>Matches]
    
    I --> K[⚖️ RRF Fusion<br/>0.7 vector + 0.3 keyword]
    J --> K
    
    K --> L[Top-20 Combined<br/>Candidates]
    
    L --> M[🎖️ Cross-Encoder<br/>Reranking]
    M --> N[Score Each Pair<br/>query + chunk]
    N --> O[Top-5 Best<br/>Chunks]
    
    O --> P[🤖 LLM Generation<br/>GPT-4o-mini]
    P --> Q[📊 Final Answer<br/>+ Sources + Metadata]
    
    style A fill:#e1f5ff
    style E fill:#fff9c4
    style K fill:#ffe0b2
    style M fill:#f8bbd0
    style P fill:#c5cae9
    style Q fill:#c8e6c9
```

## 🎯 Szczegółowy Flow: Hybrid Search + Reranking

```mermaid
sequenceDiagram
    participant U as User
    participant QE as Query Expander
    participant VS as Vector Search
    participant KS as Keyword Search
    participant RRF as RRF Fusion
    participant RE as Reranker
    participant LLM as GPT-4o-mini
    
    U->>QE: "machine learning"
    QE->>QE: Generate 3 variants
    QE-->>U: Variants cached
    
    par Parallel Search
        QE->>VS: Search variant 1
        VS-->>RRF: 20 chunks (semantic)
        QE->>KS: Search variant 1
        KS-->>RRF: 20 chunks (keyword)
    end
    
    RRF->>RRF: Combine scores<br/>0.7*vector + 0.3*keyword
    RRF-->>RE: 20 candidates
    
    RE->>RE: Cross-encoder scoring<br/>batch_size=32
    RE-->>LLM: Top-5 chunks
    
    LLM->>LLM: Generate answer<br/>with context
    LLM-->>U: Answer + Sources
    
    Note over U,LLM: Total time: ~1.5s
```

## 🏗️ Architektura Komponentów

```mermaid
graph LR
    subgraph "Frontend"
        A[User Interface]
    end
    
    subgraph "API Layer"
        B[FastAPI Endpoints]
        B1[/documents/upload]
        B2[/query]
        B3[/health]
    end
    
    subgraph "Services"
        C[DocumentService]
        D[QueryService]
    end
    
    subgraph "Processing"
        E[RAGAnythingParser]
        F[SemanticChunker]
        G[OpenAI Embedder]
    end
    
    subgraph "Retrieval"
        H[QueryExpander]
        I[HybridSearchEngine]
        J[CrossEncoderReranker]
    end
    
    subgraph "Storage"
        K[(ChromaDB<br/>Vector Store)]
        L[(BM25 Index<br/>Keyword)]
    end
    
    A --> B
    B --> B1
    B --> B2
    B --> B3
    
    B1 --> C
    B2 --> D
    
    C --> E
    C --> F
    C --> G
    
    E --> K
    F --> K
    G --> K
    E --> L
    
    D --> H
    D --> I
    D --> J
    
    I --> K
    I --> L
    
    style A fill:#e1f5ff
    style K fill:#c8e6c9
    style L fill:#c8e6c9
    style I fill:#fff9c4
    style J fill:#f8bbd0
```

## 📊 Data Flow: Od Dokumentu do Odpowiedzi

```mermaid
flowchart TB
    subgraph "1. Document Ingestion"
        A[PDF Document] --> B[Parse & Extract]
        B --> C[Text Chunks]
        B --> D[Images]
        B --> E[Tables]
    end
    
    subgraph "2. Embedding & Indexing"
        C --> F[Text Embeddings<br/>1536 dim]
        D --> G[Image Embeddings<br/>CLIP]
        E --> H[Table Embeddings]
        
        F --> I[Vector DB]
        G --> I
        H --> I
        
        C --> J[BM25 Index]
    end
    
    subgraph "3. Query Processing"
        K[User Query] --> L[Query Expansion]
        L --> M[3 Variants]
    end
    
    subgraph "4. Retrieval"
        M --> N[Hybrid Search]
        N --> I
        N --> J
        
        I --> O[Semantic Results]
        J --> P[Keyword Results]
        
        O --> Q[RRF Fusion]
        P --> Q
        
        Q --> R[20 Candidates]
    end
    
    subgraph "5. Reranking"
        R --> S[Cross-Encoder]
        S --> T[Top-5 Chunks]
    end
    
    subgraph "6. Generation"
        T --> U[LLM Context]
        K --> U
        U --> V[GPT-4o-mini]
        V --> W[Final Answer]
    end
    
    style A fill:#e1f5ff
    style K fill:#e1f5ff
    style N fill:#fff9c4
    style S fill:#f8bbd0
    style V fill:#c5cae9
    style W fill:#c8e6c9
```

## 🎛️ Configuration Impact

```mermaid
graph TD
    A[Configuration] --> B{ENABLE_QUERY_EXPANSION}
    B -->|true| C[Multi-Query/HyDE]
    B -->|false| D[Single Query]
    
    C --> E{ENABLE_HYBRID_SEARCH}
    D --> E
    
    E -->|true| F[Vector + Keyword<br/>RRF Fusion]
    E -->|false| G[Vector Only]
    
    F --> H{ENABLE_RERANKING}
    G --> H
    
    H -->|true| I[Cross-Encoder<br/>Reranking]
    H -->|false| J[Direct Results]
    
    I --> K[Top-5 Results]
    J --> K
    
    K --> L[LLM Generation]
    
    style A fill:#e1f5ff
    style F fill:#fff9c4
    style I fill:#f8bbd0
    style L fill:#c5cae9
```

## 🔄 Caching Strategy

```mermaid
graph LR
    A[Query] --> B{Cache Check}
    
    B -->|Hit| C[Return Cached<br/>Expansion]
    B -->|Miss| D[Generate<br/>Expansion]
    
    D --> E[Store in Cache<br/>TTL: 1h]
    E --> F[Continue Pipeline]
    C --> F
    
    F --> G[Hybrid Search]
    G --> H{Reranking Cache}
    
    H -->|Hit| I[Cached Scores]
    H -->|Miss| J[Compute Scores]
    
    J --> K[Store Scores]
    K --> L[Return Results]
    I --> L
    
    style B fill:#fff9c4
    style H fill:#fff9c4
    style C fill:#c8e6c9
    style I fill:#c8e6c9
```

## 📈 Performance Optimization

```mermaid
graph TD
    A[Query Request] --> B[Query Expansion<br/>~80ms]
    
    B --> C{Parallel Execution}
    
    C --> D[Vector Search<br/>~50ms]
    C --> E[Keyword Search<br/>~20ms]
    
    D --> F[RRF Fusion<br/>~10ms]
    E --> F
    
    F --> G[Reranking<br/>~150ms]
    
    G --> H{GPU Available?}
    H -->|Yes| I[GPU Batch<br/>2x faster]
    H -->|No| J[CPU Processing]
    
    I --> K[LLM Generation<br/>~1200ms]
    J --> K
    
    K --> L[Total: ~1500ms]
    
    style C fill:#fff9c4
    style H fill:#ffe0b2
    style I fill:#c8e6c9
    style L fill:#c8e6c9
```

## 🎯 Scoring Mechanism

```mermaid
graph TD
    A[Chunk] --> B{Scoring Stages}
    
    B --> C[Stage 1: Vector<br/>Cosine Similarity<br/>0.0 - 1.0]
    B --> D[Stage 2: Keyword<br/>BM25 Score<br/>0.0 - 20.0]
    
    C --> E[Normalize Vector<br/>score_v]
    D --> F[Normalize BM25<br/>score_k]
    
    E --> G[RRF Score<br/>0.7*score_v + 0.3*score_k]
    F --> G
    
    G --> H[Stage 3: Reranking<br/>Cross-Encoder<br/>-10.0 - 10.0]
    
    H --> I[Normalize to [0,1]<br/>final_score]
    
    I --> J{final_score > threshold?}
    J -->|Yes| K[Include in Results]
    J -->|No| L[Discard]
    
    style G fill:#fff9c4
    style H fill:#f8bbd0
    style K fill:#c8e6c9
```

---

## 📝 Legenda

- 🔵 **Niebieski**: User input/output
- 🟡 **Żółty**: Processing/Computation
- 🟠 **Pomarańczowy**: Fusion/Combination
- 🔴 **Różowy**: Reranking/Scoring
- 🟣 **Fioletowy**: LLM Generation
- 🟢 **Zielony**: Final result/Success

---

## 🎓 Kluczowe Punkty

1. **Parallel Processing**: Vector + Keyword search równolegle
2. **Multi-Stage Scoring**: RRF → Reranking → Final
3. **Caching**: Query expansion i reranking scores
4. **GPU Acceleration**: Dla reranking (2x szybciej)
5. **Configurable**: Każdy stage można włączyć/wyłączyć
