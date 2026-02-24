# 🚀 RAG-Anything Multimodal System - Quick Start

## Uruchomienie serwera

### Opcja 1: Skrypt startowy
```bash
./start_server.sh
```

### Opcja 2: Bezpośrednio
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📍 Dostępne endpointy

Po uruchomieniu serwer będzie dostępny na `http://localhost:8000`

### Dokumentacja API
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Health Check
```bash
curl http://localhost:8000/health
```

Odpowiedź pokaże multimodal capabilities:
```json
{
  "status": "healthy",
  "service": "RAG-Anything Multimodal System",
  "version": "2.0.0",
  "capabilities": {
    "multimodal": true,
    "vlm_enabled": true,
    "vlm_provider": "openai",
    "vision_encoder": "clip-vit-large-patch14",
    "rag_anything": true
  }
}
```

## 📤 Upload dokumentu

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@data/raw/your_document.pdf" \
  -F "metadata={\"tags\":[\"test\"],\"description\":\"Test document\"}"
```

## 🔍 Query dokumentów

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Co jest przedmiotem ubezpieczenia?",
    "top_k": 5,
    "include_visual": true,
    "modality_filter": null
  }'
```

## 📋 Lista dokumentów

```bash
curl http://localhost:8000/api/v1/documents
```

## 🎨 Multimodal Features

System obsługuje:
- ✅ **Text extraction** - PyMuPDF/MinerU
- ✅ **Image extraction** - z metadanymi i pozycją
- ✅ **Chart detection** - automatyczne wykrywanie wykresów
- ✅ **Table extraction** - strukturalne dane tabelaryczne
- ✅ **VLM (Vision-Language Model)** - GPT-4V dla zrozumienia obrazów
- ✅ **Multimodal chunking** - łączenie tekstu z wizualizacjami
- ✅ **Multimodal embeddings** - OpenAI (text) + CLIP (vision)
- ✅ **Cross-modal retrieval** - wyszukiwanie między modalnościami

## 🧪 Testy

### Uruchom wszystkie testy
```bash
uv run pytest tests/ -v
```

### Tylko property tests
```bash
uv run pytest tests/test_*_properties.py -v
```

### Demo multimodal
```bash
uv run python scripts/demo_multimodal_rag.py
```

## 📊 Status testów

Aktualnie: **22/22 testy przechodzą** ✅
- Config tests: 8/8
- Model tests: 8/8
- Parser tests: 6/6

## 🔧 Konfiguracja

Edytuj `.env` aby skonfigurować:
- `OPENAI_API_KEY` - klucz API OpenAI
- `VLM_PROVIDER` - provider VLM (openai, local)
- `VLM_MODEL` - model VLM (gpt-4-vision-preview)
- `VISION_ENCODER` - encoder wizji (clip-vit-large-patch14)
- `ENABLE_VLM` - włącz/wyłącz VLM (true/false)
- `ENABLE_MULTIMODAL_CHUNKING` - włącz multimodal chunking (true/false)

## 📚 Więcej informacji

Zobacz pełną dokumentację w:
- `README.md` - główna dokumentacja
- `.kiro/specs/pdf-rag-system/` - specyfikacja systemu
- `app/` - kod źródłowy z komentarzami
