# RAG Chat UI

Prosty interfejs czatu do testowania systemu RAG.

## Instalacja

```bash
cd frontend
npm install
```

## Uruchomienie

1. Upewnij się, że backend działa na porcie 8000:
```bash
# W głównym katalogu projektu
./start_server.sh
```

2. Uruchom frontend:
```bash
cd frontend
npm start
```

Aplikacja otworzy się automatycznie w przeglądarce na `http://localhost:3000`

## Funkcje

- 💬 Chat z systemem RAG
- 📚 Lista dostępnych dokumentów
- 📎 Wyświetlanie źródeł odpowiedzi
- ⏱️ Czas przetwarzania zapytań
- 🎨 Responsywny interfejs

## API Endpoints

Frontend komunikuje się z następującymi endpointami:

- `GET /api/documents` - Lista dokumentów
- `POST /api/query` - Zapytanie do RAG

## Konfiguracja

Backend URL jest skonfigurowany w `package.json` jako proxy:
```json
"proxy": "http://localhost:8000"
```
