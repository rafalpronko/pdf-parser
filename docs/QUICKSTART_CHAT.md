# 🚀 Quick Start - RAG Chat UI

Szybki przewodnik uruchomienia interfejsu czatu z RAG.

## Krok 1: Uruchom Backend

```bash
# Upewnij się, że masz skonfigurowany .env z OPENAI_API_KEY
./start_server.sh
```

Backend będzie dostępny na `http://localhost:8000`

## Krok 2: Wgraj Dokumenty

Możesz wgrać dokumenty przez API lub skrypt:

```bash
# Przez skrypt
uv run python scripts/upload_pdfs.py data/raw/

# Lub przez curl
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@twoj_dokument.pdf"
```

## Krok 3: Uruchom Frontend

```bash
./start_frontend.sh
```

Frontend otworzy się automatycznie w przeglądarce na `http://localhost:3000`

## Krok 4: Testuj!

1. Po lewej stronie zobaczysz listę wgranych dokumentów
2. W polu tekstowym na dole wpisz pytanie
3. Kliknij 📤 lub naciśnij Enter
4. Odpowiedź pojawi się z cytatami ze źródeł

## Przykładowe Pytania

- "Co to jest RAG?"
- "Jakie są główne funkcje systemu?"
- "Podsumuj kluczowe punkty z dokumentu"
- "Wyjaśnij architekturę systemu"

## Funkcje UI

### Lista Dokumentów (Lewa Strona)
- 📄 Nazwa pliku
- 📊 Liczba stron i chunków
- Automatyczne odświeżanie

### Chat (Środek)
- 💬 Historia konwersacji
- 📎 Źródła dla każdej odpowiedzi
- ⏱️ Czas przetwarzania
- 🎨 Czytelny interfejs

### Źródła
Dla każdej odpowiedzi zobaczysz:
- Nazwę pliku źródłowego
- Numer strony
- Wynik relevancji (%)
- Fragment tekstu

## Rozwiązywanie Problemów

### Backend nie odpowiada
```bash
# Sprawdź czy backend działa
curl http://localhost:8000/health
```

### Brak dokumentów
```bash
# Sprawdź listę dokumentów
curl http://localhost:8000/api/documents
```

### Frontend nie łączy się z backendem
- Upewnij się, że backend działa na porcie 8000
- Sprawdź konsolę przeglądarki (F12) dla błędów

### Błędy CORS
Backend ma już skonfigurowany CORS, ale jeśli masz problemy:
- Sprawdź czy używasz `http://localhost:3000` (nie `127.0.0.1`)
- Zrestartuj oba serwery

## Wskazówki

1. **Pierwsze uruchomienie**: Instalacja zależności frontendu może potrwać kilka minut
2. **Hot reload**: Zmiany w kodzie React automatycznie odświeżają przeglądarkę
3. **Debugowanie**: Otwórz DevTools (F12) aby zobaczyć requesty API
4. **Wydajność**: Pierwsze zapytanie może być wolniejsze (cold start)

## Następne Kroki

- Eksperymentuj z różnymi pytaniami
- Wgraj własne dokumenty PDF
- Sprawdź parametry zapytań (top_k, temperature)
- Zobacz źródła w odpowiedziach

Miłego testowania! 🎉
