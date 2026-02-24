#!/bin/bash
# Start RAG-Anything Multimodal API Server

echo "🚀 Starting RAG-Anything Multimodal API Server..."
echo "📍 Server will be available at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔍 Health check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
