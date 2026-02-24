#!/bin/bash

echo "🚀 Starting RAG Chat Frontend..."
echo ""
echo "Make sure the backend is running on http://localhost:8000"
echo ""

cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

echo "🎨 Starting development server..."
npm start
