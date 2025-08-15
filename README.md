# AI Solutions Lab - Simple & Working

A simple, working AI lab that demonstrates RAG (Retrieval-Augmented Generation) with vector search.

## What This Actually Does

- **RAG System**: Simple document search using embeddings and FAISS
- **FastAPI API**: REST endpoints for adding documents and searching
- **Working Demo**: Actually runs and gives results

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the RAG System
```bash
python src/ai_lab/simple_rag.py
```

### 3. Run the API
```bash
python src/ai_lab/app.py
```

### 4. Use the API
```bash
# Add documents
curl -X POST "http://localhost:8000/documents" \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Machine learning is AI", "Python is great", "FastAPI is modern"]}'

# Search
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "k": 3}'
```

## Run Tests
```bash
python -m pytest tests/ -v
```

## What We Built

- **Simple RAG**: `SimpleRAG` class that actually works
- **FastAPI App**: Basic API with `/documents` and `/search` endpoints
- **Working Tests**: Tests that pass and verify functionality

## Next Steps

This foundation works. Now we can add:
- Document ingestion from files
- Better chunking
- LLM integration for answers
- More sophisticated search

But first, **this actually works** and demonstrates the core concept.
