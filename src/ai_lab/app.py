"""Simple FastAPI app."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

from .simple_rag import SimpleRAG

app = FastAPI(title="AI Solutions Lab", version="0.0.1")

# Global RAG instance
rag = SimpleRAG()

class QueryRequest(BaseModel):
    query: str
    k: int = 3

class DocumentRequest(BaseModel):
    documents: List[str]

@app.get("/")
def read_root():
    return {"message": "AI Solutions Lab - Simple RAG API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/documents")
def add_documents(request: DocumentRequest):
    """Add documents to the RAG index."""
    try:
        rag.add_documents(request.documents)
        return {"message": f"Added {len(request.documents)} documents", "count": len(request.documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_documents(request: QueryRequest):
    """Search for similar documents."""
    try:
        results = rag.search(request.query, request.k)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
