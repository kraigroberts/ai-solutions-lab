# AI Solutions Lab

A production-minded lab demonstrating AI Solutions Engineering skills: clean system design, LLM integrations, RAG pipelines, lightweight agent patterns, tests + CI, containers, and comprehensive documentation.

## 🎯 Mission

Build a small but production-ready lab showcasing:
- **LLM-powered features**: Chat, retrieval-augmented answering, simple "agent" tool-calls
- **Clean service boundary**: FastAPI + CLI utilities
- **Local-first vector store**: FAISS with optional Pinecone toggle
- **Solid repo hygiene**: Tests, CI, typed code, linting, packaging, Docker
- **Recruiter-friendly**: Copy-paste commands and architecture diagram

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   CLI Tools     │    │   Data Layer    │
│                 │    │                 │    │                 │
│  /chat          │    │  chat           │    │  ./data/docs    │
│  /rag/answer    │    │  rag            │    │  ./data/index   │
│  /agent/run     │    │  agent          │    │                 │
│  /health        │    │  ingest         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Core Modules  │
                    │                 │
                    │  LLM Router     │
                    │  RAG Pipeline   │
                    │  Tool Registry  │
                    │  Config Mgmt    │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Backends      │
                    │                 │
                    │  Local (llama)  │
                    │  OpenAI         │
                    │  Anthropic      │
                    │  FAISS/Pinecone │
                    └─────────────────┘
```

## 🚀 Quickstart (Local-First)

### Prerequisites
- Python 3.10+
- 4GB RAM minimum
- Git

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/ai-solutions-lab.git
cd ai-solutions-lab
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### 2. Download Local Model (Optional)
```bash
# Download a small GGUF model for local inference
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O ./models/tinyllama.gguf
```

### 3. Ingest Sample Documents
```bash
# Build FAISS index from sample docs
python -m ai_lab.cli ingest build --src ./data/docs --out ./data/index
```

### 4. Start the API
```bash
# Start FastAPI server
uvicorn src.ai_lab.api:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test RAG
```bash
# Test RAG endpoint
curl -X POST "http://localhost:8000/rag/answer" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'

# Or use CLI
python -m ai_lab.cli rag --query "What is machine learning?"
```

### 6. Test Agent
```bash
# Test agent endpoint
curl -X POST "http://localhost:8000/agent/run" \
  -H "Content-Type: application/json" \
  -d '{"goal": "Calculate 15 * 23 and search for information about Python"}'

# Or use CLI
python -m ai_lab.cli agent --goal "Calculate 15 * 23 and search for information about Python"
```

## 🔧 Usage Examples

### RAG Pipeline
```bash
# Build index from custom docs
python -m ai_lab.cli ingest build --src /path/to/your/docs --out ./data/custom_index

# Query with custom index
python -m ai_lab.cli rag --query "Your question here" --index ./data/custom_index
```

### Chat Interface
```bash
# Local chat (if model downloaded)
python -m ai_lab.cli chat --message "Hello, how are you?"

# API endpoint
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

### Agent Tools
```bash
# Run agent with specific goal
python -m ai_lab.cli agent --goal "Search for information about FastAPI and then calculate 2^10"
```

## ☁️ Cloud Integration (Optional)

### Environment Variables
Create `.env` file from `.env.example`:
```bash
cp .env.example .env
```

Configure your preferred backends:
```bash
# LLM Backend (local|openai|anthropic)
MODEL_BACKEND=openai

# Embeddings Backend (local|openai)
EMBEDDINGS_BACKEND=openai

# API Keys (only if using cloud services)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

### Pinecone Vector Store
```bash
# Set environment
export PINECONE_API_KEY=your_key
export VECTOR_STORE=pinecone

# Rebuild index
python -m ai_lab.cli ingest build --src ./data/docs --out pinecone
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Lint code
ruff check src/

# Type checking
mypy src/
```

## 🐳 Docker

### Build & Run
```bash
# Build image
docker build -t ai-solutions-lab .

# Run container
docker run -p 8000:8000 ai-solutions-lab

# Or use docker-compose
docker-compose up
```

## 📁 Project Structure

```
ai-solutions-lab/
├── src/ai_lab/
│   ├── __init__.py
│   ├── api.py              # FastAPI application
│   ├── cli.py              # CLI interface
│   ├── config.py           # Configuration management
│   ├── llm/                # LLM provider modules
│   │   ├── router.py       # LLM routing logic
│   │   ├── openai.py       # OpenAI integration
│   │   ├── anthropic.py    # Anthropic integration
│   │   └── llama_cpp.py    # Local llama.cpp integration
│   ├── rag/                # RAG pipeline
│   │   ├── ingest.py       # Document ingestion
│   │   ├── retrieve.py     # Vector retrieval
│   │   └── answer.py       # Answer generation
│   └── tools/              # Agent tools
│       ├── search.py       # Search tool
│       ├── math.py         # Math evaluation
│       └── webstub.py      # Web utilities
├── data/
│   ├── docs/               # Sample documents
│   └── index/              # FAISS index storage
├── tests/                  # Test suite
├── .github/workflows/      # CI/CD
├── Dockerfile              # Container definition
├── pyproject.toml          # Project configuration
└── requirements.txt         # Dependencies
```

## 🚀 Development

### Local Development
```bash
# Install in development mode
pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit install

# Format code
ruff format src/

# Check types
mypy src/
```

### Adding New Tools
1. Create tool in `src/ai_lab/tools/`
2. Register in tool registry
3. Add tests in `tests/test_tools.py`
4. Update documentation

## 📊 Performance

- **Local Model**: ~2-4 seconds per response (depending on model size)
- **RAG Query**: ~1-2 seconds (FAISS), ~2-3 seconds (Pinecone)
- **Agent Execution**: ~3-5 seconds (varies by tool complexity)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Ensure CI passes
6. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🎯 Target Role Alignment

This project demonstrates skills relevant to **AI Solutions Engineer** positions:
- **LLM Integration**: Multiple provider support with fallback strategies
- **RAG Pipelines**: Document ingestion, vector search, context-aware answering
- **Agent Patterns**: Tool calling, orchestration, audit trails
- **Backend Integration**: FastAPI, async patterns, clean service boundaries
- **DevOps**: CI/CD, Docker, testing, linting, type safety
- **Documentation**: Clear setup, usage examples, architecture diagrams

Perfect for showcasing to recruiters and technical interviewers!
