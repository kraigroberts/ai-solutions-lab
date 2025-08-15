"""
Configuration management for AI Solutions Lab.

Uses Pydantic settings for type-safe environment variable handling
with sensible defaults for local-first operation.
"""

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # LLM Backend Configuration
    model_backend: Literal["local", "openai", "anthropic"] = Field(
        default="local", description="LLM provider to use for chat and generation"
    )

    # Embeddings Backend Configuration
    embeddings_backend: Literal["local", "openai"] = Field(
        default="local", description="Embeddings provider for vector operations"
    )

    # Vector Store Configuration
    vector_store: Literal["faiss", "pinecone"] = Field(
        default="faiss", description="Vector database backend"
    )

    # API Keys (optional for local operation)
    openai_api_key: Optional[str] = Field(
        default=None, description="OpenAI API key for GPT models and embeddings"
    )

    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key for Claude models"
    )

    pinecone_api_key: Optional[str] = Field(
        default=None, description="Pinecone API key for cloud vector store"
    )

    pinecone_environment: Optional[str] = Field(
        default=None, description="Pinecone environment (e.g., us-west1-gcp)"
    )

    pinecone_index_name: Optional[str] = Field(
        default="ai-solutions-lab", description="Pinecone index name"
    )

    # Local Model Configuration
    local_model_path: Optional[str] = Field(
        default="./models/tinyllama.gguf", description="Path to local GGUF model file"
    )

    local_model_context_size: int = Field(
        default=2048, description="Context window size for local models"
    )

    local_model_max_tokens: int = Field(
        default=512, description="Maximum tokens to generate with local models"
    )

    # Embeddings Configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for local embeddings",
    )

    embedding_dimension: int = Field(
        default=384, description="Dimension of embedding vectors"
    )

    # RAG Configuration
    chunk_size: int = Field(
        default=1000, description="Document chunk size for processing"
    )

    chunk_overlap: int = Field(
        default=200, description="Overlap between document chunks"
    )

    top_k: int = Field(
        default=5, description="Number of top chunks to retrieve for RAG"
    )

    # Data Paths
    data_dir: Path = Field(
        default=Path("./data"), description="Base directory for data storage"
    )

    docs_dir: Path = Field(
        default=Path("./data/docs"), description="Directory containing source documents"
    )

    index_dir: Path = Field(
        default=Path("./data/index"), description="Directory for FAISS index storage"
    )

    models_dir: Path = Field(
        default=Path("./models"), description="Directory for local model files"
    )

    # Server Configuration
    host: str = Field(
        default="0.0.0.0", description="Host to bind the FastAPI server to"
    )

    port: int = Field(default=8000, description="Port to bind the FastAPI server to")

    debug: bool = Field(default=False, description="Enable debug mode")

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")

    # Rate Limiting
    rate_limit_per_minute: int = Field(
        default=60, description="Rate limit for API endpoints per minute"
    )

    @field_validator("data_dir", "docs_dir", "index_dir", "models_dir")
    @classmethod
    def ensure_directories_exist(cls, v: Path) -> Path:
        """Ensure data directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("local_model_path")
    @classmethod
    def validate_local_model_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate local model path if specified."""
        if v and not Path(v).exists():
            # Don't fail validation, just warn
            print(f"Warning: Local model path {v} does not exist")
        return v

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: Optional[str], info) -> Optional[str]:
        """Validate OpenAI key if using OpenAI backend."""
        if info.data.get("model_backend") == "openai" and not v:
            print("Warning: OpenAI backend selected but no API key provided")
        return v

    @field_validator("anthropic_api_key")
    @classmethod
    def validate_anthropic_key(cls, v: Optional[str], info) -> Optional[str]:
        """Validate Anthropic key if using Anthropic backend."""
        if info.data.get("model_backend") == "anthropic" and not v:
            print("Warning: Anthropic backend selected but no API key provided")
        return v

    @field_validator("pinecone_api_key")
    @classmethod
    def validate_pinecone_key(cls, v: Optional[str], info) -> Optional[str]:
        """Validate Pinecone key if using Pinecone backend."""
        if info.data.get("vector_store") == "pinecone" and not v:
            print("Warning: Pinecone backend selected but no API key provided")
        return v

    def is_local_mode(self) -> bool:
        """Check if running in local-only mode."""
        return (
            self.model_backend == "local"
            and self.embeddings_backend == "local"
            and self.vector_store == "faiss"
        )

    def has_openai(self) -> bool:
        """Check if OpenAI is available."""
        return bool(self.openai_api_key)

    def has_anthropic(self) -> bool:
        """Check if Anthropic is available."""
        return bool(self.anthropic_api_key)

    def has_pinecone(self) -> bool:
        """Check if Pinecone is available."""
        return bool(self.pinecone_api_key and self.pinecone_environment)


# Global settings instance
settings = Settings()


# Convenience functions
def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> None:
    """Reload settings from environment."""
    global settings
    settings = Settings()
