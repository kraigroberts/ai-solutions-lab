"""Configuration management for AI Solutions Lab."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")

    # Data paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    index_dir: Optional[Path] = Field(default=None, description="Index directory")

    # Mock LLM settings
    mock_llm_enabled: bool = Field(
        default=True, description="Enable mock LLM responses"
    )
    mock_response_delay: float = Field(
        default=0.1, description="Mock response delay in seconds"
    )

    # RAG settings
    top_k: int = Field(default=5, description="Number of top results to return")
    chunk_size: int = Field(default=1000, description="Document chunk size")
    chunk_overlap: int = Field(default=200, description="Chunk overlap size")

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


# Global settings instance
settings = get_settings()
