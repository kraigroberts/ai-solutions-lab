"""LLM configuration and provider management."""

import os
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    
    # Provider selection
    provider: Literal["openai", "anthropic", "local", "none"] = "none"
    
    # OpenAI configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 1000
    openai_temperature: float = 0.7
    
    # Anthropic configuration
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-haiku-20240307"
    anthropic_max_tokens: int = 1000
    anthropic_temperature: float = 0.7
    
    # Local model configuration
    local_model_path: Optional[str] = None
    local_model_type: Literal["llama", "mistral", "other"] = "llama"
    local_max_tokens: int = 1000
    local_temperature: float = 0.7
    
    # Common settings
    timeout: int = 30
    retry_attempts: int = 3
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.local_model_path = self.local_model_path or os.getenv("LOCAL_MODEL_PATH")
    
    def is_configured(self) -> bool:
        """Check if the selected provider is properly configured."""
        if self.provider == "openai":
            return bool(self.openai_api_key)
        elif self.provider == "anthropic":
            return bool(self.anthropic_api_key)
        elif self.provider == "local":
            return bool(self.local_model_path and Path(self.local_model_path).exists())
        elif self.provider == "none":
            return True
        return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the configured provider."""
        if self.provider == "openai":
            return {
                "provider": "openai",
                "model": self.openai_model,
                "configured": self.is_configured(),
                "api_key_present": bool(self.openai_api_key)
            }
        elif self.provider == "anthropic":
            return {
                "provider": "anthropic",
                "model": self.anthropic_model,
                "configured": self.is_configured(),
                "api_key_present": bool(self.anthropic_api_key)
            }
        elif self.provider == "local":
            return {
                "provider": "local",
                "model_path": self.local_model_path,
                "model_type": self.local_model_type,
                "configured": self.is_configured(),
                "model_exists": bool(self.local_model_path and Path(self.local_model_path).exists())
            }
        else:
            return {
                "provider": "none",
                "configured": True,
                "message": "No LLM provider configured"
            }
    
    def save_to_file(self, file_path: str = "llm_config.json") -> None:
        """Save configuration to a JSON file."""
        config_data = {
            "provider": self.provider,
            "openai_model": self.openai_model,
            "openai_max_tokens": self.openai_max_tokens,
            "openai_temperature": self.openai_temperature,
            "anthropic_model": self.anthropic_model,
            "anthropic_max_tokens": self.anthropic_max_tokens,
            "anthropic_temperature": self.anthropic_temperature,
            "local_model_path": self.local_model_path,
            "local_model_type": self.local_model_type,
            "local_max_tokens": self.local_max_tokens,
            "local_temperature": self.local_temperature,
            "timeout": self.timeout,
            "retry_attempts": self.retry_attempts
        }
        
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Configuration saved to {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: str = "llm_config.json") -> "LLMConfig":
        """Load configuration from a JSON file."""
        if not Path(file_path).exists():
            return cls()
        
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
            
            return cls(**config_data)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return cls()

def create_default_config() -> LLMConfig:
    """Create a default configuration."""
    config = LLMConfig()
    
    # Try to auto-detect available providers
    if os.getenv("OPENAI_API_KEY"):
        config.provider = "openai"
    elif os.getenv("ANTHROPIC_API_KEY"):
        config.provider = "anthropic"
    elif os.getenv("LOCAL_MODEL_PATH"):
        config.provider = "local"
    
    return config

def main():
    """Demo the LLM configuration system."""
    print("LLM Configuration System")
    print("=" * 40)
    
    # Create default config
    config = create_default_config()
    
    # Show current configuration
    print(f"Provider: {config.provider}")
    print(f"Configured: {config.is_configured()}")
    
    # Show provider info
    provider_info = config.get_provider_info()
    print(f"Provider Info: {provider_info}")
    
    # Save configuration
    config.save_to_file()
    
    # Load configuration
    loaded_config = LLMConfig.load_from_file()
    print(f"Loaded provider: {loaded_config.provider}")

if __name__ == "__main__":
    main()
