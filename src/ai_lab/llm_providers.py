"""LLM provider implementations."""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from .llm_config import LLMConfig

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str, context: Optional[str] = None, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        # Only set up client if both package and API key are available
        if openai and config.openai_api_key and config.openai_api_key != "test_key":
            openai.api_key = config.openai_api_key
            self.client = openai
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return bool(self.client and self.config.openai_api_key)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "openai",
            "model": self.config.openai_model,
            "available": self.is_available(),
            "max_tokens": self.config.openai_max_tokens,
            "temperature": self.config.openai_temperature
        }
    
    def generate_response(self, prompt: str, context: Optional[str] = None, **kwargs) -> str:
        """Generate response using OpenAI API."""
        if not self.is_available():
            raise RuntimeError("OpenAI provider not available")
        
        # Build the full prompt with context
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = prompt
        
        try:
            response = self.client.ChatCompletion.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Answer questions based on the provided context when available."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=kwargs.get('max_tokens', self.config.openai_max_tokens),
                temperature=kwargs.get('temperature', self.config.openai_temperature),
                timeout=kwargs.get('timeout', self.config.timeout)
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        # Only set up client if both package and API key are available
        if anthropic and config.anthropic_api_key and config.anthropic_api_key != "test_key":
            self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    
    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        return bool(self.client and self.config.anthropic_api_key)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information."""
        return {
            "provider": "anthropic",
            "model": self.config.anthropic_model,
            "available": self.is_available(),
            "max_tokens": self.config.anthropic_max_tokens,
            "temperature": self.config.anthropic_temperature
        }
    
    def generate_response(self, prompt: str, context: Optional[str] = None, **kwargs) -> str:
        """Generate response using Anthropic Claude API."""
        if not self.is_available():
            raise RuntimeError("Anthropic provider not available")
        
        # Build the full prompt with context
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = prompt
        
        try:
            response = self.client.messages.create(
                model=self.config.anthropic_model,
                max_tokens=kwargs.get('max_tokens', self.config.anthropic_max_tokens),
                temperature=kwargs.get('temperature', self.config.anthropic_temperature),
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

class LocalProvider(LLMProvider):
    """Local LLM provider using llama-cpp-python."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        if Llama and config.local_model_path:
            try:
                self.model = Llama(
                    model_path=config.local_model_path,
                    n_ctx=2048,
                    n_threads=4
                )
            except Exception as e:
                print(f"Warning: Could not load local model: {e}")
    
    def is_available(self) -> bool:
        """Check if local model is available."""
        return bool(self.model and self.config.local_model_path)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get local model information."""
        return {
            "provider": "local",
            "model_path": self.config.local_model_path,
            "model_type": self.config.local_model_type,
            "available": self.is_available(),
            "max_tokens": self.config.local_max_tokens,
            "temperature": self.config.local_temperature
        }
    
    def generate_response(self, prompt: str, context: Optional[str] = None, **kwargs) -> str:
        """Generate response using local model."""
        if not self.is_available():
            raise RuntimeError("Local model not available")
        
        # Build the full prompt with context
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = prompt
        
        try:
            response = self.model(
                full_prompt,
                max_tokens=kwargs.get('max_tokens', self.config.local_max_tokens),
                temperature=kwargs.get('temperature', self.config.local_temperature),
                stop=["\n\n", "Question:", "Context:"]
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            raise RuntimeError(f"Local model error: {e}")

class MockProvider(LLMProvider):
    """Mock provider for testing and development."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def is_available(self) -> bool:
        """Mock provider is always available."""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "provider": "mock",
            "model": "mock-model",
            "available": True,
            "message": "Mock provider for testing"
        }
    
    def generate_response(self, prompt: str, context: Optional[str] = None, **kwargs) -> str:
        """Generate mock response."""
        if context:
            return f"Mock response based on context: {context[:50]}... and question: {prompt[:30]}..."
        else:
            return f"Mock response to: {prompt[:50]}..."

def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Factory function to create the appropriate LLM provider."""
    if config.provider == "openai":
        return OpenAIProvider(config)
    elif config.provider == "anthropic":
        return AnthropicProvider(config)
    elif config.provider == "local":
        return LocalProvider(config)
    else:
        return MockProvider(config)

def main():
    """Demo the LLM providers."""
    print("LLM Providers Demo")
    print("=" * 40)
    
    # Create config
    config = LLMConfig(provider="none")
    
    # Test each provider
    providers = [
        OpenAIProvider(config),
        AnthropicProvider(config),
        LocalProvider(config),
        MockProvider(config)
    ]
    
    for provider in providers:
        print(f"\nProvider: {provider.__class__.__name__}")
        print(f"Available: {provider.is_available()}")
        print(f"Info: {provider.get_model_info()}")
        
        if provider.is_available():
            try:
                response = provider.generate_response("What is machine learning?")
                print(f"Response: {response[:100]}...")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
