import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal
from functools import lru_cache


Provider = Literal["ollama", "openai", "anthropic", "openrouter"]


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""

    provider: Provider = "ollama"
    model: str = "qwen2.5-coder:7b"

    # Ollama
    ollama_host: str = "http://127.0.0.1:11434"

    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    # Temperature
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load config from environment variables."""
        config = cls()

        # Provider
        if os.getenv("KEOVIL_PROVIDER"):
            config.provider = os.getenv("KEOVIL_PROVIDER")

        # Model
        if os.getenv("KEOVIL_MODEL"):
            config.model = os.getenv("KEOVIL_MODEL")

        # Ollama
        if os.getenv("OLLAMA_HOST"):
            config.ollama_host = os.getenv("OLLAMA_HOST")

        # API Keys
        if os.getenv("OPENAI_API_KEY"):
            config.openai_api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("ANTHROPIC_API_KEY"):
            config.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if os.getenv("OPENROUTER_API_KEY"):
            config.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        # Temperature
        if os.getenv("KEOVIL_TEMPERATURE"):
            config.temperature = float(os.getenv("KEOVIL_TEMPERATURE"))

        return config

    @classmethod
    def from_file(cls, filepath: Optional[str] = None) -> "LLMConfig":
        """Load config from JSON file."""
        if filepath is None:
            filepath = str(Path.home() / ".keovil" / "config.json")

        config_file = Path(filepath)

        # If file doesn't exist, start with env vars or defaults
        if not config_file.exists():
            return cls.from_env()

        try:
            with open(config_file) as f:
                data = json.load(f)

            config = cls()

            # Apply file config
            if "provider" in data:
                config.provider = data["provider"]
            if "model" in data:
                config.model = data["model"]
            if "ollama_host" in data:
                config.ollama_host = data["ollama_host"]
            if "openai_api_key" in data:
                config.openai_api_key = data["openai_api_key"]
            if "anthropic_api_key" in data:
                config.anthropic_api_key = data["anthropic_api_key"]
            if "openrouter_api_key" in data:
                config.openrouter_api_key = data["openrouter_api_key"]
            if "temperature" in data:
                config.temperature = data["temperature"]

            # Environment variables override file config
            return cls.from_env()

        except (json.JSONDecodeError, IOError):
            return cls.from_env()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "provider": self.provider,
            "model": self.model,
            "ollama_host": self.ollama_host,
            "openai_api_key": self.openai_api_key or "",
            "anthropic_api_key": self.anthropic_api_key or "",
            "openrouter_api_key": self.openrouter_api_key or "",
            "temperature": self.temperature,
        }

    @classmethod
    def create_default(cls) -> "LLMConfig":
        """Create default config with sensible defaults."""
        return cls.from_env()


def get_default_config() -> LLMConfig:
    """Get default LLM config (file -> env -> default)."""
    return LLMConfig.from_file()


def save_config(config: LLMConfig, filepath: Optional[str] = None) -> None:
    """Save config to JSON file."""
    if filepath is None:
        filepath = str(Path.home() / ".keovil" / "config.json")

    config_file = Path(filepath)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
