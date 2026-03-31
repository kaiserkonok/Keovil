import os
from typing import Optional
from functools import lru_cache

from .llm_config import LLMConfig, get_default_config


# Default model per provider
DEFAULT_MODELS = {
    "ollama": "qwen2.5-coder:7b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "openrouter": "openai/gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
}


def get_llm(config: Optional[LLMConfig] = None):
    """Get LLM instance based on config.

    Args:
        config: LLMConfig object. If None, loads from file/env/defaults.

    Returns:
        LangChain LLM instance
    """
    if config is None:
        config = get_default_config()

    return _create_llm(config)


def _create_llm(config: LLMConfig):
    """Create LLM based on provider."""

    if config.provider == "ollama":
        return _create_ollama(config)
    elif config.provider == "openai":
        return _create_openai(config)
    elif config.provider == "anthropic":
        return _create_anthropic(config)
    elif config.provider == "openrouter":
        return _create_openrouter(config)
    elif config.provider == "gemini":
        return _create_gemini(config)
    else:
        # Fallback to ollama
        return _create_ollama(config)


def _create_ollama(config: LLMConfig):
    """Create Ollama LLM."""
    from langchain_ollama import OllamaLLM

    host = config.ollama_host
    if not host.startswith("http"):
        host = f"http://{host}"

    return OllamaLLM(
        model=config.model,
        temperature=config.temperature,
        base_url=host,
    )


def _create_openai(config: LLMConfig):
    """Create OpenAI LLM."""
    from langchain_openai import ChatOpenAI

    api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY or configure in settings."
        )

    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        api_key=api_key,
    )


def _create_anthropic(config: LLMConfig):
    """Create Anthropic LLM."""
    from langchain_anthropic import ChatAnthropic

    api_key = config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY or configure in settings."
        )

    # Anthropic uses different model format
    model = config.model
    if not model.startswith("claude-"):
        model = f"claude-3-5-haiku-20241022"

    return ChatAnthropic(
        model=model,
        temperature=config.temperature,
        anthropic_api_key=api_key,
    )


def _create_openrouter(config: LLMConfig):
    """Create OpenRouter LLM."""
    from langchain_openai import ChatOpenAI

    api_key = config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenRouter API key required. Set OPENROUTER_API_KEY or configure in settings."
        )

    # OpenRouter uses OpenAI-compatible API
    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def _create_gemini(config: LLMConfig):
    """Create Google Gemini LLM."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key required. Set GEMINI_API_KEY or configure in settings."
        )

    return ChatGoogleGenerativeAI(
        model=config.model,
        temperature=config.temperature,
        google_api_key=api_key,
    )


def get_model_name(provider: str, custom_model: Optional[str] = None) -> str:
    """Get model name for a provider."""
    if custom_model:
        return custom_model
    return DEFAULT_MODELS.get(provider, DEFAULT_MODELS["ollama"])
