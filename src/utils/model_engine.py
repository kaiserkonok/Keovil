import os
from langchain_ollama import OllamaLLM

_llm_instance = None


def get_llm():
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    if not ollama_host.startswith("http"):
        ollama_host = f"http://{ollama_host}"

    _llm_instance = OllamaLLM(
        model="qwen2.5-coder:7b-instruct",
        temperature=0,
        base_url=ollama_host,
    )
    return _llm_instance
