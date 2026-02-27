from langchain_ollama import OllamaLLM

_llm_instance = None


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = OllamaLLM(
            model="qwen2.5-coder:7b-instruct",
            temperature=0,
        )
    return _llm_instance
