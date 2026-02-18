import os
from pathlib import Path
from langchain_community.chat_models import ChatLlamaCpp
from langchain_openai import ChatOpenAI
# import OllamaLLM
from langchain_ollama import OllamaLLM
from langchain_community.llms import LlamaCpp

class KeovilModelEngine:
    _instance = None

    @classmethod
    def get_llm(cls, temperature=0):
        if cls._instance is None:
            home = Path.home()
            model_path = os.path.join(home, ".keovil_storage/keo/evil.gguf")
            print(f"🚀 [ModelEngine] Loading weights into RTX 5060 Ti...")
            # cls._instance = ChatOpenAI(
            #     base_url="http://localhost:7977/v1",
            #     api_key="not-needed",
            #     model="local-model",
            #     temperature=temperature,
            # )

            cls._instance = ChatLlamaCpp(model_path=model_path, temperature=0, n_gpu_layers=-1, n_ctx=32768, flash_attn=True, stop=["<|im_end|>", "<|endoftext|>"])

            # cls._instance = OllamaLLM(
            #     model='qwen2.5-coder:7b-instruct',
            #     temperature=0,
            # )
        return cls._instance