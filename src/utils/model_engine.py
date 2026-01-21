import os
import torch
import threading
from pathlib import Path
from typing import Optional, List, Any, Mapping
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from langchain_core.language_models.llms import LLM


class Theme:
    CYAN, MAGENTA, GREEN, YELLOW, RED = "\033[96m", "\033[95m", "\033[92m", "\033[93m", "\033[91m"
    BOLD, RESET = "\033[1m", "\033[0m"


class K_LLMWrapper(LLM):
    """Universal LangChain Wrapper for RTX 5060 Ti"""
    engine: Any

    @property
    def _llm_type(self) -> str:
        return "blackwell_cuda_engine"

    # THE FIX: This signature is exactly what LangChain v0.3 expects
    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[Any] = None,
            **kwargs: Any,
    ) -> str:
        # Use ChatML formatting to maintain 'Ollama-level' accuracy
        # This wrapper now handles ANY prompt (RAG, SQL, Chat)
        result = self.engine(
            prompt,
            max_tokens=kwargs.get("max_tokens", 2048),
            temperature=kwargs.get("temperature", 0),
            repeat_penalty=1.1,
            stop=stop or ["<|im_end|>", "</s>"]
        )
        return result["choices"][0]["text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"engine": "llama-cpp-python", "device": "RTX 5060 Ti"}


class ModelEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(ModelEngine, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return

        self.model_dir = Path.home() / ".k_rag_storage" / "models"
        os.makedirs(self.model_dir, exist_ok=True)

        # High-precision Q8_0 (8-bit) for best accuracy on your 16GB VRAM
        self.repo = "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
        self.file = "Qwen2.5-Coder-7B-Instruct-Q8_0.gguf"
        self.model_path = str(self.model_dir / self.file)

        if not os.path.exists(self.model_path):
            print(f"{Theme.YELLOW}📥 Pulling weights...{Theme.RESET}")
            hf_hub_download(repo_id=self.repo, filename=self.file, local_dir=self.model_dir)

        # MAX SETTINGS FOR 5060 Ti
        self.raw_llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=-1,  # Forces 100% of model to VRAM
            n_ctx=32768,  # 32k context window for huge RAG docs/schemas
            n_batch=1024,  # Optimized for Blackwell throughput
            flash_attn=True,  # Native hardware acceleration
            use_mlock=True,  # Prevents RAM swapping
            verbose=False
        )

        self._llm_instance = K_LLMWrapper(engine=self.raw_llm)
        self._print_calibration()
        self._initialized = True

    def _print_calibration(self):
        print(f"{Theme.MAGENTA}{Theme.BOLD}--- ENGINE 7.0: BLACKWELL UNLEASHED ---{Theme.RESET}")
        print(f"{Theme.CYAN}Hardware:  {Theme.RESET}RTX 5060 Ti (Compute 9.0)")
        print(f"{Theme.CYAN}Context:   {Theme.RESET}32,768 Tokens")

    @property
    def llm(self):
        """Use this to plug into any LangChain chain."""
        return self._llm_instance