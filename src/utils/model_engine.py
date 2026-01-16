import os
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from langchain_community.chat_models import ChatLlamaCpp  # New Import


class Theme:
    """Cyberpunk Terminal Theme"""
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class ModelEngine:
    def __init__(self, model_type="instruct"):
        self.model_dir = Path.home() / ".k_rag_storage" / "models"
        os.makedirs(self.model_dir, exist_ok=True)

        self.models = {
            "coder": {
                "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                "file": "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
            },
            "instruct": {
                "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
                "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
            }
        }

        model_info = self.models[model_type]
        self.model_path = str(self.model_dir / model_info["file"])

        # Download logic (keeping your existing logic)
        if not os.path.exists(self.model_path):
            hf_hub_download(repo_id=model_info["repo"], filename=model_info["file"], local_dir=self.model_dir)

        # Hardware Scouting
        self.has_gpu = torch.cuda.is_available()
        self.gpu_layers = -1 if self.has_gpu else 0

        print(f"{Theme.MAGENTA}{Theme.BOLD}--- HARDWARE CALIBRATED ---{Theme.RESET}")
        print(f"{Theme.CYAN}Device:{Theme.RESET} {'RTX 5060 Ti (16GB VRAM)' if self.has_gpu else 'CPU'}")

        self._llm_instance = None

    @property
    def llm(self):
        """Returns the LangChain-compatible ChatLlamaCpp instance"""
        if self._llm_instance is None:
            print(f"{Theme.MAGENTA}🧠 Loading Model into VRAM...{Theme.RESET}")

            # Using ChatLlamaCpp instead of raw Llama
            self._llm_instance = ChatLlamaCpp(
                model_path=self.model_path,
                n_gpu_layers=self.gpu_layers,
                n_ctx=16384,  # Large context for RAG + History
                n_batch=512,  # Optimized for 5060 Ti throughput
                flash_attn=True,  # Massively reduces VRAM usage
                last_n_tokens_size=64,
                chat_format="chatml",
                temperature=0,  # Critical for SQL/RAG precision
                verbose=False,
                streaming=True  # Allows for real-time response UI
            )
        return self._llm_instance

    def generate(self, system_prompt, user_prompt):
        """Helper for simple direct calls without a chain"""
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = self.llm.invoke(messages)
        return response.content