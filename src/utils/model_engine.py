import os
import torch
import threading
from pathlib import Path
from huggingface_hub import hf_hub_download
from langchain_community.chat_models import ChatLlamaCpp


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
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
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

        # Updated shared constants in your ModelEngine
        self.repo = "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
        self.file = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        self.model_path = str(self.model_dir / self.file)

        if not os.path.exists(self.model_path):
            hf_hub_download(repo_id=self.repo, filename=self.file, local_dir=self.model_dir)

        # 1. DYNAMIC HARDWARE SCOUTING
        self.has_gpu = torch.cuda.is_available()
        self.allocated_layers = 0
        self.dynamic_ctx = 8192  # Default for CPU
        self.vram_info = "N/A"

        if self.has_gpu:
            try:
                # Get real-time VRAM info
                free_b, total_b = torch.cuda.mem_get_info()
                free_gb = free_b / (1024 ** 3)
                total_gb = total_b / (1024 ** 3)

                # Buffer: Leave 15% or 0.5GB (whichever is larger) for OS
                buffer = max(0.5, total_gb * 0.15)
                usable_vram = max(0, free_gb - buffer)

                # Qwen 2.5 7B Q4_K_M weights: ~160MB per layer (28 layers total)
                self.allocated_layers = max(0, min(int(usable_vram / 0.160), 28))

                # Context (KV Cache) calculation: ~110MB per 1k tokens
                remaining_for_ctx = max(0, usable_vram - (self.allocated_layers * 0.160))
                calc_tokens = int((remaining_for_ctx / 0.11) * 1024)

                # Floor of 5120 to ensure RAG works, cap at 32k
                self.dynamic_ctx = max(5120, min(calc_tokens, 32768))
                self.vram_info = f"{free_gb:.1f}GB Free / {total_gb:.1f}GB Total"

            except Exception as e:
                self.allocated_layers = 0
                self.dynamic_ctx = 8192

        self._print_calibration()
        self._llm_instance = None
        self._initialized = True

    def _print_calibration(self):
        print(f"{Theme.MAGENTA}{Theme.BOLD}--- UNIVERSAL ENGINE CALIBRATED ---{Theme.RESET}")
        device_name = torch.cuda.get_device_name(0) if self.has_gpu else "CPU"
        print(f"{Theme.CYAN}Hardware:  {Theme.RESET}{device_name}")
        print(f"{Theme.CYAN}VRAM:      {Theme.RESET}{self.vram_info}")
        print(f"{Theme.CYAN}Offload:   {Theme.RESET}{self.allocated_layers}/28 layers to GPU")
        print(f"{Theme.CYAN}Context:   {Theme.RESET}{self.dynamic_ctx} tokens (Safe-Limit)")

    @property
    def llm(self):
        if self._llm_instance is None:
            self._llm_instance = ChatLlamaCpp(
                model_path=self.model_path,
                n_gpu_layers=self.allocated_layers,
                n_ctx=self.dynamic_ctx,
                n_batch=512 if self.has_gpu else 128,
                flash_attn=True if self.allocated_layers > 10 else False,
                chat_format="chatml",
                temperature=0,
                verbose=False,
                streaming=True
            )
        return self._llm_instance

    def generate(self, system_prompt, user_prompt):
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        return self.llm.invoke(messages).content