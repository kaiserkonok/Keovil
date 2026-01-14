import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from pathlib import Path

class ModelEngine:
    def __init__(self, model_type="coder"):
        self.model_dir = Path.home() / ".k_rag_storage" / "models"
        os.makedirs(self.model_dir, exist_ok=True)

        self.models = {
            "coder": {"repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "file": "qwen2.5-coder-7b-instruct-q4_k_m.gguf"},
            "instruct": {"repo": "Qwen/Qwen2.5-7B-Instruct-GGUF", "file": "qwen2.5-7b-instruct-q4_k_m.gguf"}
        }

        model_info = self.models[model_type]
        model_path = os.path.join(self.model_dir, model_info["file"])

        if not os.path.exists(model_path):
            print(f"Downloading {model_type}...")
            hf_hub_download(repo_id=model_info["repo"], filename=model_info["file"], local_dir=self.model_dir)

        # Optimized for RTX 5060 Ti 16GB
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,      # Fully offload to VRAM
            n_ctx=16384,          # Increased context for RAG/Coding
            flash_attn=True,      # Much faster on your 50-series card
            chat_format="chatml", # Explicitly set for Qwen/ChatML models
            verbose=False         # Clean terminal output
        )

    def generate(self, system_prompt, user_prompt):
        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # Removed manual stop tokens as 'chatml' format handles <|im_end|>
            temperature=0.1, # 0.1 is usually better than 0 for code stability
            max_tokens=2048   # Ensure response doesn't get cut off mid-code
        )
        return output["choices"][0]["message"]["content"]