from dataclasses import dataclass
from typing import Optional


@dataclass
class KeovilConfig:
    """Configuration for Keovil RAG system."""

    data_dir: Optional[str] = None
    """Directory containing files to index. If None, uses default storage location."""

    storage_dir: Optional[str] = None
    """Base storage directory. If None, uses default: ~/.keovil_storage or ~/.keovil_storage_dev"""

    collection_name: str = "keovil"
    """Name of the vector collection in Qdrant."""

    auto_watch: bool = False
    """Enable automatic file watching. When True, monitors data_dir for changes."""

    auto_index: bool = True
    """Automatically index files on initialization."""

    top_k: int = 5
    """Number of documents to retrieve for each query."""

    mode: str = "development"
    """Mode: 'development' or 'production'. Affects default storage paths."""

    llm_provider: str = "ollama"
    """LLM provider: 'ollama' (local) or 'openai', 'anthropic' (cloud)."""

    llm_model: str = "qwen2.5-coder:7b-instruct"
    """Model name to use for LLM."""

    chunker_model: str = "lightonai/GTE-ModernColBERT-v1"
    """Model for token counting in chunking."""
