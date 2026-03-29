"""
Keovil - Private RAG System for Documents & Structured Data

A powerful retrieval-augmented generation system that runs entirely locally.
Supports document Q&A and SQL query generation.

Usage:
    from keovil import KeovilRAG

    rag = KeovilRAG(data_dir="/path/to/files")
    rag.ingest(["file1.pdf", "file2.txt"])
    answer = rag.query("What is this about?")
"""

from .rag import KeovilRAG
from .config import KeovilConfig

__version__ = "0.1.0"
__all__ = ["KeovilRAG", "KeovilConfig"]
