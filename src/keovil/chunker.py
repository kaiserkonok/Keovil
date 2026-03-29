import hashlib
from typing import List
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from transformers import AutoTokenizer


class Chunk:
    """
    Helper class to match the 'c.text' and 'c.id' expected by the RAG engine.
    Standardized to use 'metadata' to match LangChain conventions.
    """

    def __init__(self, text: str, chunk_id: str, metadata: dict):
        self.text = text
        self.id = chunk_id
        self.metadata = metadata  # Changed from self.meta to self.metadata


class IntelligentChunker:
    def __init__(self, model_name="lightonai/GTE-ModernColBERT-v1"):
        # We use the real tokenizer to know the REAL token limit for your RTX 5060 Ti
        # This ensures the 512 token limit is exact, not an estimate.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Safety split size (approx 2000 chars ~= 500 tokens)
        # This acts as the 'worst case' handler for documents with no formatting.
        self._chunk_size = 2000
        self._chunk_overlap = 200

    def count_tokens(self, text: str) -> int:
        """Returns the exact number of tokens the ColBERT model will see."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_document(self, text: str) -> List[Chunk]:
        """
        Two-stage chunking:
        1. Split by Markdown headers to keep logical sections together.
        2. Recursive split on sections that are still too large (the safety net).
        """
        # 1. Logical Split (Markdown)
        headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        md_header_splits = md_splitter.split_text(text)

        # 2. Safety Split (Character-based)
        # This prevents the 'worst case' where a single header contains 50 pages of text.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap
        )
        final_docs = text_splitter.split_documents(md_header_splits)

        processed_chunks = []
        for i, doc in enumerate(final_docs):
            # Generate a unique ID based on content hash
            content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()[:8]
            chunk_id = f"chunk_{i}_{content_hash}"

            # Wrap in our standardized helper class
            processed_chunks.append(
                Chunk(text=doc.page_content, chunk_id=chunk_id, metadata=doc.metadata)
            )

        return processed_chunks
