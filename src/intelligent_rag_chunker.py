import hashlib
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class Chunk:
    """Helper class to match the 'c.text' and 'c.id' expected by the RAG engine."""
    def __init__(self, text, chunk_id, metadata):
        self.text = text
        self.id = chunk_id
        self.meta = metadata

class IntelligentChunker:
    def __init__(self):
        self._chunk_size = 1000
        self._chunk_overlap = 100

    def chunk_document(self, text):
        # 1. Define headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, 
            strip_headers=False
        )
        md_header_splits = markdown_splitter.split_text(text)
        
        # 2. Use Recursive splitter on the resulting documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size, 
            chunk_overlap=self._chunk_overlap
        )

        # Split the markdown documents further if they are too large
        final_docs = text_splitter.split_documents(md_header_splits)

        processed_chunks = []
        for i, doc in enumerate(final_docs):
            # Generate a unique ID based on the text content hash + index
            # This helps the RAG engine track specific chunks accurately
            content_hash = hashlib.md_index = hashlib.sha256(doc.page_content.encode()).hexdigest()[:8]
            chunk_id = f"chunk_{i}_{content_hash}"
            
            # Wrap in our helper class
            processed_chunks.append(Chunk(
                text=doc.page_content,
                chunk_id=chunk_id,
                metadata=doc.metadata
            ))

        return processed_chunks