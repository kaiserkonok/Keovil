import os
from datetime import datetime
import threading
import pdfplumber
import docx
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder
import intelligent_rag_chunker
from pathlib import Path
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Union

# ----------------------
# ANSI Color Class for Debugging
# ----------------------
class Colors:
    HEADER = '\033[95m'  # Magenta
    OKBLUE = '\033[94m'  # Blue
    OKCYAN = '\033[96m'  # Cyan
    OKGREEN = '\033[92m' # Green
    WARNING = '\033[93m' # Yellow
    FAIL = '\033[91m'    # Red
    ENDC = '\033[0m'     # Reset color
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# ----------------------
# RRF Utility Function (Outside the Class)
# ----------------------
RRF_K = 60


def reciprocal_rank_fusion(
        results: List[List[Document]],
        k: int = RRF_K
) -> List[Document]:
    """
    Combines ranked results from multiple lists (e.g., Vector and BM25)
    using Reciprocal Rank Fusion (RRF).
    """
    fused_scores = {}
    doc_map = {}

    # Iterate over each retrieval method's results
    for rank_list in results:
        # doc_id is typically a unique hash or combination of source/chunk_id
        for rank, doc in enumerate(rank_list):
            # Ensure doc is a Document object before accessing metadata
            if not isinstance(doc, Document):
                continue

            doc_id = doc.metadata.get("source") + str(doc.metadata.get("chunk_id"))

            # Map doc_id to Document object
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

            # Calculate RRF score: 1 / (K + rank + 1)
            score = 1 / (k + rank + 1)

            # Accumulate the score
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + score

    # Sort document IDs by the fused RRF score
    sorted_doc_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)

    # Return the final list of Document objects
    return [doc_map[doc_id] for doc_id in sorted_doc_ids]


# ----------------------
# Watchdog handler for real-time ingestion
# ----------------------
class NewFileHandler(FileSystemEventHandler):
    def __init__(self, rag_instance):
        self.rag = rag_instance

    def on_created(self, event):
        if event.is_directory:
            return
        print(f"{Colors.OKCYAN}[Watcher]{Colors.ENDC} New file detected: {event.src_path}")
        self.rag.ingest(new_files=[event.src_path])

    def on_modified(self, event):
        if event.is_directory:
            return
        print(f"{Colors.OKCYAN}[Watcher]{Colors.ENDC} File modified: {event.src_path}")
        self.rag.ingest(new_files=[event.src_path])


# ----------------------
# College RAG System
# ----------------------
class CollegeRAG:
    def __init__(self, data_dir, top_k=7, rerank_top_k=15, store_dir=None):
        self.data_dir = data_dir
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.docs = []
        self.chat_history = []
        self.vectorestore = None
        self.bm25_retriever = None
        self.doc_map = {}
        self.lock = threading.Lock()

        if store_dir is None:
            store_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "faiss_store")
        self.store_dir = os.path.abspath(store_dir)

        # LLM + reranker
        self.llm = OllamaLLM(model='qwen2.5:7b-instruct', streaming=True, temperature=1)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"{Colors.HEADER}Reranker is initializing on device: {Colors.BOLD}{device}{Colors.ENDC}")

        # --- SAFE INITIALIZATION ---
        self.reranker = CrossEncoder(
            model_name='cross-encoder/ms-marco-MiniLM-L-12-v2',
            device=device
        )

        # Initialize your intelligent chunker
        self.chunker = intelligent_rag_chunker.IntelligentChunker()

        # Load docs and create vectorstore
        self.create_vectorestore(force_create=False)

        if self.vectorestore:
            # Rebuild self.docs and BM25 index from loaded store
            self.docs = list(self.vectorestore.docstore._dict.values())
            self._build_bm25_index()

        # Start Watchdog for automatic ingestion
        event_handler = NewFileHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.data_dir, recursive=True)
        self.observer.start()

    # ----------------------
    # Helper to build/rebuild the BM25 index
    # ----------------------
    def _build_bm25_index(self):
        if not self.docs:
            self.bm25_retriever = None
            self.doc_map = {}
            return

        print(f'{Colors.OKBLUE}[Hybrid Search] Creating BM25 index...{Colors.ENDC}')

        # 1. Prepare tokenized corpus
        bm25_tokenized_corpus = [
            doc.page_content.lower().split(" ") for doc in self.docs
        ]

        # 2. Initialize the BM25 retriever
        self.bm25_retriever = BM25Okapi(bm25_tokenized_corpus)

        # 3. Create map from corpus index to Document object
        self.doc_map = {i: doc for i, doc in enumerate(self.docs)}

        print(f"{Colors.OKBLUE}[Hybrid Search] BM25 index created with {len(self.docs)} documents.{Colors.ENDC}")

    # ----------------------
    # Load and chunk documents using IntelligentChunker
    # ----------------------
    def _load_docs(self):
        self.docs = self.get_docs(self.data_dir)

    def get_docs(self, data_dir, docs=None):
        if docs is None:
            docs = []

        for fname in os.listdir(data_dir):
            fpath = os.path.join(data_dir, fname)
            if os.path.isdir(fpath):
                self.get_docs(fpath, docs)
                continue

            ext = fname.lower().split('.')[-1]
            try:
                text = ""
                if ext == 'txt':
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()

                elif ext == 'docx':
                    doc = docx.Document(fpath)
                    text = "\n".join([p.text for p in doc.paragraphs])

                elif ext == 'pdf':
                    with pdfplumber.open(fpath) as pdf:
                        pages = []
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                pages.append(page_text)
                        text = "\n".join(pages)

                if text:
                    # Use your intelligent chunker
                    chunk_objs = self.chunker.chunk_document(text)
                    for c in chunk_objs:
                        docs.append(Document(
                            page_content=c.text,
                            metadata={"source": fpath, "chunk_id": c.id, **c.meta}
                        ))

            except Exception as e:
                print(f"{Colors.FAIL}Failed {fpath}: {e}{Colors.ENDC}")
                continue

        return docs

    # ----------------------
    # Create or load FAISS vectorstore
    # ----------------------
    def create_vectorestore(self, force_create=False):
        with self.lock:
            embedding_model = OllamaEmbeddings(model='bge-m3:latest')
            store_path = self.store_dir

            if os.path.exists(store_path) and not force_create:
                print(f'{Colors.OKGREEN}Loading FAISS vectorstore...{Colors.ENDC}')
                self.vectorestore = FAISS.load_local(store_path, embedding_model,
                                                     allow_dangerous_deserialization=True)
            else:
                print(f'{Colors.WARNING}Creating new vectorestores...{Colors.ENDC}')
                self._load_docs()
                print(f"Total documents loaded: {len(self.docs)}")
                self.vectorestore = FAISS.from_documents(self.docs, embedding_model)
                self.vectorestore.save_local(store_path)
                self._build_bm25_index()

            self.retriever = self.vectorestore.as_retriever(search_kwargs={'k': self.rerank_top_k})

    # ----------------------
    # Incremental ingestion
    # ----------------------
    def ingest(self, new_files=None):
        """
        Incrementally ingest files and rebuilds the BM25 index afterwards.
        """
        with self.lock:
            if not new_files:
                return

            for fpath in new_files:
                file_path = Path(fpath)

                if file_path.name.endswith("~") or file_path.name.startswith(".#") or file_path.suffix == ".swp":
                    print(f"{Colors.WARNING}[Ingest] Skipping temporary file: {fpath}{Colors.ENDC}")
                    continue

                print(f"{Colors.BOLD}[Ingest] Updating file: {fpath}{Colors.ENDC}")

                ids_to_delete = [k for k, doc in self.vectorestore.docstore._dict.items()
                                 if doc.metadata.get("source") == fpath]

                if ids_to_delete:
                    self.vectorestore.delete(ids_to_delete)
                    print(f"{Colors.OKBLUE}[Ingest] Removed {len(ids_to_delete)} old chunks.{Colors.ENDC}")

                text = ""
                ext = file_path.suffix.lower()[1:]
                try:
                    if ext == "txt":
                        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                    elif ext == "docx":
                        docx_doc = docx.Document(fpath)
                        text = "\n".join([p.text for p in docx_doc.paragraphs])
                    elif ext == "pdf":
                        with pdfplumber.open(fpath) as pdf:
                            pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
                            text = "\n".join(pages)
                    else:
                        print(f"{Colors.FAIL}[Ingest] Unsupported file type: {fpath}{Colors.ENDC}")
                        continue
                except Exception as e:
                    print(f"{Colors.FAIL}[Ingest] Failed to read {fpath}: {e}{Colors.ENDC}")
                    continue

                if not text.strip():
                    print(f"{Colors.WARNING}[Ingest] File {fpath} is empty. Skipping.{Colors.ENDC}")
                    continue

                chunks = self.chunker.chunk_document(text.strip())
                new_docs = [
                    Document(page_content=c.text, metadata={"source": fpath, "chunk_id": c.id, **c.meta})
                    for c in chunks
                ]

                if not new_docs:
                    print(f"{Colors.WARNING}[Ingest] No chunks to add from {fpath}. Skipping.{Colors.ENDC}")
                    continue

                self.vectorestore.add_documents(new_docs)
                print(f"{Colors.OKGREEN}[Ingest] Added {len(new_docs)} new chunks from {fpath}.{Colors.ENDC}")

            self.docs = list(self.vectorestore.docstore._dict.values())
            self._build_bm25_index()
            self.retriever = self.vectorestore.as_retriever(search_kwargs={'k': self.rerank_top_k})

            self.vectorestore.save_local(self.store_dir)
            print(f"{Colors.OKGREEN}[Ingest] Ingestion completed for {len(new_files)} file(s)!{Colors.ENDC}")

    # ----------------------
    # Rewrite query (Unchanged)
    # ----------------------
    def rewrite_query(self, query, chat_history=None):
        if chat_history is None:
            chat_history = self.chat_history

        prompt = f"""
            You are a RAG search helper. Rewrite the user’s query so it is independent and self-contained. 

            Guidelines:
            1. Keep the query minimal. Do not add extra words, names, or details unless they are strictly required to resolve pronouns or ambiguity.
            2. If the query is already clear, leave it unchanged.
            3. Resolve pronouns, fragments, or vague references using the chat history so the query stands alone.
            4. Do not expand the query with institution names, locations, or other context unless the user explicitly mentioned them.
            5. Your rewritten query should be as short and focused as possible.

            Now rewrite:
            User's Query: {query}

            Chat History:
            {chat_history}

            Rewritten Query:
        """.strip()

        return self.llm.invoke(prompt).strip()

    # ----------------------
    # Ask a question (streaming or normal) - MODIFIED FOR HYBRID RAG
    # ----------------------
    def ask(self, query, chat_history=None, stream=False):
        # 1️⃣ Setup and Query Rewriting (Unchanged)
        if chat_history is None:
            chat_history = self.chat_history

        # 1️⃣ Rewrite query to be self-contained
        rewritten_query = self.rewrite_query(query, chat_history)
        print(f"{Colors.OKCYAN}[DEBUG] Rewritten Query: {Colors.BOLD}{rewritten_query}{Colors.ENDC}")

        if not self.vectorestore or not self.bm25_retriever:
            print(f"{Colors.FAIL}[DEBUG] RAG system not ready or empty index.{Colors.ENDC}")
            return "RAG system not fully initialized or empty index."

        k_retrieve = self.rerank_top_k

        # 2️⃣ Retrieve from FAISS (Vector/Semantic Search)
        vector_results = self.vectorestore.similarity_search(rewritten_query, k=k_retrieve)
        print(f"{Colors.OKBLUE}[DEBUG] Vector Search Retrieved: {len(vector_results)} chunks.{Colors.ENDC}")

        # 3️⃣ Retrieve from BM25 (Keyword/Lexical Search)
        tokenized_query = rewritten_query.lower().split(" ")
        bm25_scores = self.bm25_retriever.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k_retrieve]
        bm25_results = [self.doc_map[i] for i in bm25_indices if i in self.doc_map]
        print(f"{Colors.OKBLUE}[DEBUG] BM25 Search Retrieved: {len(bm25_results)} chunks.{Colors.ENDC}")

        # 4️⃣ Reciprocal Rank Fusion (RRF)
        fused_candidates = reciprocal_rank_fusion([vector_results, bm25_results])
        print(f"{Colors.OKGREEN}[DEBUG] RRF Fused Candidates (Unique): {len(fused_candidates)} chunks.{Colors.ENDC}")

        if not fused_candidates:
            print(f"{Colors.FAIL}[DEBUG] No documents retrieved after fusion.{Colors.ENDC}")
            return "No relevant documents found."

        # 5️⃣ Rerank and Score Normalization
        rerank_pool = fused_candidates[:k_retrieve]
        sentences = [doc.page_content for doc in rerank_pool]
        pairs = [[rewritten_query, s] for s in sentences]
        scores = self.reranker.predict(pairs)

        # Sigmoid Score Normalization
        scores_array = np.array(scores)
        probabilities = 1 / (1 + np.exp(-scores_array))  # Apply Sigmoid function

        # 🟢 NEW: 5.5 Dynamic Relative Filtering (The Robust Solution)

        if not probabilities.size:
            # This should be caught by the earlier check, but kept for robustness
            return "No documents found after RRF."

        # 1. Determine the highest score (the ceiling)
        MAX_PROBABILITY = np.max(probabilities)

        # 2. Set the Relative Confidence Threshold
        # We include any chunk that is at least 70% as good as the best one.
        RELATIVE_CONFIDENCE_PERCENTAGE = 0.70
        DYNAMIC_THRESHOLD = MAX_PROBABILITY * RELATIVE_CONFIDENCE_PERCENTAGE

        # 3. Set a minimum absolute floor (to block total noise, like the 0.0001 table chunks)
        ABSOLUTE_FLOOR = 0.10  # Must be at least 10% confident.

        # Filter the indices: Must meet the DYNAMIC_THRESHOLD OR the ABSOLUTE_FLOOR
        filtered_indices = [
            i for i, prob in enumerate(probabilities)
            if prob >= DYNAMIC_THRESHOLD or prob >= ABSOLUTE_FLOOR
        ]

        # Ensure the absolute best chunk is always included if it passes the absolute floor
        if not filtered_indices and MAX_PROBABILITY >= ABSOLUTE_FLOOR:
            # This covers the case where the highest score is low (e.g., 0.40) and the
            # dynamic threshold (0.28) failed to capture it, but it's still the best bet.
            filtered_indices.append(np.argmax(probabilities))

        if not filtered_indices:
            print(f"{Colors.FAIL}[DEBUG] All documents filtered below absolute floor ({ABSOLUTE_FLOOR}).{Colors.ENDC}")
            return "No highly relevant documents found to answer the question."

        # Re-sort the *filtered* results by probability score
        filtered_indices_sorted = sorted(filtered_indices, key=lambda i: probabilities[i], reverse=True)[:self.top_k]

        # 6️⃣ Collect top documents based on reranker
        print(
            f"{Colors.HEADER}--- Top {len(filtered_indices_sorted)} Reranked Chunks (Dynamic Filtered) ---{Colors.ENDC}")
        top_docs = []
        for i, idx in enumerate(filtered_indices_sorted):
            doc = rerank_pool[idx]
            top_docs.append(doc.page_content)
            source = os.path.basename(doc.metadata.get("source", "N/A"))
            chunk_id = doc.metadata.get("chunk_id", "N/A")

            # Print the new Probability Score
            print(
                f"{Colors.BOLD}[{i + 1}] Prob Score: {probabilities[idx]:.4f}{Colors.ENDC} | Source: {source} (ID: {chunk_id})")
            print(f"    Snippet: {doc.page_content.replace(os.linesep, ' ')}...{Colors.ENDC}")
        print(f"{Colors.HEADER}--------------------------------------{Colors.ENDC}")

        # 7️⃣ Combine top docs into context
        context = "\n\n".join(top_docs)

        # 8️⃣ Generate answer (Unchanged Prompt)
        prompt = f"""
            You are Lora. You are a Private AI Assistant created by Kaiser Konok.

            Answer the question **only using the given context**. 
            If someone does a typing mistake, like typing the same name with some different spelling, still give the correct answer with correct names. If you get context that is not related to the query, don't get confused with it. You might need to ignore it. 
            Be friendly. Current date and time: {datetime.now()}

            Use chat history to understand the conversation better and make your responses more natural and coherent.

            CHAT HISTORY:
            {chat_history}

            CONTEXT:
            {context}

            QUESTION:
            {query}

            ANSWER:
        """.strip()

        if stream:
            # ... (streaming logic remains the same) ...
            response = ""
            print(f"{Colors.BOLD}AI:{Colors.ENDC} ", end="", flush=True)
            for chunk in self.llm.stream(prompt):
                print(chunk, end="", flush=True)
                response += chunk
            chat_history.append(("You", query))
            chat_history.append(("AI", response))
            print()
            return response
        else:
            response = self.llm.invoke(prompt)
            chat_history.append(("You", query))
            chat_history.append(("AI", response))
            return response


# ----------------------
# Run interactively
# ----------------------
if __name__ == "__main__":
    data_dir = '/home/kaiserkonok/computer_programming/K_RAG/test_data/'
    rag = CollegeRAG(data_dir)
    rag.create_vectorestore(force_create=True)

    for doc in rag.docs:
        print(f'{Colors.HEADER}___ Starting of Chunk ({os.path.basename(doc.metadata.get("source", "N/A"))}) ___{Colors.ENDC}\n')
        print(doc.page_content)
        print(f'\n{Colors.HEADER}___ Ending of Chunk ___{Colors.ENDC}')

    try:
        while True:
            question = input(f"{Colors.BOLD}You:{Colors.ENDC} ")
            rag.ask(question, stream=True)
    except KeyboardInterrupt:
        print(f"\n{Colors.FAIL}Stopping...{Colors.ENDC}")
        rag.observer.stop()
        rag.observer.join()