import os
import json
import shutil
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
    OKGREEN = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'  # Red
    ENDC = '\033[0m'  # Reset color
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

    def on_deleted(self, event):
        if event.is_directory:
            return
        print(f"{Colors.WARNING}[Watcher]{Colors.ENDC} File deleted: {event.src_path}")
        self.rag.remove_file(event.src_path)


# ----------------------
# College RAG System
# ----------------------
class CollegeRAG:
    def __init__(self, data_dir=None, top_k=7, rerank_top_k=15, store_dir=None, llm_model='qwen2.5:14b-instruct-q4_K_M', temperature=0):
        # --- PATH CONFIGURATION (Home Directory Defaults) ---
        home = str(Path.home())
        base_storage = os.path.join(home, ".k_rag_storage")

        if data_dir is None:
            data_dir = os.path.join(base_storage, "data")
        if store_dir is None:
            store_dir = os.path.join(base_storage, "faiss_store")

        self.data_dir = os.path.abspath(data_dir)
        self.store_dir = os.path.abspath(store_dir)

        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.store_dir, exist_ok=True)

        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.docs = []
        # self.chat_history is used only for the interactive terminal test
        self.chat_history = []
        self.vectorestore = None
        self.bm25_retriever = None
        self.doc_map = {}
        self.lock = threading.Lock()

        # LLM + reranker (MODIFIED: Now accepts model and temperature dynamically)
        self.llm = OllamaLLM(model=llm_model, streaming=True, temperature=temperature)

        # Intelligence LLM (3B) for query strategy
        self.intel_llm = OllamaLLM(model='qwen2.5:14b-instruct-q4_K_M', temperature=0)

        self.embedding_model = OllamaEmbeddings(model='bge-m3:latest')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"{Colors.HEADER}Reranker is initializing on device: {Colors.BOLD}{device}{Colors.ENDC}")

        # --- SAFE INITIALIZATION ---
        # Upgraded to BGE Reranker for better accuracy
        self.reranker = CrossEncoder(
            model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
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

        # FIX: Check if directory exists before listing
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            return docs

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
            store_path = self.store_dir

            if force_create:
                print(f"{Colors.WARNING}Force Create active. Wiping old index and clearing memory...{Colors.ENDC}")
                self.vectorestore = None
                self.docs = []  # <--- Crucial: Wipes the in-memory list
                self.bm25_retriever = None

                if os.path.exists(store_path):
                    # We only delete store_dir (the FAISS cache), NOT data_dir (your PDFs)
                    shutil.rmtree(store_path)
                os.makedirs(store_path, exist_ok=True)

            # Check if the folder is empty or non-existent
            store_exists = os.path.exists(store_path) and len(os.listdir(store_path)) > 0

            if store_exists and not force_create:
                try:
                    print(f'{Colors.OKGREEN}Loading existing FAISS vectorstore from {store_path}...{Colors.ENDC}')
                    self.vectorestore = FAISS.load_local(
                        store_path,
                        self.embedding_model,
                        allow_dangerous_deserialization=True
                    )
                except Exception as e:
                    print(f"{Colors.WARNING}Failed to load store: {e}. Recreating...{Colors.ENDC}")
                    self.vectorestore = None

            # Rebuild if forced or if loading failed
            if self.vectorestore is None:
                print(f'{Colors.WARNING}Scanning source files for fresh indexing...{Colors.ENDC}')
                # Since self.docs was cleared above, get_docs will only find current files
                self._load_docs()

                if self.docs:
                    self.vectorestore = FAISS.from_documents(self.docs, self.embedding_model)
                    self.vectorestore.save_local(store_path)
                    self._build_bm25_index()
                else:
                    print(f"{Colors.FAIL}No documents found in {self.data_dir}.{Colors.ENDC}")

            # Ensure the retriever is updated to use the new index
            if self.vectorestore:
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

            all_new_docs = []  # Collect all new chunks

            for fpath in new_files:
                file_path = Path(fpath)

                if file_path.name.endswith("~") or file_path.name.startswith(".#") or file_path.suffix == ".swp":
                    print(f"{Colors.WARNING}[Ingest] Skipping temporary file: {fpath}{Colors.ENDC}")
                    continue

                print(f"{Colors.BOLD}[Ingest] Updating file: {fpath}{Colors.ENDC}")

                # Remove old chunks only if vectorestore exists
                if self.vectorestore:
                    ids_to_delete = [k for k, doc in self.vectorestore.docstore._dict.items()
                                     if doc.metadata.get("source") == fpath]

                    if ids_to_delete:
                        self.vectorestore.delete(ids_to_delete)
                        print(f"{Colors.OKBLUE}[Ingest] Removed {len(ids_to_delete)} old chunks.{Colors.ENDC}")

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
                new_file_docs = [
                    Document(page_content=c.text, metadata={"source": fpath, "chunk_id": c.id, **c.meta})
                    for c in chunks
                ]
                all_new_docs.extend(new_file_docs)

            if not all_new_docs:
                return

            # FIX: If vectorestore was None (empty start), create it now. Else, add.
            if self.vectorestore is None:
                self.vectorestore = FAISS.from_documents(all_new_docs, self.embedding_model)
            else:
                self.vectorestore.add_documents(all_new_docs)

            print(f"{Colors.OKGREEN}[Ingest] Added {len(all_new_docs)} new chunks.{Colors.ENDC}")

            self.docs = list(self.vectorestore.docstore._dict.values())
            self._build_bm25_index()
            self.retriever = self.vectorestore.as_retriever(search_kwargs={'k': self.rerank_top_k})

            self.vectorestore.save_local(self.store_dir)
            print(f"{Colors.OKGREEN}[Ingest] Ingestion completed!{Colors.ENDC}")

    def remove_file(self, fpath):
        """
        Remove all chunks of a deleted file from the vectorstore and BM25 index.
        """
        with self.lock:
            if not self.vectorestore:
                return

            ids_to_delete = [
                doc_id for doc_id, doc in self.vectorestore.docstore._dict.items()
                if doc.metadata.get("source") == fpath
            ]

            if not ids_to_delete:
                print(f"{Colors.WARNING}[Remove] No chunks found for deleted file: {fpath}{Colors.ENDC}")
                return

            self.vectorestore.delete(ids_to_delete)
            print(f"{Colors.OKBLUE}[Remove] Deleted {len(ids_to_delete)} chunks from: {fpath}{Colors.ENDC}")

            # Update docs and rebuild BM25
            self.docs = list(self.vectorestore.docstore._dict.values())

            # FIX: If last file deleted, reset store to None
            if not self.docs:
                self.vectorestore = None
                self.bm25_retriever = None
                print(f"{Colors.WARNING}[Remove] All documents removed. Store is empty.{Colors.ENDC}")
            else:
                self._build_bm25_index()
                self.retriever = self.vectorestore.as_retriever(search_kwargs={'k': self.rerank_top_k})
                self.vectorestore.save_local(self.store_dir)

            print(f"{Colors.OKGREEN}[Remove] Vectorstore updated.{Colors.ENDC}")

    # ----------------------
    # Get Search Intelligence (World Class Upgrade)
    # ----------------------
    def get_search_intelligence(self, query, chat_history):
        prompt = f"""
            You are a Search Intent Architect. Your goal is to prepare a multi-path search strategy.

            ### TASK 1: STANDALONE QUERY
            Rewrite the "User Query" to be entirely self-contained. 
            - Use the "Chat History" to resolve all pronouns (he, she, it, they, that) and vague references.
            - Ensure the subject of the conversation is explicitly named.
            - Maintain the original intent without adding external knowledge.
            - Don't try to make it big. 

            ### TASK 2: HYPOTHETICAL DOCUMENT (HyDE)
            Create a brief snippet of text that would serve as a perfect, direct answer to the standalone query. 
            - Use professional and factual language.
            - This is for semantic matching; focus on how a document would logically state the information.

            ### TASK 3: KEYWORD EXTRACTION
            Extract the most significant search terms strictly from the "Standalone Query".
            - Focus on unique names, technical nouns, and core actions.
            - Do not include common stop words or words only found in the HyDE section.

            ### CONTEXT:
            Chat History: {chat_history}
            User Query: {query}

            ### OUTPUT FORMAT:
            JSON only.
            {{
                "standalone": "string",
                "hyde": "string",
                "keywords": ["list", "of", "terms"]
            }}
        """.strip()

        try:
            raw_response = self.intel_llm.invoke(prompt)
            start = raw_response.find('{')
            end = raw_response.rfind('}') + 1
            if start == -1: return {"standalone": query, "hyde": query, "keywords": query.split()}

            intel = json.loads(raw_response[start:end])

            # Smart Filtering: Ensure keywords exist in the standalone text
            intel['keywords'] = [w for w in intel['keywords'] if w.lower() in intel['standalone'].lower()]

            return intel
        except Exception:
            return {"standalone": query, "hyde": query, "keywords": [w for w in query.split() if len(w) > 3]}

    # ----------------------
    # Ask a question (streaming or normal) - MODIFIED FOR HISTORY MANAGEMENT
    # ----------------------
    def ask(self, query, chat_history: Union[List[tuple], None] = None, stream=False):

        # Determine which history to use and if we should update the internal state.
        use_internal_history = False
        if chat_history is None:
            chat_history = self.chat_history
            use_internal_history = True  # This is true only for the terminal test

        # 1️⃣ Setup and Query Intelligence Transformation

        # World-Class Upgrade: Get Standalone, HyDE, and Keywords in one call
        intel = self.get_search_intelligence(query, chat_history)
        print(intel)
        rewritten_query = intel['standalone']
        print(f"{Colors.OKCYAN}[DEBUG] Rewritten Query: {Colors.BOLD}{rewritten_query}{Colors.ENDC}")

        # FIX: Friendly message instead of failing if index is empty
        if not self.vectorestore or not self.bm25_retriever:
            msg = "I haven't learned anything yet! Please upload some documents so I can help you."
            if stream:
                # We yield the msg chunk by chunk to simulate streaming if needed
                print(f"{Colors.BOLD}AI:{Colors.ENDC} {msg}")
                return msg
            return msg

        with self.lock:
            k_retrieve = self.rerank_top_k

            # 2️⃣ Retrieve from FAISS Path 1 (Standalone Semantic Search)
            vector_results_standalone = self.vectorestore.similarity_search(rewritten_query, k=k_retrieve)

            # World-Class Upgrade: Retrieve from FAISS Path 2 (HyDE Semantic Search)
            vector_results_hyde = self.vectorestore.similarity_search(intel['hyde'], k=k_retrieve)

            print(
                f"{Colors.OKBLUE}[DEBUG] Vector Search Retrieved: {len(vector_results_standalone) + len(vector_results_hyde)} chunks.{Colors.ENDC}")

            # 3️⃣ Retrieve from BM25 (Keyword/Lexical Search using specific keywords)
            kw_string = " ".join(intel['keywords'])
            tokenized_query = kw_string.lower().split(" ")
            bm25_scores = self.bm25_retriever.get_scores(tokenized_query)
            bm25_indices = np.argsort(bm25_scores)[::-1][:k_retrieve]
            bm25_results = [self.doc_map[i] for i in bm25_indices if i in self.doc_map]
            print(f"{Colors.OKBLUE}[DEBUG] BM25 Search Retrieved: {len(bm25_results)} chunks.{Colors.ENDC}")

        # 4️⃣ Reciprocal Rank Fusion (RRF) - Now fusing 3 paths
        fused_candidates = reciprocal_rank_fusion([vector_results_standalone, vector_results_hyde, bm25_results])
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

        # 🟢 NEW: 5.5 Adaptive Top-K Selection (The "Gap" Method)

        if not probabilities.size:
            print(f"{Colors.FAIL}[DEBUG] No documents found after RRF.{Colors.ENDC}")
            return "No relevant documents found."

        # 1. Start with the indices sorted by the reranker (descending)
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probabilities = probabilities[sorted_indices]

        # 2. Check for a significant score drop (We only look at the top self.top_k chunks for the gap)
        score_drops = []
        for i in range(1, min(self.top_k, len(sorted_probabilities))):
            # Calculate the drop from the previous chunk
            drop = sorted_probabilities[i - 1] - sorted_probabilities[i]
            score_drops.append((drop, i))

        # 3. Determine the cut-off point

        MAX_DROP_THRESHOLD = 0.02
        cut_index = self.top_k  # Default to 7

        if score_drops:
            # STRATEGY: Find the FIRST drop that exceeds our threshold.
            # This prevents "drifting" into low-quality chunks further down the list.
            for drop, index in score_drops:
                if drop > MAX_DROP_THRESHOLD:
                    cut_index = index
                    print(
                        f"{Colors.WARNING}[DEBUG] Gap Detected! Cutting off at index {index} (Drop: {drop:.4f}){Colors.ENDC}")
                    break  # Exit the loop at the first significant drop

        # 4. Filter the final list (Always includes at least 1, and max self.top_k)
        final_indices = sorted_indices[:max(1, cut_index)]
        top_docs_indices = final_indices

        # 5. Re-sort the final selection for the final output (not strictly necessary as it's already sorted)
        filtered_indices_sorted = top_docs_indices

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
            You are Lora, a private AI Assistant.

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

        print(prompt)

        if stream:
            response = ""
            print(f"{Colors.BOLD}AI:{Colors.ENDC} ", end="", flush=True)
            for chunk in self.llm.stream(prompt):
                print(chunk, end="", flush=True)
                response += chunk

            # 🟢 Conditional History Update (for terminal test only)
            if use_internal_history:
                self.chat_history.append(("You", query))
                self.chat_history.append(("AI", response))

            print()
            return response
        else:
            response = self.llm.invoke(prompt)

            # 🟢 Conditional History Update (for terminal test only)
            if use_internal_history:
                self.chat_history.append(("You", query))
                self.chat_history.append(("AI", response))

            return response


# ----------------------
# Run interactively
# ----------------------
if __name__ == "__main__":
    # Updated to use home-based default storage if no data_dir provided
    rag = CollegeRAG()
    rag.create_vectorestore(force_create=True)

    if rag.docs:
        for doc in rag.docs:
            print(
                f'{Colors.HEADER}___ Starting of Chunk ({os.path.basename(doc.metadata.get("source", "N/A"))}) ___{Colors.ENDC}\n')
            print(doc.page_content)
            print(f'\n{Colors.HEADER}___ Ending of Chunk ___{Colors.ENDC}')
    else:
        print(f"{Colors.WARNING}No initial documents found. Waiting for uploads to {rag.data_dir}...{Colors.ENDC}")

    try:
        while True:
            question = input(f"{Colors.BOLD}You:{Colors.ENDC} ")
            # History works here because rag.ask uses and updates self.chat_history
            rag.ask(question, stream=True)
    except KeyboardInterrupt:
        print(f"\n{Colors.FAIL}Stopping...{Colors.ENDC}")
        rag.observer.stop()
        rag.observer.join()