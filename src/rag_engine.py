import os
import json
import shutil
from datetime import datetime
import threading
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
from concurrent.futures import ThreadPoolExecutor
from utils.document_processor import DocumentProcessor


# ----------------------
# ANSI Color Class for Debugging
# ----------------------
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ITALIC = '\033[3m'  # <--- Added this line


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
        self.doc_processor = DocumentProcessor(use_gpu=torch.cuda.is_available())

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

    def get_docs(self, data_dir):
        all_paths = [Path(os.path.join(root, f)) for root, _, files in os.walk(data_dir) for f in files]
        return self.doc_processor.convert_to_documents(all_paths, self.chunker)

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
        with self.lock:
            if not new_files: return

            paths = [Path(f) for f in new_files]

            # 1. Cleanup vectorstore for these specific paths to prevent duplicates
            if self.vectorestore:
                for p in paths:
                    source_str = str(p)
                    ids = [k for k, v in self.vectorestore.docstore._dict.items()
                           if v.metadata.get("source") == source_str]
                    if ids:
                        self.vectorestore.delete(ids)

            # 2. Process via the utility (GPU accelerated via DocumentProcessor)
            print(f"{Colors.OKCYAN}[Ingest] Processing {len(paths)} files...{Colors.ENDC}")
            new_docs = self.doc_processor.convert_to_documents(paths, self.chunker)

            # 3. Finalize
            if new_docs:
                if self.vectorestore is None:
                    self.vectorestore = FAISS.from_documents(new_docs, self.embedding_model)
                else:
                    self.vectorestore.add_documents(new_docs)

                # Update the in-memory document list for BM25
                self.docs = list(self.vectorestore.docstore._dict.values())

                # REBUILD indexes to include new data
                self._build_bm25_index()

                # FIX: Update the retriever so rag.ask() sees the new chunks immediately
                self.retriever = self.vectorestore.as_retriever(search_kwargs={'k': self.rerank_top_k})

                # Persist to disk
                self.vectorestore.save_local(self.store_dir)

                print(f"{Colors.OKGREEN}[Ingest] Success: {len(new_docs)} new chunks added to index.{Colors.ENDC}")


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
            
            - Keep the query minimal. Do not add extra words, names, or details unless they are strictly required to resolve pronouns or ambiguity.
            - If the query is already clear, leave it unchanged.
            - Resolve pronouns, fragments, or vague references using the chat history so the query stands alone.
            - Do not expand the query with institution names, locations, or other context unless the user explicitly mentioned them.
            - Your rewritten query should be as short and focused as possible.

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
            raw_response = self.llm.invoke(prompt)
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
        use_internal_history = False
        if chat_history is None:
            chat_history = self.chat_history
            use_internal_history = True

        print(f"\n{Colors.HEADER}{'=' * 20} RAG PIPELINE START {'=' * 20}{Colors.ENDC}")

        # 1️⃣ Fast Intent Extraction (Intelligence)
        print(f"{Colors.OKCYAN}[1/5] Analyzing Intent...{Colors.ENDC}")
        intel = self.get_search_intelligence(query, chat_history)
        rewritten_query = intel.get('standalone', query)
        keywords = intel.get('keywords', [])

        print(f"    {Colors.BOLD}• Rewritten:{Colors.ENDC} {rewritten_query}")
        print(f"    {Colors.BOLD}• Keywords:{Colors.ENDC} {', '.join(keywords)}")
        print(f"    {Colors.BOLD}• Hyde:{Colors.ENDC} {intel.get('hyde', 'N/A')}")

        if not self.vectorestore:
            print(f"{Colors.FAIL}[Error] Vectorstore is empty!{Colors.ENDC}")
            return "I haven't learned anything yet!"

        # 2️⃣ Parallel Multi-Path Retrieval
        print(f"{Colors.OKCYAN}[2/5] Executing Hybrid Retrieval (Vector + BM25)...{Colors.ENDC}")
        k_initial = 25

        def vector_search(q):
            return self.vectorestore.similarity_search(q, k=k_initial)

        def bm25_search(kw):
            tokenized_query = " ".join(kw).lower().split()
            scores = self.bm25_retriever.get_scores(tokenized_query)
            indices = np.argsort(scores)[::-1][:k_initial]
            return [self.doc_map[i] for i in indices if i in self.doc_map]

        with ThreadPoolExecutor(max_workers=3) as executor:
            f1 = executor.submit(vector_search, rewritten_query)
            f2 = executor.submit(vector_search, intel.get('hyde', rewritten_query))
            f3 = executor.submit(bm25_search, keywords)
            results = [f1.result(), f2.result(), f3.result()]

        # 3️⃣ Deduplication & RRF
        fused_candidates = reciprocal_rank_fusion(results)
        print(f"    {Colors.OKBLUE}• Fused into {len(fused_candidates)} unique candidates.{Colors.ENDC}")

        # 4️⃣ Pruned Reranking (GPU Accelerated)
        print(f"{Colors.OKCYAN}[3/5] Reranking top 20 candidates on RTX 5060 Ti...{Colors.ENDC}")
        rerank_pool = fused_candidates[:20]
        pairs = [[rewritten_query, doc.page_content] for doc in rerank_pool]

        # Calculate raw logits
        raw_logits = self.reranker.predict(pairs, batch_size=20, show_progress_bar=False)
        ranked_indices = np.argsort(raw_logits)[::-1]

        # 5️⃣ Dynamic Selection (Sigmoid + Gap Method + Token Safety)
        print(f"{Colors.OKCYAN}[4/5] Applying Adaptive Context Selection...{Colors.ENDC}")

        # Convert Logits to Probabilities
        reranked_scores = [1 / (1 + np.exp(-float(raw_logits[i]))) for i in ranked_indices]
        reranked_docs = [rerank_pool[i] for i in ranked_indices]

        # Configuration
        absolute_threshold = 0.35  # Confidence Floor
        relative_drop_limit = 0.4  # 40% Cliff
        MAX_WORDS = 3500  # Safety Limit

        final_top_k = 1
        current_word_count = len(reranked_docs[0].page_content.split())

        # Log first document
        print(
            f"      {Colors.OKGREEN}[Rank 1]{Colors.ENDC} Score: {reranked_scores[0]:.4f} | Words: {current_word_count}")

        for i in range(1, len(reranked_scores)):
            curr_s = reranked_scores[i]
            prev_s = reranked_scores[i - 1]
            doc_words = len(reranked_docs[i].page_content.split())

            # A. Check absolute floor
            if curr_s < absolute_threshold:
                print(
                    f"      {Colors.WARNING}• Stop: Score {curr_s:.4f} below floor ({absolute_threshold}){Colors.ENDC}")
                break

            # B. Check for the "Cliff" (only if previous was strong)
            if prev_s > 0.50:
                drop = (prev_s - curr_s) / (prev_s + 1e-6)
                if drop > relative_drop_limit:
                    print(f"      {Colors.WARNING}• Stop: Cliff detected ({drop:.2%} drop){Colors.ENDC}")
                    break

            # C. Check Token Safety
            if (current_word_count + doc_words) > MAX_WORDS:
                print(f"      {Colors.WARNING}• Stop: MAX_WORDS limit reached{Colors.ENDC}")
                break

            current_word_count += doc_words
            final_top_k += 1
            print(f"      {Colors.OKBLUE}[Rank {i + 1}]{Colors.ENDC} Score: {curr_s:.4f} | Words: {doc_words}")

        top_docs = reranked_docs[:final_top_k]

        # DEBUG PRINTING
        # ----------------------
        print(f"\n{Colors.HEADER}┌{'─' * 78}┐{Colors.ENDC}")
        print(
            f"{Colors.HEADER}│ {Colors.BOLD}PROMOTED CONTEXT CHUNKS (Reranked & Filtered){Colors.ENDC}{Colors.HEADER}{' ' * 32}│{Colors.ENDC}")
        print(f"{Colors.HEADER}├{'─' * 78}┤{Colors.ENDC}")

        for i, doc in enumerate(top_docs):
            score = reranked_scores[i]
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            chunk_id = doc.metadata.get("chunk_id", "N/A")

            # Header for each chunk
            print(
                f"{Colors.OKCYAN}  Rank {i + 1} {Colors.ENDC}| {Colors.OKGREEN}Score: {score:.4f}{Colors.ENDC} | {Colors.BOLD}File:{Colors.ENDC} {source} | {Colors.BOLD}ID:{Colors.ENDC} {chunk_id}")

            # Content Preview (Wrapped for readability)
            content = doc.page_content.replace("\n", " ").strip()
            print(f"  {Colors.OKBLUE}↳{Colors.ENDC} {Colors.ITALIC}{content}{Colors.ENDC}")

            # Separator between chunks
            if i < len(top_docs) - 1:
                print(f"{Colors.OKBLUE}  " + "┄" * 70 + f"{Colors.ENDC}")

        print(f"{Colors.HEADER}└{'─' * 78}┘{Colors.ENDC}\n")

        # 6️⃣ Stream/Generate Response (Original Lora Prompt)
        print(f"\n{Colors.OKCYAN}[5/5] Generating Answer from {len(top_docs)} chunks...{Colors.ENDC}")
        print(f"{Colors.HEADER}{'=' * 56}{Colors.ENDC}\n")

        context_text = "\n\n".join([d.page_content for d in top_docs])

        prompt = f"""
            You are Lora, a private AI Assistant.

            Answer the question **only using the given context**. 
            If someone does a typing mistake, like typing the same name with some different spelling, still give the correct answer with correct names. If you get context that is not related to the query, don't get confused with it. You might need to ignore it. 
            Be friendly. Current date and time: {datetime.now()}

            Use chat history to understand the conversation better and make your responses more natural and coherent.

            CHAT HISTORY:
            {chat_history[:-3]}

            CONTEXT:
            {context_text}

            QUESTION:
            {query}

            ANSWER:
        """.strip()

        if stream:
            response = ""
            print(f"{Colors.BOLD}Lora:{Colors.ENDC} ", end="", flush=True)
            for chunk in self.llm.stream(prompt):
                print(chunk, end="", flush=True)
                response += chunk
            if use_internal_history:
                self.chat_history.append(("You", query))
                self.chat_history.append(("AI", response))
            print(f"\n\n{Colors.HEADER}{'=' * 56}{Colors.ENDC}")
            return response

        res = self.llm.invoke(prompt)
        if use_internal_history:
            self.chat_history.append(("You", query))
            self.chat_history.append(("AI", res))
        return res


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