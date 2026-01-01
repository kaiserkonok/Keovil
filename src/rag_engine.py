import os
import json
import hashlib
import threading
from pathlib import Path
from typing import List, Dict, Any, Union
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, PatternMatchingEventHandler
from datetime import datetime
from langchain_core.documents import Document
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM
import intelligent_rag_chunker
from utils.document_processor import DocumentProcessor
from colbert_engine import ColBERTEngine
import torch


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
    ITALIC = '\033[3m'


# ----------------------
# Watchdog handler
# ----------------------
class NewFileHandler(PatternMatchingEventHandler):
    def __init__(self, rag_instance):
        # Define EXACTLY what the RAG system is allowed to touch
        super().__init__(
            patterns=["*.txt", "*.pdf", "*.docx", "*.pptx", "*.md"],
            ignore_directories=True,
            case_sensitive=False
        )
        self.rag = rag_instance

    def on_created(self, event):
        print(f"File created and allowed: {event.src_path}")
        self.rag.ingest([event.src_path])

    def on_modified(self, event):
        self.rag.ingest([event.src_path])

    def on_deleted(self, event):
        self.rag.remove_file(event.src_path)


# ----------------------
# College RAG System
# ----------------------
class CollegeRAG:
    def __init__(self, data_dir=None, top_k=5, llm_model='llama3.2:latest', temperature=0):
        home = Path.home()
        base_storage = home / ".k_rag_storage"
        self.data_dir = Path(data_dir or base_storage / "data").absolute()
        self.manifest_path = base_storage / "vector_manifest.json"

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(base_storage, exist_ok=True)

        self.top_k = top_k
        self.lock = threading.Lock()
        self.chat_history = []

        # 1. Load Manifest State
        self.manifest = self._load_manifest()

        # 2. Initialize Engines
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"{Colors.HEADER}Initializing ColBERT Engine on: {Colors.BOLD}{device}{Colors.ENDC}")

        self.engine = ColBERTEngine(collection_name="krag", device=device)
        self.doc_processor = DocumentProcessor(use_gpu=torch.cuda.is_available())
        self.chunker = intelligent_rag_chunker.IntelligentChunker()

        self.llm = OllamaLLM(model=llm_model, streaming=True, temperature=temperature)
        self.query_llm = OllamaLLM(model='qwen2.5:7b-instruct', temperature=0)

        # --- UPDATED: History-Aware Retriever Setup ---
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone search query. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # This automatically handles the "Rewrite Query" logic using your ColBERT engine
        self.history_aware_retriever = create_history_aware_retriever(
            self.query_llm,
            self.engine.as_retriever(search_kwargs={"k": self.top_k}),
            contextualize_q_prompt
        )

        # 3. Automatic Startup Sync (Atomic & Idempotent)
        self._initial_sync()

        # 4. Start Watchdog
        self.observer = Observer()
        self.observer.schedule(NewFileHandler(self), str(self.data_dir), recursive=True)
        self.observer.start()
        print(f"{Colors.OKCYAN}👀 Monitoring {self.data_dir} for changes...{Colors.ENDC}")

    def _get_file_hash(self, filepath):
        """Generates MD5 hash for change detection."""
        hasher = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                while chunk := f.read(8192): hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            print(f"{Colors.FAIL}Error hashing {filepath}: {e}{Colors.ENDC}")
            return None

    def _load_manifest(self):
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=4)

    def _initial_sync(self):
        """Standardizes the vector store with the local filesystem on boot, filtering for RAG-only files."""
        print(f"{Colors.OKCYAN}[Sync] Reconciling Store...{Colors.ENDC}")

        # 1. Define strictly supported extensions for the RAG/ColBERT pipeline
        # This prevents CSV/Excel/SQL files from entering the vector store
        SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.pptx', '.md'}

        # 2. Verify Qdrant isn't empty (Self-healing if DB was wiped manually)
        try:
            current_points = self.engine.get_points_count()
            if current_points == 0 and self.manifest:
                print(f"{Colors.WARNING}[Sync] Qdrant is empty. Forcing full re-index.{Colors.ENDC}")
                self.manifest = {}
        except Exception as e:
            print(f"{Colors.FAIL}[Sync] Qdrant connection error: {e}{Colors.ENDC}")

        # 3. Discovery: Only pick up files that match our supported extensions
        current_files = {
            str(p.absolute()): p for p in self.data_dir.rglob('*')
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        }

        # 4. A. Cleanup deletions (Manifest has it, Disk doesn't)
        # This also handles cases where a user might have changed a .txt to a .csv;
        # the .txt record will be purged because it's no longer in 'current_files'
        for path in list(self.manifest.keys()):
            if path not in current_files:
                self.remove_file(path)

        # 5. B. Process new/modified (Disk has it, Manifest is missing it or hash differs)
        to_process = []
        for p_str, p_obj in current_files.items():
            f_hash = self._get_file_hash(p_str)
            if p_str not in self.manifest or self.manifest[p_str] != f_hash:
                to_process.append(p_str)

        if to_process:
            print(f"{Colors.OKBLUE}[Sync] Found {len(to_process)} new/modified files.{Colors.ENDC}")
            self.ingest(to_process)
        else:
            print(f"{Colors.OKGREEN}[Sync] Filesystem is clean. No duplicates or invalid files indexed.{Colors.ENDC}")

    def aggregate_to_limit(self, raw_chunks: List[Any], token_limit: int = 512):
        """
        Standardizes fragments into perfect 512-token blocks.
        Now uses the standardized .metadata attribute.
        """
        standardized_docs = []
        current_text_block = []
        current_tokens = 0
        current_source = None

        for c in raw_chunks:
            # Access metadata and text safely
            # DocumentProcessor/Chunker now both use .metadata
            source = c.metadata.get("source", "Unknown")

            # Handles LangChain (.page_content) and our Chunker (.text)
            text_content = getattr(c, 'page_content', getattr(c, 'text', ""))
            tokens = self.chunker.count_tokens(text_content)

            # TRIGGER: File change OR Token limit reached
            file_changed = (source != current_source and current_source is not None)
            limit_reached = (current_tokens + tokens > token_limit)

            if file_changed or limit_reached:
                if current_text_block:
                    standardized_docs.append(Document(
                        page_content="\n\n".join(current_text_block),
                        metadata={"source": current_source}
                    ))
                # Reset for the next block
                current_text_block, current_tokens = [], 0

            current_text_block.append(text_content)
            current_tokens += tokens
            current_source = source

        # Catch the final block remaining in the loop
        if current_text_block:
            standardized_docs.append(Document(
                page_content="\n\n".join(current_text_block),
                metadata={"source": current_source}
            ))

        return standardized_docs

    def ingest(self, new_files: List[str] = None):
        """
        Synchronizes file system changes with the Qdrant vector database.
        Ensures absolute path consistency and applies the 512-token aggregation.
        """
        if not new_files:
            return

        with self.lock:
            # 1. Normalize all incoming paths to Absolute Strings
            valid_paths = []
            for f in new_files:
                p = Path(f).absolute()
                if p.exists():
                    valid_paths.append(str(p))

            if not valid_paths:
                return

            # 2. Purge old vectors first
            for p_str in valid_paths:
                print(f"{Colors.WARNING}[Sync] Purging old state for: {os.path.basename(p_str)}{Colors.ENDC}")
                try:
                    self.engine.delete_by_source(p_str)
                except Exception as e:
                    print(f"❌ Error during purge of {p_str}: {e}")

            # 3. Process and Vectorize
            print(f"{Colors.OKCYAN}[Ingest] Vectorizing {len(valid_paths)} files...{Colors.ENDC}")
            try:
                # Get raw fragments (handles the 'worst case' messy formatting)
                raw_docs = self.doc_processor.convert_to_documents(valid_paths, self.chunker)

                if raw_docs:
                    # --- THE UPDATE: APPLY YOUR AGGREGATION LOGIC ---
                    print(f"{Colors.OKBLUE}[Ingest] Aggregating fragments into standardized blocks...{Colors.ENDC}")
                    final_docs = self.aggregate_to_limit(raw_docs, token_limit=512)

                    print(
                        f"{Colors.OKBLUE}[Ingest] Created {len(final_docs)} blocks from {len(raw_docs)} fragments.{Colors.ENDC}")

                    # Multi-vector ingestion for ColBERT using aggregated blocks
                    self.engine.ingest_batches(final_docs, batch_size=32)

                    # 4. Update manifest to prevent redundant processing
                    for p_str in valid_paths:
                        self.manifest[p_str] = self._get_file_hash(p_str)

                    self._save_manifest()
                    print(f"{Colors.OKGREEN}[Ingest] Success: Optimized chunks updated.{Colors.ENDC}")
                else:
                    print(f"{Colors.WARNING}[Ingest] No text content extracted from files.{Colors.ENDC}")

            except Exception as e:
                print(f"❌ Ingestion failed: {e}")
                import traceback
                traceback.print_exc()

    def remove_file(self, fpath):
        """Removes vectors and manifest entry for a file."""
        with self.lock:
            p_str = str(Path(fpath).absolute())
            self.engine.delete_by_source(p_str)
            if p_str in self.manifest:
                del self.manifest[p_str]
            self._save_manifest()
            print(f"{Colors.WARNING}[Remove] Purged: {os.path.basename(fpath)}{Colors.ENDC}")

    def _format_chat_history(self, chat_history):
        formatted_chat = ""

        for (curr_chatter, curr_chat) in chat_history:
            chat = f"{"User" if curr_chatter == 'You' else 'AI'}: {curr_chat}"
            formatted_chat += chat + "\n"

        return formatted_chat

    def ask(self, query, chat_history=None, stream=False):
        # 1. Determine which history to use (Flask passed or internal)
        history = chat_history if chat_history is not None else self.chat_history

        # 2. Convert raw history tuples to LangChain Message objects for the retriever
        # This is required for create_history_aware_retriever to function
        lc_history = []
        for role, text in history[-6:]:
            if role == "You":
                lc_history.append(HumanMessage(content=text))
            else:
                lc_history.append(AIMessage(content=text))

        print(f"\n{Colors.HEADER}{'=' * 20} COLBERT SEARCH (History-Aware) {'=' * 20}{Colors.ENDC}")

        # 3. Retrieve Documents
        # This one call uses Qwen to contextualize the query AND searches Qdrant via ColBERT
        try:
            docs = self.history_aware_retriever.invoke({
                "input": query,
                "chat_history": lc_history
            })
        except Exception as e:
            print(f"{Colors.FAIL}[Error] Retrieval failed: {e}{Colors.ENDC}")
            return "I encountered an error while searching for information."

        if not docs:
            print(f"{Colors.FAIL}[Error] No matches found in Qdrant!{Colors.ENDC}")
            return "I don't have information on that yet."

        # --- DEBUG PRINTING (RETAINED & ADAPTED FOR DOC OBJECTS) ---
        print(f"\n{Colors.HEADER}┌{'─' * 78}┐{Colors.ENDC}")
        context_chunks = []
        for i, doc in enumerate(docs):
            src = os.path.basename(doc.metadata.get("source", "Unknown"))
            # Note: score isn't always returned by the generic retriever wrapper
            print(f"{Colors.OKCYAN}  Match {i + 1} {Colors.ENDC}| {src}")
            context_chunks.append(doc.page_content)
            print(
                f"  {Colors.OKBLUE}↳{Colors.ENDC} {Colors.ITALIC}{doc.page_content[:150].replace('\n', ' ')}...{Colors.ENDC}")
        print(f"{Colors.HEADER}└{'─' * 78}┘{Colors.ENDC}\n")

        # 4. Final Answer Generation
        context_text = "\n\n".join(context_chunks)
        formatted_chat = self._format_chat_history(history[-6:])

        # Your custom prompt exactly as requested
        prompt = f"""
            You are Lora, a private AI Assistant.

            Answer the question **only using the given context**. 
            If someone does a typing mistake, like typing the same name with some different spelling, still give the correct answer with correct names. If you get context that is not related to the query, don't get confused with it. You might need to ignore it. 
            Be friendly. Current date and time: {datetime.now()}

            Use chat history to understand the conversation better and make your responses more natural and coherent.

            CHAT HISTORY:
            {formatted_chat}

            CONTEXT:
            {context_text}

            QUESTION:
            {query}

            ANSWER:
        """.strip()

        if stream:
            print(f"{Colors.BOLD}Lora:{Colors.ENDC} ", end="", flush=True)
            ans = ""
            for chunk in self.llm.stream(prompt):
                print(chunk, end="", flush=True)
                ans += chunk

            # Update internal history only if we aren't using Flask's passed history
            if chat_history is None:
                self.chat_history.append(("You", query))
                self.chat_history.append(("AI", ans))

            print(f"\n\n{Colors.HEADER}{'=' * 56}{Colors.ENDC}")
            return ans

        res = self.llm.invoke(prompt)
        if chat_history is None:
            self.chat_history.append(("You", query))
            self.chat_history.append(("AI", res))
        return res


# ----------------------
# Entry Point / Testing
# ----------------------
if __name__ == "__main__":
    # CollegeRAG will automatically sync on init
    rag = CollegeRAG()

    print(f"\n{Colors.OKGREEN}System initialized. Type your question below (exit to quit).{Colors.ENDC}")
    try:
        while True:
            question = input(f"{Colors.BOLD}You:{Colors.ENDC} ")
            if question.lower() in ['exit', 'quit']: break
            if not question.strip(): continue
            rag.ask(question, stream=True)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Shutting down...{Colors.ENDC}")
        rag.observer.stop()
        rag.observer.join()