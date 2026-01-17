import os
import hashlib
import threading
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, PatternMatchingEventHandler
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.chat_models import ChatLlamaCpp
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import intelligent_rag_chunker
from utils.document_processor import DocumentProcessor
from colbert_engine import ColBERTEngine
import torch
from langchain_core.globals import set_verbose

set_verbose(True)


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
        super().__init__(
            patterns=["*.txt", "*.pdf", "*.docx", "*.pptx", "*.md"],
            ignore_directories=True,
            case_sensitive=False
        )
        self.rag = rag_instance

    def on_created(self, event):
        # NEW: Queue the file for batching instead of instant ingestion
        self.rag.queue_file(event.src_path)

    def on_modified(self, event):
        # NEW: Queue the file for batching
        self.rag.queue_file(event.src_path)

    def on_deleted(self, event):
        # Deletions remain immediate as they are low-resource
        self.rag.remove_file(event.src_path)


# ----------------------
# College RAG System
# ----------------------
class CollegeRAG:
    def __init__(self, data_dir=None, top_k=5, socketio=None):
        home = Path.home()
        base_storage = home / ".k_rag_storage"
        self.data_dir = Path(data_dir or base_storage / "data").absolute()
        self.socketio = socketio
        self.status = {"state": "idle", "current_file": "", "progress": 0, "total_files": 0}

        self.db_dir = base_storage / "database"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_db = self.db_dir / "manifest.db"
        self._init_manifest_db()

        os.makedirs(self.data_dir, exist_ok=True)
        self.top_k = top_k
        self.lock = threading.Lock()
        self.pending_files = set()
        self.queue_lock = threading.Lock()
        self.chat_history = []

        # --- 1. ENGINE & LLM ---
        # Optimized for RTX 5060 Ti 16GB
        from src.utils.model_engine import ModelEngine
        self.model_engine = ModelEngine()
        self.llm = self.model_engine.llm

        # --- 2. VECTOR & DOCUMENT TOOLS ---
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.engine = ColBERTEngine(collection_name="krag", device=device)
        self.doc_processor = DocumentProcessor(use_gpu=torch.cuda.is_available())
        self.chunker = intelligent_rag_chunker.IntelligentChunker()

        # Base retriever from your ColBERT engine
        base_retriever = self.engine.as_retriever(search_kwargs={"k": self.top_k})

        # Section 3: Rephraser Configuration
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone search query. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Strict QA Prompt: Forces Lora to stick to the facts and stop over-thinking titles.
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are Lora, a precise AI assistant. Answer the user question ONLY using the provided context. "
                "If the information is not present in the context, say you do not know. "
                "Do not use outside knowledge or speculate on titles like CEO/COO unless stated in the text."
                "\n\nCONTEXT:\n{context}"
            )),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # --- 4. HISTORY-AWARE RETRIEVER SETUP ---
        # This is the "powerful" logic that combines the rephraser and retriever
        from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm,
            base_retriever,
            self.contextualize_q_prompt
        )

        # --- 5. START WORKERS ---
        self._initial_sync()
        threading.Thread(target=self._batch_worker, daemon=True).start()

        self.observer = Observer()
        self.observer.schedule(NewFileHandler(self), str(self.data_dir), recursive=True)
        self.observer.start()

    def get_status(self):
        """Returns the current state for the UI."""
        # Check if there are files waiting in the queue
        with self.queue_lock:
            pending_count = len(self.pending_files)

        if self.status["state"] == "idle" and pending_count > 0:
            return {"state": "waiting", "message": f"Waiting for quiet period... ({pending_count} files queued)"}

        return self.status

    def broadcast_status(self):
        if not self.socketio:
            return
        try:
            # We add a specific 'reason' so the frontend can distinguish
            # between vectorizing and just thinking.
            self.socketio.emit('system_status', {
                "is_busy": self.status["state"] != "idle",
                "reason": self.status["state"],  # 'processing', 'waiting', or 'idle'
                "sql_syncing": False,
                "rag": self.get_status()
            }, namespace='/')
        except Exception as e:
            print(f"Socket Error: {e}")

    # ----------------------
    # NEW: SQLite Database Logic
    # ----------------------
    def _init_manifest_db(self):
        """Creates the manifest table if it doesn't exist."""
        conn = sqlite3.connect(self.manifest_db)
        conn.execute("CREATE TABLE IF NOT EXISTS file_hashes (path TEXT PRIMARY KEY, hash TEXT)")
        conn.commit()
        conn.close()

    def _get_stored_hashes(self):
        """Retrieves all file hashes from SQLite."""
        conn = sqlite3.connect(self.manifest_db)
        cursor = conn.execute("SELECT path, hash FROM file_hashes")
        data = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return data

    def _update_manifest_batch(self, file_data: Dict[str, str]):
        """Updates the database with a batch of new file hashes."""
        conn = sqlite3.connect(self.manifest_db)
        conn.executemany("INSERT OR REPLACE INTO file_hashes VALUES (?, ?)", list(file_data.items()))
        conn.commit()
        conn.close()

    # ----------------------
    # NEW: Debounce/Batch Worker Logic
    # ----------------------
    def queue_file(self, path):
        """Adds file to a pending set for the worker to process later."""
        with self.queue_lock:
            self.pending_files.add(str(Path(path).absolute()))

    def _batch_worker(self):
        """Background thread that processes files every 5 seconds if the queue isn't empty."""
        while True:
            time.sleep(5)
            to_process = []
            with self.queue_lock:
                if self.pending_files:
                    to_process = list(self.pending_files)
                    self.pending_files.clear()

            if to_process:
                # Set status to processing immediately so UI catches it
                self.status["state"] = "processing"
                print(
                    f"{Colors.OKCYAN}[Worker] Quiet period detected. Processing {len(to_process)} files.{Colors.ENDC}")
                self.ingest(to_process)
    # ----------------------
    # Core RAG Methods
    # ----------------------
    def _get_file_hash(self, filepath):
        hasher = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                while chunk := f.read(8192): hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            print(f"{Colors.FAIL}Error hashing {filepath}: {e}{Colors.ENDC}")
            return None

    def _initial_sync(self):
        """Reconciles filesystem with vector store using SQLite for speed."""
        print(f"{Colors.OKCYAN}[Sync] Reconciling Store...{Colors.ENDC}")

        # NEW: Check if the vector store is actually empty
        # If Qdrant is empty, we must clear the librarian's notebook (SQLite)
        try:
            # We ask the engine for information about the collection
            collection_info = self.engine.client.get_collection(self.engine.collection_name)
            if collection_info.points_count == 0:
                print(f"{Colors.WARNING}[Sync] Vector store is empty! Forcing full re-sync...{Colors.ENDC}")
                conn = sqlite3.connect(self.manifest_db)
                conn.execute("DELETE FROM file_hashes")
                conn.commit()
                conn.close()
        except Exception as e:
            # If the collection doesn't even exist yet, that's fine too
            print(f"{Colors.OKCYAN}[Sync] Starting fresh...{Colors.ENDC}")

        SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.pptx', '.md'}

        db_state = self._get_stored_hashes()

        current_files = {
            str(p.absolute()): p for p in self.data_dir.rglob('*')
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        }

        # A. Cleanup deletions
        for path in list(db_state.keys()):
            if path not in current_files:
                self.remove_file(path)

        # B. Process new/modified
        to_process = []
        for p_str in current_files:
            f_hash = self._get_file_hash(p_str)
            if db_state.get(p_str) != f_hash:
                to_process.append(p_str)

        if to_process:
            print(f"{Colors.OKBLUE}[Sync] Found {len(to_process)} new/modified files.{Colors.ENDC}")
            self.ingest(to_process)
        else:
            print(f"{Colors.OKGREEN}[Sync] Filesystem is clean.{Colors.ENDC}")

    def aggregate_to_limit(self, raw_chunks: List[Any], token_limit: int = 512):
        standardized_docs = []
        current_text_block = []
        current_tokens = 0
        current_source = None

        for c in raw_chunks:
            source = c.metadata.get("source", "Unknown")
            text_content = getattr(c, 'page_content', getattr(c, 'text', ""))
            tokens = self.chunker.count_tokens(text_content)

            file_changed = (source != current_source and current_source is not None)
            limit_reached = (current_tokens + tokens > token_limit)

            if file_changed or limit_reached:
                if current_text_block:
                    standardized_docs.append(Document(
                        page_content="\n\n".join(current_text_block),
                        metadata={"source": current_source}
                    ))
                current_text_block, current_tokens = [], 0

            current_text_block.append(text_content)
            current_tokens += tokens
            current_source = source

        if current_text_block:
            standardized_docs.append(Document(
                page_content="\n\n".join(current_text_block),
                metadata={"source": current_source}
            ))
        return standardized_docs

    def ingest(self, new_files: List[str] = None):
        if not new_files: return

        # Update status to Processing
        self.status["state"] = "processing"
        self.status["total_files"] = len(new_files)
        self.status["progress"] = 0
        self.broadcast_status()  # <--- SHOUT: "I started!"

        try:
            with self.lock:
                valid_paths = [str(Path(f).absolute()) for f in new_files if Path(f).exists()]
                if not valid_paths:
                    self.status["state"] = "idle"  # Reset if no valid files
                    return

                # Purge phase
                for p_str in valid_paths:
                    self.engine.delete_by_source(p_str)

                print(f"{Colors.OKCYAN}[Ingest] Vectorizing {len(valid_paths)} files...{Colors.ENDC}")

                # Process phase
                # Note: If your DocumentProcessor supports single files,
                # we could loop here to update the progress bar per file.
                raw_docs = self.doc_processor.convert_to_documents(valid_paths, self.chunker)

                if raw_docs:
                    self.status["current_file"] = "Aggregating chunks..."
                    final_docs = self.aggregate_to_limit(raw_docs, token_limit=512)

                    self.status["current_file"] = "Storing in Database..."
                    self.engine.ingest_batches(final_docs, batch_size=32)

                    # Update manifest in SQLite
                    updates = {p: self._get_file_hash(p) for p in valid_paths}
                    self._update_manifest_batch(updates)
                    print(f"{Colors.OKGREEN}[Ingest] Success updated manifest in DB.{Colors.ENDC}")
                else:
                    print(f"{Colors.WARNING}[Ingest] No content extracted.{Colors.ENDC}")

        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
        finally:
            # IMPORTANT: This ensures the UI stops loading even if the code crashes
            self.status["state"] = "idle"
            self.status["progress"] = 100
            self.status["current_file"] = ""
            self.broadcast_status()  # <--- SHOUT: "I'm finished!"

    def remove_file(self, fpath):
        # Update status so the UI shows the "Purging" card
        self.status["state"] = "processing"
        self.status["current_file"] = f"Purging: {os.path.basename(fpath)}"
        self.status["progress"] = 50  # Start at 50% for visual effect
        self.broadcast_status()  # <--- Add this

        try:
            with self.lock:
                p_str = str(Path(fpath).absolute())
                self.engine.delete_by_source(p_str)
                # Remove from SQLite
                conn = sqlite3.connect(self.manifest_db)
                conn.execute("DELETE FROM file_hashes WHERE path = ?", (p_str,))
                conn.commit()
                conn.close()
                print(f"{Colors.WARNING}[Remove] Purged: {os.path.basename(fpath)}{Colors.ENDC}")
        except Exception as e:
            print(f"❌ Removal failed: {e}")
        finally:
            self.status["progress"] = 100
            self.status["state"] = "idle"
            self.status["current_file"] = ""
            self.broadcast_status()  # <--- Add this

    def _format_chat_history(self, chat_history):
        formatted_chat = ""
        for (curr_chatter, curr_chat) in chat_history:
            chat = f"{'User' if curr_chatter == 'You' else 'AI'}: {curr_chat}"
            formatted_chat += chat + "\n"
        return formatted_chat

    def ask(self, query, chat_history=None, stream=False):
        """
        Powerful History-Aware RAG Pipeline.
        Uses the LangChain History-Aware Retriever for context-aware searching.
        """
        # 1. Prepare history (Limit to 6 for VRAM/Token efficiency)
        history = chat_history if chat_history is not None else self.chat_history
        lc_history = []
        for role, text in history[-6:]:
            if role == "You":
                lc_history.append(HumanMessage(content=text))
            else:
                lc_history.append(AIMessage(content=text))

        print(f"\n{Colors.HEADER}{'=' * 20} RAG PIPELINE {'=' * 20}{Colors.ENDC}")

        try:
            # --- STEP 1: DEBUG VISIBILITY (The Standalone Query) ---
            # We run the rephraser once manually just so YOU can see it.
            # This is what Lora is actually going to search for.
            rephrase_chain = self.contextualize_q_prompt | self.llm
            standalone_query = rephrase_chain.invoke({
                "chat_history": lc_history,
                "input": query
            }).content

            print(
                f"{Colors.OKCYAN}{Colors.BOLD}🔍 Standalone Query:{Colors.ENDC} {Colors.OKCYAN}{standalone_query}{Colors.ENDC}")

            # --- STEP 2: POWERFUL RETRIEVAL ---
            # The History-Aware Retriever uses the same logic but returns the Docs.
            # It handles the rephrasing internally for the search.
            docs = self.history_aware_retriever.invoke({
                "input": query,
                "chat_history": lc_history
            })

            # Show the matches using the visual helper
            self._print_matches(docs)

            # --- STEP 3: FINAL SYNTHESIS ---
            # Prepare context for the final Llama call
            context_text = "\n\n".join([d.page_content for d in docs])

            # Use the QA prompt to generate the answer
            qa_chain = self.qa_prompt | self.llm
            ans = qa_chain.invoke({
                "context": context_text,
                "chat_history": lc_history,
                "input": query
            }).content

        except Exception as e:
            print(f"{Colors.FAIL}[Error] Pipeline failure: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()
            return "I hit a snag in my retrieval logic."

        # 4. Update internal memory if this isn't a temporary chat
        if chat_history is None:
            self.chat_history.append(("You", query))
            self.chat_history.append(("AI", ans))

        return ans

    def _print_matches(self, context_docs):
        """Helper to display sources in the terminal."""
        print(f"\n{Colors.HEADER}┌{'─' * 78}┐{Colors.ENDC}")
        for i, doc in enumerate(context_docs):
            src = os.path.basename(doc.metadata.get("source", "Unknown"))
            print(f"{Colors.OKCYAN}  Match {i + 1} {Colors.ENDC}| {src}")
            # Show a snippet of the context used
            snippet = doc.page_content[:80].replace('\n', ' ').strip()
            print(f"  {Colors.OKBLUE}↳{Colors.ENDC} {Colors.ITALIC}{snippet}...{Colors.ENDC}")
        print(f"{Colors.HEADER}└{'─' * 78}┘{Colors.ENDC}\n")


if __name__ == "__main__":
    rag = CollegeRAG()
    print(f"\n{Colors.OKGREEN}System initialized. Type your question below (exit to quit).{Colors.ENDC}")

    try:
        while True:
            question = input(f"{Colors.BOLD}You:{Colors.ENDC} ")
            if question.lower() in ['exit', 'quit']: break
            if not question.strip(): continue
            rag.ask(question)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Shutting down...{Colors.ENDC}")
        rag.observer.stop()
        rag.observer.join()