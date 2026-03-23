import os
import json
import hashlib
import threading
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Union
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, PatternMatchingEventHandler
from datetime import datetime
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_classic.chains.history_aware_retriever import (
    create_history_aware_retriever,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import rag_chunker
from utils.document_processor import DocumentProcessor
from utils.model_engine import get_llm
from colbert_engine import ColBERTEngine
import torch


# ----------------------
# ANSI Color Class for Debugging
# ----------------------
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    ITALIC = "\033[3m"


# ----------------------
# Watchdog handler
# ----------------------
class NewFileHandler(PatternMatchingEventHandler):
    def __init__(self, rag_instance):
        super().__init__(
            patterns=["*.txt", "*.pdf", "*.docx", "*.pptx", "*.md"],
            ignore_directories=True,
            case_sensitive=False,
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


class RewriteLogger(StdOutCallbackHandler):
    def on_chain_end(self, outputs, **kwargs):
        # Retrieve tags from the 'kwargs' which contain the run configuration
        tags = kwargs.get("tags", [])

        # Only print if this specific chain step was tagged as 'rewriter'
        if "rewriter" in tags and isinstance(outputs, str):
            print(
                f"\n{Colors.WARNING}[Rewriter]{Colors.ENDC} Standalone Query: {Colors.BOLD}{outputs}{Colors.ENDC}"
            )


def format_docs_safely(docs):
    # This is the fix for the 'NoneType' error
    if not docs:
        return ""
    # We ensure every document is converted to a string properly
    return "\n\n".join(doc.page_content for doc in docs if doc is not None)


# ----------------------
# College RAG System
# ----------------------
class CollegeRAG:
    def __init__(self, data_dir=None, top_k=5, socketio=None):
        # ---------------------------------------------------------
        # 1. TOTAL ISOLATION LOGIC (SSD Side)
        # ---------------------------------------------------------
        self.mode = os.getenv("APP_MODE", "development")

        # Define physical locations on your computer
        if self.mode == "production":
            host_root = Path.home() / ".keovil_storage"
            self.collection_name = "keovil"
        else:
            host_root = Path.home() / ".keovil_storage_dev"
            self.collection_name = "keovil_dev"

        # Support Docker override, otherwise use the host_root determined above
        storage_env = os.getenv("STORAGE_BASE", str(host_root))
        self.base_storage = Path(storage_env).absolute()

        # Define standardized sub-paths
        self.data_dir = self.base_storage / "data"
        self.db_dir = self.base_storage / "database"
        self.manifest_db = self.db_dir / "manifest.db"

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._init_manifest_db()

        print(
            f"{Colors.OKCYAN}🚀 Mode: {self.mode.upper()} | Root: {self.base_storage}{Colors.ENDC}"
        )
        print(f"{Colors.OKCYAN}📦 Collection: {self.collection_name}{Colors.ENDC}")

        # ---------------------------------------------------------
        # 2. STATE & ENGINE INITIALIZATION
        # ---------------------------------------------------------
        self.socketio = socketio
        self.status = {
            "state": "idle",
            "current_file": "",
            "progress": 0,
            "total_files": 0,
        }
        self.top_k = top_k
        self.lock = threading.Lock()
        self.pending_files = set()
        self.queue_lock = threading.Lock()
        self.chat_history = []

        # GPU Detection for your RTX 5060 Ti
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"{Colors.HEADER}Initializing ColBERT Engine on: {Colors.BOLD}{device}{Colors.ENDC}"
        )

        # Pass the isolated collection name here!
        self.engine = ColBERTEngine(collection_name=self.collection_name, device=device)
        self.doc_processor = DocumentProcessor(use_gpu=torch.cuda.is_available())
        self.chunker = rag_chunker.IntelligentChunker()

        # ---------------------------------------------------------
        # 3. LLM & RETRIEVER SETUP
        # ---------------------------------------------------------
        self.llm = get_llm()
        self.query_llm = get_llm()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone search query. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.history_aware_retriever = create_history_aware_retriever(
            self.query_llm,
            self.engine.as_retriever(search_kwargs={"k": self.top_k}),
            contextualize_q_prompt,
        ).with_config({"tags": ["rewriter"]})

        qa_system_prompt = (
            "You are Keo, a private AI Agent who works with unstructured data. "
            "Answer the question **only using the given context**. "
            "If names are misspelled, correct them using the context provided. "
            "Keep answers concise and friendly. Current time: {time}\n\n"
            "CONTEXT:\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        # This links your Rewriter + Retriever + QA Generation into one unit
        self.rag_chain = (
            RunnableParallel(
                {
                    "context_docs": self.history_aware_retriever,  # Keep the objects for the Match Box
                    "input": lambda x: x["input"],
                    "chat_history": lambda x: x["chat_history"],
                    "time": lambda x: x["time"],
                }
            )
            | RunnablePassthrough.assign(
                context=lambda x: format_docs_safely(
                    x["context_docs"]
                )  # Convert objects to string here
            )
            | {
                "answer": qa_prompt | self.llm | StrOutputParser(),
                "docs": lambda x: x[
                    "context_docs"
                ],  # Pass the docs through to the final output
            }
        )

        # ---------------------------------------------------------
        # 4. BACKGROUND WORKERS & WATCHDOG
        # ---------------------------------------------------------
        self._initial_sync()
        threading.Thread(target=self._batch_worker, daemon=True).start()

        self.observer = Observer()
        self.observer.schedule(NewFileHandler(self), str(self.data_dir), recursive=True)
        self.observer.start()
        print(
            f"{Colors.OKCYAN}👀 Monitoring {self.data_dir} with 5s batching...{Colors.ENDC}"
        )

    def get_status(self):
        """Returns the current state for the UI."""
        # Check if there are files waiting in the queue
        with self.queue_lock:
            pending_count = len(self.pending_files)

        if self.status["state"] == "idle" and pending_count > 0:
            return {
                "state": "waiting",
                "message": f"Waiting for quiet period... ({pending_count} files queued)",
            }

        return self.status

    def broadcast_status(self):
        if not self.socketio:
            return
        try:
            # We add a specific 'reason' so the frontend can distinguish
            # between vectorizing and just thinking.
            self.socketio.emit(
                "system_status",
                {
                    "is_busy": self.status["state"] != "idle",
                    "reason": self.status[
                        "state"
                    ],  # 'processing', 'waiting', or 'idle'
                    "sql_syncing": False,
                    "rag": self.get_status(),
                },
                namespace="/",
            )
        except Exception as e:
            print(f"Socket Error: {e}")

    # ----------------------
    # NEW: SQLite Database Logic
    # ----------------------
    def _init_manifest_db(self):
        """Creates the manifest table if it doesn't exist."""
        conn = sqlite3.connect(self.manifest_db)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS file_hashes (path TEXT PRIMARY KEY, hash TEXT)"
        )
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
        conn.executemany(
            "INSERT OR REPLACE INTO file_hashes VALUES (?, ?)", list(file_data.items())
        )
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
                    f"{Colors.OKCYAN}[Worker] Quiet period detected. Processing {len(to_process)} files.{Colors.ENDC}"
                )
                self.ingest(to_process)

    # ----------------------
    # Core RAG Methods
    # ----------------------
    def _get_file_hash(self, filepath):
        hasher = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            print(f"{Colors.FAIL}Error hashing {filepath}: {e}{Colors.ENDC}")
            return None

    def _initial_sync(self):
        """Reconciles filesystem with vector store using SQLite for speed."""
        print(f"{Colors.OKCYAN}[Sync] Reconciling Store...{Colors.ENDC}")

        # (Keep your Qdrant empty-check logic as is, it's good)
        try:
            collection_info = self.engine.client.get_collection(
                self.engine.collection_name
            )
            if collection_info.points_count == 0:
                print(
                    f"{Colors.WARNING}[Sync] Vector store is empty! Forcing full re-sync...{Colors.ENDC}"
                )
                conn = sqlite3.connect(self.manifest_db)
                conn.execute("DELETE FROM file_hashes")
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"{Colors.OKCYAN}[Sync] Starting fresh...{Colors.ENDC}")

        SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".pptx", ".md"}
        db_state = self._get_stored_hashes()

        # --- THE FIX: Use relative_to(self.base_storage) ---
        current_files = {
            str(p.relative_to(self.base_storage)): p
            for p in self.data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        }

        # A. Cleanup deletions
        for rel_path in list(db_state.keys()):
            if rel_path not in current_files:
                # We must reconstruct the full path so remove_file can find it on disk if needed
                full_path = self.base_storage / rel_path
                self.remove_file(str(full_path))

        # B. Process new/modified
        to_process = []
        for rel_path, abs_path in current_files.items():
            f_hash = self._get_file_hash(str(abs_path))
            # Now we compare the portable rel_path against the SQLite DB
            if db_state.get(rel_path) != f_hash:
                to_process.append(str(abs_path))

        if to_process:
            print(
                f"{Colors.OKBLUE}[Sync] Found {len(to_process)} new/modified files.{Colors.ENDC}"
            )
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
            text_content = getattr(c, "page_content", getattr(c, "text", ""))
            tokens = self.chunker.count_tokens(text_content)

            file_changed = source != current_source and current_source is not None
            limit_reached = current_tokens + tokens > token_limit

            if file_changed or limit_reached:
                if current_text_block:
                    standardized_docs.append(
                        Document(
                            page_content="\n\n".join(current_text_block),
                            metadata={"source": current_source},
                        )
                    )
                current_text_block, current_tokens = [], 0

            current_text_block.append(text_content)
            current_tokens += tokens
            current_source = source

        if current_text_block:
            standardized_docs.append(
                Document(
                    page_content="\n\n".join(current_text_block),
                    metadata={"source": current_source},
                )
            )
        return standardized_docs

    def ingest(self, new_files: List[str] = None):
        if not new_files:
            return

        # Update status to Processing
        self.status["state"] = "processing"
        self.status["total_files"] = len(new_files)
        self.status["progress"] = 0
        self.broadcast_status()  # <--- SHOUT: "I started!"

        try:
            with self.lock:
                valid_paths = [
                    str(Path(f).absolute()) for f in new_files if Path(f).exists()
                ]
                if not valid_paths:
                    self.status["state"] = "idle"  # Reset if no valid files
                    return

                # Purge phase
                for p_str in valid_paths:
                    self.engine.delete_by_source(p_str)

                print(
                    f"{Colors.OKCYAN}[Ingest] Vectorizing {len(valid_paths)} files...{Colors.ENDC}"
                )

                # Process phase
                # Note: If your DocumentProcessor supports single files,
                # we could loop here to update the progress bar per file.
                raw_docs = self.doc_processor.convert_to_documents(
                    valid_paths, self.chunker
                )

                if raw_docs:
                    self.status["current_file"] = "Aggregating chunks..."

                    for doc in raw_docs:
                        # Convert the absolute source to a relative one for the vector DB
                        abs_src = Path(doc.metadata["source"])
                        doc.metadata["source"] = str(
                            abs_src.relative_to(self.base_storage)
                        )

                    final_docs = self.aggregate_to_limit(raw_docs, token_limit=512)

                    self.status["current_file"] = "Storing in Database..."
                    self.engine.ingest_batches(final_docs, batch_size=32)

                    # Update manifest in SQLite
                    updates = {
                        str(
                            Path(p).relative_to(self.base_storage)
                        ): self._get_file_hash(p)
                        for p in valid_paths
                    }
                    self._update_manifest_batch(updates)
                    print(
                        f"{Colors.OKGREEN}[Ingest] Success updated manifest in DB.{Colors.ENDC}"
                    )
                else:
                    print(
                        f"{Colors.WARNING}[Ingest] No content extracted.{Colors.ENDC}"
                    )

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
                p_abs = Path(fpath).absolute()
                p_rel = str(p_abs.relative_to(self.base_storage))

                self.engine.delete_by_source(p_rel)  # Use p_rel
                # Remove from SQLite
                conn = sqlite3.connect(self.manifest_db)
                conn.execute("DELETE FROM file_hashes WHERE path = ?", (p_rel,))
                conn.commit()
                conn.close()
                print(
                    f"{Colors.WARNING}[Remove] Purged: {os.path.basename(fpath)}{Colors.ENDC}"
                )
        except Exception as e:
            print(f"❌ Removal failed: {e}")
        finally:
            self.status["progress"] = 100
            self.status["state"] = "idle"
            self.status["current_file"] = ""
            self.broadcast_status()  # <--- Add this

    def _format_chat_history(self, chat_history):
        formatted_chat = ""
        for curr_chatter, curr_chat in chat_history:
            chat = f"{'User' if curr_chatter == 'You' else 'AI'}: {curr_chat}"
            formatted_chat += chat + "\n"
        return formatted_chat

    def ask(self, query, chat_history=None, stream=False):
        print(f"DEBUG: Retriever Object Type: {type(self.history_aware_retriever)}")
        print(f"DEBUG: Engine Object Type: {type(self.engine)}")

        history = chat_history if chat_history is not None else self.chat_history

        lc_history = []
        for role, text in history[-6:]:
            lc_history.append(
                HumanMessage(content=text) if role == "You" else AIMessage(content=text)
            )

        input_params = {
            "input": query,
            "chat_history": lc_history,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # USE THE CHAIN HERE
        result = self.rag_chain.invoke(
            input_params, config=RunnableConfig(callbacks=[RewriteLogger()])
        )

        # The chain now returns a dictionary with 'answer' and 'docs'
        answer = result["answer"]
        docs = result["docs"]

        if not docs:
            print("no docs found")

        # Show your Match Box using the docs the chain found
        if docs:
            self._print_match_box(docs)

        if chat_history is None:
            self.chat_history.append(("You", query))
            self.chat_history.append(("AI", answer))

        return answer

    def _print_match_box(self, docs):
        """Helper to maintain your beautiful UI matching box."""
        print(f"\n{Colors.HEADER}┌{'─' * 78}┐{Colors.ENDC}")
        for i, doc in enumerate(docs):
            # Handle cases where source is a relative path string
            src = os.path.basename(doc.metadata.get("source", "Unknown"))
            print(f"{Colors.OKCYAN}  Match {i + 1} {Colors.ENDC}| {src}")
            content_preview = doc.page_content[:150].replace("\n", " ")
            print(
                f"  {Colors.OKBLUE}↳{Colors.ENDC} {Colors.ITALIC}{content_preview}...{Colors.ENDC}"
            )
        print(f"{Colors.HEADER}└{'─' * 78}┘{Colors.ENDC}\n")
