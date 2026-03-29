import os
import json
import hashlib
import threading
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Union
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
from .chunker import IntelligentChunker
from .colbert import ColBERTEngine
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.document_processor import DocumentProcessor
from utils.model_engine import get_llm
import torch


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


class RewriteLogger(StdOutCallbackHandler):
    def on_chain_end(self, outputs, **kwargs):
        tags = kwargs.get("tags", [])
        if "rewriter" in tags and isinstance(outputs, str):
            print(
                f"\n{Colors.WARNING}[Rewriter]{Colors.ENDC} Standalone Query: {Colors.BOLD}{outputs}{Colors.ENDC}"
            )


def format_docs_safely(docs):
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs if doc is not None)


class KeovilRAG:
    """Core RAG system for Keovil.

    Provides ingest() and query() methods without UI-specific features.
    For full app with auto-watching, use CollegeRAG instead.
    """

    def __init__(
        self,
        data_dir: str = None,
        storage_dir: str = None,
        collection_name: str = "keovil",
        auto_index: bool = True,
        top_k: int = 5,
        mode: str = "development",
    ):
        self.mode = mode

        if storage_dir:
            host_root = Path(storage_dir).absolute()
        else:
            if self.mode == "production":
                host_root = Path.home() / ".keovil_storage"
            elif self.mode == "sdk":
                host_root = Path.cwd() / "keovil_data"
            else:
                host_root = Path.home() / ".keovil_storage_dev"

        storage_env = os.getenv("STORAGE_BASE", str(host_root))
        self.base_storage = Path(storage_env).absolute()

        if data_dir:
            self.data_dir = Path(data_dir).resolve()
        else:
            self.data_dir = self.base_storage / "data"

        self.db_dir = self.base_storage / "database"
        self.manifest_db = self.db_dir / "manifest.db"
        self.collection_name = collection_name

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._init_manifest_db()

        print(
            f"{Colors.OKCYAN}🚀 Mode: {self.mode.upper()} | Root: {self.base_storage}{Colors.ENDC}"
        )
        print(f"{Colors.OKCYAN}📦 Collection: {self.collection_name}{Colors.ENDC}")

        self.top_k = top_k
        self.lock = threading.Lock()
        self.chat_history = []

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"{Colors.HEADER}Initializing ColBERT Engine on: {Colors.BOLD}{device}{Colors.ENDC}"
        )

        self.engine = ColBERTEngine(collection_name=self.collection_name, device=device)
        self.doc_processor = DocumentProcessor(use_gpu=torch.cuda.is_available())
        self.chunker = IntelligentChunker()

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

        self.rag_chain = (
            RunnableParallel(
                {
                    "context_docs": self.history_aware_retriever,
                    "input": lambda x: x["input"],
                    "chat_history": lambda x: x["chat_history"],
                    "time": lambda x: x["time"],
                }
            )
            | RunnablePassthrough.assign(
                context=lambda x: format_docs_safely(x["context_docs"])
            )
            | {
                "answer": qa_prompt | self.llm | StrOutputParser(),
                "docs": lambda x: x["context_docs"],
            }
        )

        if auto_index:
            self._initial_sync()

    def _init_manifest_db(self):
        conn = sqlite3.connect(self.manifest_db)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS file_hashes (path TEXT PRIMARY KEY, hash TEXT)"
        )
        conn.commit()
        conn.close()

    def _get_stored_hashes(self):
        conn = sqlite3.connect(self.manifest_db)
        cursor = conn.execute("SELECT path, hash FROM file_hashes")
        data = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return data

    def _update_manifest_batch(self, file_data: Dict[str, str]):
        conn = sqlite3.connect(self.manifest_db)
        conn.executemany(
            "INSERT OR REPLACE INTO file_hashes VALUES (?, ?)", list(file_data.items())
        )
        conn.commit()
        conn.close()

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
        print(f"{Colors.OKCYAN}[Sync] Reconciling Store...{Colors.ENDC}")

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

        current_files = {
            str(p.relative_to(self.base_storage)): p
            for p in self.data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        }

        for rel_path in list(db_state.keys()):
            if rel_path not in current_files:
                full_path = self.base_storage / rel_path
                self.remove_file(str(full_path))

        to_process = []
        for rel_path, abs_path in current_files.items():
            f_hash = self._get_file_hash(str(abs_path))
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
        """Index files into the vector store."""
        if not new_files:
            return

        try:
            with self.lock:
                valid_paths = [
                    str(Path(f).absolute()) for f in new_files if Path(f).exists()
                ]
                if not valid_paths:
                    return

                for p_str in valid_paths:
                    self.engine.delete_by_source(p_str)

                print(
                    f"{Colors.OKCYAN}[Ingest] Vectorizing {len(valid_paths)} files...{Colors.ENDC}"
                )

                raw_docs = self.doc_processor.convert_to_documents(
                    valid_paths, self.chunker
                )

                if raw_docs:
                    for doc in raw_docs:
                        abs_src = Path(doc.metadata["source"])
                        doc.metadata["source"] = str(
                            abs_src.relative_to(self.base_storage)
                        )

                    final_docs = self.aggregate_to_limit(raw_docs, token_limit=512)

                    self.engine.ingest_batches(final_docs, batch_size=32)

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

    def remove_file(self, fpath):
        """Remove a file from the index."""
        try:
            with self.lock:
                p_abs = Path(fpath).absolute()
                p_rel = str(p_abs.relative_to(self.base_storage))

                self.engine.delete_by_source(p_rel)
                conn = sqlite3.connect(self.manifest_db)
                conn.execute("DELETE FROM file_hashes WHERE path = ?", (p_rel,))
                conn.commit()
                conn.close()
                print(
                    f"{Colors.WARNING}[Remove] Purged: {os.path.basename(fpath)}{Colors.ENDC}"
                )
        except Exception as e:
            print(f"❌ Removal failed: {e}")

    def reindex(self):
        """Force re-index all files."""
        conn = sqlite3.connect(self.manifest_db)
        conn.execute("DELETE FROM file_hashes")
        conn.commit()
        conn.close()
        self._initial_sync()

    def query(self, question: str, chat_history: List = None) -> str:
        """Ask a question and get an answer."""
        history = chat_history if chat_history is not None else self.chat_history

        lc_history = []
        for role, text in history[-6:]:
            lc_history.append(
                HumanMessage(content=text) if role == "You" else AIMessage(content=text)
            )

        input_params = {
            "input": question,
            "chat_history": lc_history,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        result = self.rag_chain.invoke(
            input_params, config=RunnableConfig(callbacks=[RewriteLogger()])
        )

        answer = result["answer"]
        docs = result["docs"]

        if not docs:
            print("no docs found")

        if docs:
            self._print_match_box(docs)

        if chat_history is None:
            self.chat_history.append(("You", question))
            self.chat_history.append(("AI", answer))

        return answer

    def _print_match_box(self, docs):
        """Helper to print matching documents."""
        print(f"\n{Colors.HEADER}┌{'─' * 78}┐{Colors.ENDC}")
        for i, doc in enumerate(docs):
            src = os.path.basename(doc.metadata.get("source", "Unknown"))
            print(f"{Colors.OKCYAN}  Match {i + 1} {Colors.ENDC}| {src}")
            content_preview = doc.page_content[:150].replace("\n", " ")
            print(
                f"  {Colors.OKBLUE}↳{Colors.ENDC} {Colors.ITALIC}{content_preview}...{Colors.ENDC}"
            )
        print(f"{Colors.HEADER}└{'─' * 78}┘{Colors.ENDC}\n")
