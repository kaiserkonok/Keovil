import os
import threading
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from keovil.rag import KeovilRAG, Colors


# ----------------------
# Watchdog handler
# ----------------------
class NewFileHandler(PatternMatchingEventHandler):
    def __init__(self, rag_instance):
        super().__init__(
            patterns=["*.txt", "*.pdf", "*.docx", ".pptx", "*.md"],
            ignore_directories=True,
            case_sensitive=False,
        )
        self.rag = rag_instance

    def on_created(self, event):
        self.rag.queue_file(event.src_path)

    def on_modified(self, event):
        self.rag.queue_file(event.src_path)

    def on_deleted(self, event):
        self.rag.remove_file(event.src_path)


# ----------------------
# College RAG System (App Layer)
# ----------------------
class CollegeRAG(KeovilRAG):
    """Full RAG system with auto-watching and UI status updates.

    Inherits from KeovilRAG and adds:
    - Automatic file watching via watchdog
    - Background batch processing
    - SocketIO status broadcasts
    """

    def __init__(self, data_dir=None, storage_dir=None, top_k=5, socketio=None):
        self.socketio = socketio
        self.status = {
            "state": "idle",
            "current_file": "",
            "progress": 0,
            "total_files": 0,
        }
        self.pending_files = set()
        self.queue_lock = threading.Lock()

        mode = os.getenv("APP_MODE", "development")
        collection_name = "keovil" if mode == "production" else "keovil_dev"

        super().__init__(
            data_dir=data_dir,
            storage_dir=storage_dir,
            collection_name=collection_name,
            auto_index=True,
            top_k=top_k,
            mode=mode,
        )

        threading.Thread(target=self._batch_worker, daemon=True).start()

        self.observer = Observer()
        self.observer.schedule(NewFileHandler(self), str(self.data_dir), recursive=True)
        self.observer.start()
        print(
            f"{self.Colors.OKCYAN}👀 Monitoring {self.data_dir} with 5s batching...{self.Colors.ENDC}"
        )

    @property
    def Colors(self):
        from keovil.rag import Colors

        return Colors

    def get_status(self):
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
            self.socketio.emit(
                "system_status",
                {
                    "is_busy": self.status["state"] != "idle",
                    "reason": self.status["state"],
                    "sql_syncing": False,
                    "rag": self.get_status(),
                },
                namespace="/",
            )
        except Exception as e:
            print(f"Socket Error: {e}")

    def queue_file(self, path):
        with self.queue_lock:
            self.pending_files.add(str(Path(path).absolute()))

    def _batch_worker(self):
        while True:
            time.sleep(5)
            to_process = []
            with self.queue_lock:
                if self.pending_files:
                    to_process = list(self.pending_files)
                    self.pending_files.clear()

            if to_process:
                self.status["state"] = "processing"
                print(
                    f"{self.Colors.OKCYAN}[Worker] Quiet period detected. Processing {len(to_process)} files.{self.Colors.ENDC}"
                )
                self.ingest(to_process)

    def ingest(self, new_files=None):
        if not new_files:
            return

        self.status["state"] = "processing"
        self.status["total_files"] = len(new_files)
        self.status["progress"] = 0
        self.broadcast_status()

        try:
            super().ingest(new_files)
        finally:
            self.status["state"] = "idle"
            self.status["progress"] = 100
            self.status["current_file"] = ""
            self.broadcast_status()

    def remove_file(self, fpath):
        self.status["state"] = "processing"
        self.status["current_file"] = f"Purging: {os.path.basename(fpath)}"
        self.status["progress"] = 50
        self.broadcast_status()

        try:
            super().remove_file(fpath)
        finally:
            self.status["progress"] = 100
            self.status["state"] = "idle"
            self.status["current_file"] = ""
            self.broadcast_status()

    def ask(self, query, chat_history=None, stream=False):
        print(f"DEBUG: Retriever Object Type: {type(self.history_aware_retriever)}")
        print(f"DEBUG: Engine Object Type: {type(self.engine)}")

        history = chat_history if chat_history is not None else self.chat_history
        answer = super().query(query, history)

        if chat_history is None:
            self.chat_history.append(("You", query))
            self.chat_history.append(("AI", answer))

        return answer
