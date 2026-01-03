import os
import pandas as pd
import sqlite3
import threading
import re
import time
from pathlib import Path
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from colorama import Fore, Style, init
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

init(autoreset=True)


class SQLQueryAgent:
    def __init__(self, db_uri: str, model_name: str = 'qwen2.5-coder:7b-instruct'):
        self.db_uri = db_uri.replace('\\', '/')
        self.model_name = model_name
        self.engine = create_engine(self.db_uri, pool_pre_ping=True)
        self.db = None
        self.llm = None
        self._lock = threading.Lock()
        self.refresh_agent()

    def refresh_agent(self):
        with self._lock:
            try:
                self.db = SQLDatabase(self.engine)
                self.llm = ChatOllama(
                    model=self.model_name,
                    temperature=0,
                    num_ctx=16384,
                    timeout=180,
                    verbose=True
                )
                print(f"{Fore.CYAN}🔄 SQL Engine & Schema Re-Initialized.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}✖ SQL Agent init failed: {e}{Style.RESET_ALL}")

    def ask(self, query: str):
        # ... (Keeping your original prompt logic exactly as requested)
        with self._lock:
            if not self.db: return "SQL system not initialized."
            try:
                schema = self.db.get_table_info()
                system_context = f"You are a Senior Data Analyst.\nSCHEMA:\n{schema}\n\nALWAYS Thought: logic\nSQL in ```sql blocks."
                initial_response = self.llm.invoke(f"{system_context}\n\nUser: {query}").content

                thought_process = ""
                thought_match = re.search(r"Thought:(.*?)SQL:", initial_response, re.DOTALL | re.IGNORECASE)
                if thought_match: thought_process = thought_match.group(1).strip()

                sql_match = re.search(r"```sql\n(.*?)\n```", initial_response, re.DOTALL)
                all_results_html = []
                data_for_chat_summary = []

                if sql_match:
                    with self.engine.connect() as conn:
                        for sql_query in [q.strip() for q in sql_match.group(1).split(';') if q.strip()]:
                            res = conn.execute(text(sql_query))
                            cols, rows = list(res.keys()), res.fetchall()
                            if rows:
                                data_for_chat_summary.append(
                                    {"total": len(rows), "sample": [dict(zip(cols, r)) for r in rows[:15]]})
                                md = "| " + " | ".join(cols) + " |\n| " + "--- | " * len(cols) + "\n"
                                for r in rows: md += "| " + " | ".join([str(x) for x in r]) + " |\n"
                                all_results_html.append(f'<div class="df-scroll-container">\n\n{md}\n\n</div>')

                if not sql_match: return initial_response
                final_chat = self.llm.invoke(
                    f"User: {query}\nData: {data_for_chat_summary}\nRespond naturally.").content
                return f"{final_chat}\n\n### 📊 Data Records\n" + "\n\n".join(all_results_html)
            except Exception as e:
                return f"Error: {e}"


class IngestionHandler(FileSystemEventHandler):
    """Only triggers on structural changes, ignoring file 'modifications' to prevent loops."""

    def __init__(self, manager):
        self.manager = manager
        self.valid_exts = (".csv", ".xlsx", ".xls")

    def process(self, event):
        if event.is_directory: return
        # Strictly ignore anything related to the database file
        if "main.db" in event.src_path: return

        if any(event.src_path.lower().endswith(x) for x in self.valid_exts):
            if self.manager.is_syncing: return

            if hasattr(self, '_timer') and self._timer: self._timer.cancel()
            self._timer = threading.Timer(2.0, self.manager.sync_database)
            self._timer.start()

    def on_created(self, event):
        self.process(event)

    def on_deleted(self, event):
        self.process(event)

    def on_moved(self, event):
        self.process(event)
    # on_modified is intentionally removed to kill the loop


class StructuredDataAgent:
    def __init__(self, db_path=None, watch_dir=None):
        home = str(Path.home())
        base = os.path.join(home, ".k_rag_storage")

        # Isolation: Ensure DB is NOT in the same folder being watched
        self.watch_dir = os.path.abspath(watch_dir or os.path.join(base, "data"))
        self.db_path = os.path.abspath(db_path or os.path.join(base, "database", "main.db"))

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.watch_dir, exist_ok=True)

        self.db_uri = f"sqlite:///{self.db_path}"
        self.agent = SQLQueryAgent(self.db_uri)
        self.observer = Observer()
        self.is_syncing = False

    def sync_database(self):
        if self.is_syncing: return
        self.is_syncing = True
        print(f"{Fore.YELLOW}🔄 Syncing Folder to Database Mirror...")

        try:
            active_tables = []
            for root, _, files in os.walk(self.watch_dir):
                for f_name in files:
                    if f_name.lower().endswith((".csv", ".xlsx", ".xls")):
                        fp = os.path.join(root, f_name)
                        # Remove symbols to prevent SQL token errors
                        t_name = os.path.splitext(f_name)[0].replace(" ", "_").replace("-", "_").replace("&",
                                                                                                         "and").lower()
                        active_tables.append(t_name)

                        try:
                            df = pd.read_csv(fp) if f_name.endswith(".csv") else pd.read_excel(fp)
                            df.to_sql(t_name, self.agent.engine, if_exists="replace", index=False)
                            print(f"{Fore.BLUE}📦 Updated Table: {t_name}")
                        except Exception as e:
                            print(f"{Fore.RED}✖ Error loading {f_name}: {e}")

            with self.agent.engine.begin() as conn:
                existing = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()
                for (et,) in existing:
                    if not et.startswith('sqlite_') and et not in active_tables:
                        conn.execute(text(f'DROP TABLE IF EXISTS "{et}"'))
                        print(f"{Fore.RED}🔥 Dropped orphaned table: {et}")
                if active_tables: conn.execute(text("VACUUM"))

            self.agent.refresh_agent()
            print(f"{Fore.GREEN}✅ Sync Complete. {len(active_tables)} tables active.{Style.RESET_ALL}")
        finally:
            time.sleep(1)  # Let the file system cool down
            self.is_syncing = False

    def query(self, text_input: str):
        return self.agent.ask(text_input)

    def start_monitoring(self):
        self.sync_database()
        self.observer.schedule(IngestionHandler(self), self.watch_dir, recursive=True)
        self.observer.start()
        print(f"{Fore.YELLOW}👀 Monitoring Directory: {self.watch_dir}")

    def stop(self):
        self.observer.stop()
        self.observer.join()