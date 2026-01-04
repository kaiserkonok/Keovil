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
from tqdm import tqdm

init(autoreset=True)

class SQLQueryAgent:
    def __init__(self, db_uri: str, model_name: str = 'qwen2.5-coder:7b-instruct'):
        self.db_uri = db_uri.replace('\\', '/')
        self.model_name = model_name
        self.engine = create_engine(self.db_uri, pool_pre_ping=True, connect_args={"timeout": 30})
        self.db = None
        self.llm = None
        self._lock = threading.Lock()
        self.refresh_agent()

    def refresh_agent(self):
        with self._lock:
            try:
                # IMPORTANT: Clear the old connection pool
                self.engine.dispose()
                self.db = SQLDatabase(self.engine)
                # Optimized for your RTX 5060 Ti 16GB VRAM
                self.llm = ChatOllama(
                    model=self.model_name,
                    temperature=0,
                    num_ctx=16384,
                    timeout=180
                )
                print(f"{Fore.CYAN}🔄 SQL Engine & Schema Re-Initialized.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}✖ SQL Agent init failed: {e}{Style.RESET_ALL}")

    def ask(self, query: str):
        with self._lock:
            if not self.db: return "SQL system not initialized."
            try:
                schema = self.db.get_table_info()

                # --- YOUR ORIGINAL BRAIN PROMPT ---
                system_context = f"""
                You are a Senior Data Analyst.
                DATABASE SCHEMA:
                {schema}

                INSTRUCTIONS:
                - ALWAYS start with a 'Thought:' section.
                - If the request requires data, provide the SQLite query in a ```sql block.
                - **CRITICAL**: Do NOT use "Meta-Commands" or "Dot-Commands" (e.g., .tables, .schema). 
                - Use the **Standard System Catalog** instead: 
                  - To list tables: `SELECT name FROM sqlite_master WHERE type='table';`
                  - To see table structure: `PRAGMA table_info('table_name');`
                - If multiple tables are needed, separate queries with a semicolon.
                """

                initial_response = self.llm.invoke(f"{system_context}\n\nUser Request: {query}").content

                thought_process = ""
                thought_match = re.search(r"Thought:(.*?)SQL:", initial_response, re.DOTALL | re.IGNORECASE)
                if not thought_match:
                    thought_match = re.search(r"Thought:(.*)", initial_response, re.DOTALL | re.IGNORECASE)

                if thought_match:
                    thought_process = thought_match.group(1).strip()
                    print(f"\n{Fore.MAGENTA}🧠 THINKING: {thought_process}{Style.RESET_ALL}")

                sql_match = re.search(r"```sql\n(.*?)\n```", initial_response, re.DOTALL)

                all_results_html = []
                data_for_chat_summary = []

                if sql_match:
                    sql_raw = sql_match.group(1).strip()
                    queries = [q.strip() for q in sql_raw.split(';') if q.strip()]

                    with self.engine.connect() as conn:
                        for sql_query in queries:
                            print(f"{Fore.BLUE}🖥️  EXECUTING:{Style.RESET_ALL} {sql_query}")
                            res = conn.execute(text(sql_query))
                            cols = list(res.keys())
                            rows = res.fetchall()

                            if rows:
                                data_for_chat_summary.append({
                                    "total_rows": len(rows),
                                    "columns": cols,
                                    "sample_data": [dict(zip(cols, r)) for r in rows[:15]]
                                })

                                md = "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n"
                                for r in rows:
                                    clean_row = [str(x).replace('|', '\\|') for x in r]
                                    md += "| " + " | ".join(clean_row) + " |\n"

                                all_results_html.append(f'<div class="df-scroll-container">\n\n{md}\n\n</div>')

                if not sql_match:
                    return initial_response

                # --- YOUR ORIGINAL VOICE PROMPT ---
                final_prompt = f"""
                User: {query}
                Your Logic: {thought_process}
                Data Found: {data_for_chat_summary}

                Based on the results above, give a natural, human-like response to the user.
                Explain what you found and any patterns you noticed. Don't expose the sql query.
                If there are many rows, mention the total count.
                """

                final_chat = self.llm.invoke(final_prompt).content
                print(f"{Fore.GREEN}🤖 RESPONSE READY.{Style.RESET_ALL}")

                # --- YOUR ORIGINAL OUTPUT FORMAT ---
                output = f"{final_chat}\n\n"
                if all_results_html:
                    output += "### 📊 Data Records\n" + "\n\n".join(all_results_html)

                return output

            except Exception as e:
                print(f"{Fore.RED}⚠️ Error in ask(): {str(e)}{Style.RESET_ALL}")
                return f"I ran into an issue while processing that: {str(e)}"


class IngestionHandler(FileSystemEventHandler):
    def __init__(self, manager):
        self.manager = manager
        self.valid_exts = (".csv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3")
        self._timer = None

    def process(self, event):
        if event.is_directory: return
        fname = os.path.basename(event.src_path)
        if fname in ["main.db", "sync_state.db", "main.db-journal", "main.db-wal"]: return

        if any(event.src_path.lower().endswith(x) for x in self.valid_exts):
            if hasattr(self, '_timer') and self._timer: self._timer.cancel()
            self._timer = threading.Timer(2.0, self.manager.sync_database)
            self._timer.start()

    def on_created(self, event): self.process(event)
    def on_deleted(self, event): self.process(event)
    def on_moved(self, event): self.process(event)


class StructuredDataAgent:
    def __init__(self, db_path=None, watch_dir=None):
        home = str(Path.home())
        base = os.path.join(home, ".k_rag_storage")

        self.watch_dir = os.path.abspath(watch_dir or os.path.join(base, "data"))
        self.db_path = os.path.abspath(os.path.join(base, "database", "main.db"))
        self.state_db_path = os.path.abspath(os.path.join(base, "database", "sync_state.db"))

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.watch_dir, exist_ok=True)

        self.db_uri = f"sqlite:///{self.db_path}"
        self.agent = SQLQueryAgent(self.db_uri)
        self.state_engine = create_engine(f"sqlite:///{self.state_db_path}")
        self.sync_lock = threading.Lock()
        self.observer = Observer()
        self.is_syncing = False
        self._init_metadata()

    def _init_metadata(self):
        with self.state_engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS file_history (
                    file_path TEXT PRIMARY KEY,
                    last_modified REAL,
                    file_size INTEGER
                )
            """))

    def _should_sync(self, file_path):
        if not os.path.exists(file_path): return False
        try:
            mtime = os.path.getmtime(file_path)
            fsize = os.path.getsize(file_path)
        except OSError: return False

        with self.state_engine.connect() as conn:
            result = conn.execute(
                text("SELECT last_modified, file_size FROM file_history WHERE file_path = :path"),
                {"path": file_path}
            ).fetchone()
            if result:
                if result[0] == mtime and result[1] == fsize: return False
            return True

    def _update_metadata(self, file_path):
        mtime = os.path.getmtime(file_path)
        fsize = os.path.getsize(file_path)
        with self.state_engine.begin() as conn:
            conn.execute(text("""
                INSERT OR REPLACE INTO file_history (file_path, last_modified, file_size)
                VALUES (:path, :mtime, :fsize)
            """), {"path": file_path, "mtime": mtime, "fsize": fsize})

    def sync_database(self):
        # The lock MUST be the very first thing.
        # This prevents 'Thread 15' and 'Thread 16' from running at the same time.
        if not self.sync_lock.acquire(blocking=False):
            return  # If a sync is already running, just ignore this trigger

        try:
            if self.is_syncing: return
            self.is_syncing = True

            print(f"{Fore.YELLOW}🔄 Syncing Folder (Smart Differential Sync)...")
            active_tables = []
            chunk_size = 100000

            # 1. SCAN PHASE (Gather what should exist)
            for root, _, files in os.walk(self.watch_dir):
                for f_name in files:
                    fp = os.path.join(root, f_name)
                    if f_name in ["main.db", "sync_state.db"] or not any(f_name.lower().endswith(x) for x in
                                                                         [".csv", ".xlsx", ".xls", ".db", ".sqlite",
                                                                          ".sqlite3"]):
                        continue

                    t_base = os.path.splitext(f_name)[0].replace(" ", "_").replace("-", "_").lower()

                    # Add to active list
                    if f_name.lower().endswith((".csv", ".xlsx", ".xls")):
                        active_tables.append(t_base)
                    else:  # It's a DB file
                        try:
                            with sqlite3.connect(fp) as temp_conn:
                                tbls = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", temp_conn)
                                active_tables.extend([t for t in tbls['name'] if not t.startswith('sqlite_')])
                        except:
                            pass

            # 2. SYNC PHASE (Only process if metadata says it's new/changed)
            for root, _, files in os.walk(self.watch_dir):
                for f_name in files:
                    fp = os.path.join(root, f_name)
                    ext = f_name.lower()
                    if f_name in ["main.db", "sync_state.db"] or not any(
                            ext.endswith(x) for x in [".csv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3"]):
                        continue

                    if not self._should_sync(fp):
                        continue

                    t_name = os.path.splitext(f_name)[0].replace(" ", "_").replace("-", "_").lower()

                    # Handle DB Cloning
                    if ext.endswith((".db", ".sqlite", ".sqlite3")):
                        try:
                            with sqlite3.connect(fp) as src_conn:
                                tbl_names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", src_conn)
                                for t in [tn for tn in tbl_names['name'] if not tn.startswith('sqlite_')]:
                                    # Use a fresh connection for the DROP to avoid long-held locks
                                    with self.agent.engine.begin() as conn:
                                        conn.execute(text(f'DROP TABLE IF EXISTS "{t}"'))

                                    df_iter = pd.read_sql(f'SELECT * FROM "{t}"', src_conn, chunksize=chunk_size)
                                    for chunk in df_iter:
                                        chunk.to_sql(t, self.agent.engine, if_exists='append', index=False)
                            self._update_metadata(fp)
                        except Exception as e:
                            print(f"{Fore.RED}✖ DB Error {f_name}: {e}")

                    # Handle CSV/Excel (Simplified)
                    elif ext.endswith(".csv"):
                        try:
                            df = pd.read_csv(fp, low_memory=False, encoding_errors='replace')
                            df.to_sql(t_name, self.agent.engine, if_exists='replace', index=False)
                            self._update_metadata(fp)
                        except Exception as e:
                            print(f"{Fore.RED}✖ CSV Error: {e}")

                    elif ext.endswith((".xlsx", ".xls")):
                        try:
                            df = pd.read_excel(fp)
                            df.to_sql(t_name, self.agent.engine, if_exists='replace', index=False)
                            self._update_metadata(fp)
                        except Exception as e:
                            print(f"{Fore.RED}✖ Excel Error: {e}")

            # 3. CLEANUP PHASE (Drop tables that no longer have files)
            with self.agent.engine.connect() as conn:
                # Get existing tables without keeping a long transaction
                res = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                existing_on_disk = [r[0] for r in res.fetchall() if not r[0].startswith('sqlite_')]

                for et in existing_on_disk:
                    if et not in active_tables:
                        # Drop inside its own tiny transaction
                        with self.agent.engine.begin() as drop_conn:
                            drop_conn.execute(text(f'DROP TABLE IF EXISTS "{et}"'))
                        print(f"{Fore.RED}🔥 Dropped orphaned table: {et}")

            self.agent.refresh_agent()
            print(f"{Fore.GREEN}✅ Sync Complete.{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}✖ Sync Critical Error: {e}{Style.RESET_ALL}")
        finally:
            self.is_syncing = False
            self.sync_lock.release()  # ALWAYS release the lock

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