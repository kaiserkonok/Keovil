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
from flask_socketio import emit

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
                # 1. Kill all existing connections
                self.engine.dispose()

                # 2. CLEAR THE METADATA CACHE (The important part!)
                # This wipes SQLAlchemy's memory of what tables exist
                if hasattr(self, 'db') and self.db:
                    self.db._metadata.clear()

                # 3. Re-initialize the DB wrapper
                self.db = SQLDatabase(self.engine)

                # 4. Re-initialize the LLM
                self.llm = ChatOllama(
                    model=self.model_name,
                    temperature=0,
                    num_ctx=16384,
                    timeout=180
                )
                print(f"{Fore.CYAN}🔄 CACHE PURGED: Database and AI memory are now 100% clean.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}✖ SQL Agent init failed: {e}{Style.RESET_ALL}")

    def ask(self, query: str):
        with self._lock:
            if not self.db: return "SQL system not initialized."
            try:
                # FORCE RELOAD SCHEMA BEFORE EACH QUESTION
                self.db._metadata.clear()  # Wipe SQLAlchemy's internal cache
                self.db = SQLDatabase(self.engine)  # Re-bind
                schema = self.db.get_table_info()  # Get the REAL tables only

                if not schema.strip():
                    return "The database is currently empty. Please add some files."

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


SQL_GLOBAL_LOCK = threading.Lock()

class StructuredDataAgent:
    def __init__(self, socketio=None, db_path=None, watch_dir=None):
        home = str(Path.home())
        base = os.path.join(home, ".k_rag_storage")

        self.socketio = socketio  # <--- Store the microphone!

        self.watch_dir = os.path.abspath(watch_dir or os.path.join(base, "data"))
        self.db_path = os.path.abspath(os.path.join(base, "database", "main.db"))
        self.state_db_path = os.path.abspath(os.path.join(base, "database", "sync_state.db"))

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.watch_dir, exist_ok=True)

        self.db_uri = f"sqlite:///{self.db_path}"
        self.agent = SQLQueryAgent(self.db_uri)
        self.state_engine = create_engine(f"sqlite:///{self.state_db_path}")
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
        # 1. PREVENT CONCURRENCY COLLISIONS
        if not SQL_GLOBAL_LOCK.acquire(blocking=False):
            print(f"{Fore.YELLOW}⚠️  Sync blocked: Another instance is already working.{Style.RESET_ALL}")
            return

        try:
            self.is_syncing = True
            self.broadcast_status()  # <--- ADD THIS (Shout: "I am starting!")
            print(f"{Fore.YELLOW}🔄 Syncing Folder (Smart Differential Sync)...")

            active_tables = []
            chunk_size = 100000
            valid_extensions = (".csv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3")

            # --- 2. SCAN PHASE: Determine what should exist ---
            for root, _, files in os.walk(self.watch_dir):
                for f_name in files:
                    # Filter out temporary Excel files and hidden system files
                    if f_name.startswith("~$") or f_name.startswith(".") or f_name in ["main.db", "sync_state.db"]:
                        continue

                    if not f_name.lower().endswith(valid_extensions):
                        continue

                    fp = os.path.join(root, f_name)
                    t_base = os.path.splitext(f_name)[0].replace(" ", "_").replace("-", "_").lower()

                    if f_name.lower().endswith((".csv", ".xlsx", ".xls")):
                        active_tables.append(t_base)
                    else:  # Database files
                        try:
                            with sqlite3.connect(fp) as temp_conn:
                                tbls = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", temp_conn)
                                active_tables.extend([t for t in tbls['name'] if not t.startswith('sqlite_')])
                        except:
                            pass

            # --- 3. IMPORT PHASE: Load new/changed files ---
            for root, _, files in os.walk(self.watch_dir):
                for f_name in files:
                    if f_name.startswith("~$") or f_name.startswith(".") or not self._should_sync(
                            os.path.join(root, f_name)):
                        continue

                    fp = os.path.join(root, f_name)
                    ext = f_name.lower()
                    t_name = os.path.splitext(f_name)[0].replace(" ", "_").replace("-", "_").lower()

                    # Handle Database Files
                    if ext.endswith((".db", ".sqlite", ".sqlite3")):
                        try:
                            with sqlite3.connect(fp) as src_conn:
                                tbl_names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", src_conn)
                                for t in [tn for tn in tbl_names['name'] if not tn.startswith('sqlite_')]:
                                    with self.agent.engine.begin() as conn:
                                        conn.execute(text(f'DROP TABLE IF EXISTS "{t}"'))
                                    df_iter = pd.read_sql(f'SELECT * FROM "{t}"', src_conn, chunksize=chunk_size)
                                    for chunk in df_iter:
                                        chunk.to_sql(t, self.agent.engine, if_exists='append', index=False)
                            self._update_metadata(fp)
                        except Exception as e:
                            print(f"{Fore.RED}✖ DB Error {f_name}: {e}")

                    # Handle CSV Files
                    elif ext.endswith(".csv"):
                        try:
                            df = pd.read_csv(fp, low_memory=False, encoding_errors='replace')
                            df.to_sql(t_name, self.agent.engine, if_exists='replace', index=False)
                            self._update_metadata(fp)
                        except Exception as e:
                            print(f"{Fore.RED}✖ CSV Error {f_name}: {e}")

                    # Handle Excel Files (FORCED ENGINES)
                    elif ext.endswith((".xlsx", ".xls")):
                        try:
                            # Use openpyxl for xlsx, xlrd for xls. Explicitly.
                            engine_to_use = "openpyxl" if ext.endswith(".xlsx") else "xlrd"
                            df = pd.read_excel(fp, engine=engine_to_use)
                            df.to_sql(t_name, self.agent.engine, if_exists='replace', index=False)
                            self._update_metadata(fp)
                        except Exception as e:
                            print(f"{Fore.RED}✖ Excel Error {f_name}: {e}")

            # --- 4. CLEANUP PHASE: Remove orphaned tables ---
            with self.agent.engine.connect() as conn:
                res = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                existing_on_disk = [r[0] for r in res.fetchall() if not r[0].startswith('sqlite_')]

                for et in existing_on_disk:
                    if et not in active_tables:
                        with self.agent.engine.begin() as drop_conn:
                            drop_conn.execute(text(f'DROP TABLE IF EXISTS "{et}"'))
                        print(f"{Fore.RED}🔥 Dropped orphaned table: {et}")

        except Exception as e:
            print(f"{Fore.RED}✖ Sync Critical Error: {e}{Style.RESET_ALL}")

        finally:
            # --- 5. REFRESH PHASE: Always update the AI Brain ---
            print(f"{Fore.CYAN}🧼 Finalizing sync and refreshing AI context...{Style.RESET_ALL}")
            self.agent.refresh_agent()
            self.is_syncing = False
            self.broadcast_status()  # <--- ADD THIS (Shout: "I am finished!")
            SQL_GLOBAL_LOCK.release()  # Open the door for the next sync
            print(f"{Fore.GREEN}✅ Sync Complete. System is ready.{Style.RESET_ALL}")

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

    def broadcast_status(self):
        """Standardized status broadcast using the injected socket instance."""
        if not self.socketio:
            return

        try:
            # NO MORE IMPORTS FROM APP!
            self.socketio.emit('system_status', {
                "is_busy": self.is_syncing,
                "sql_syncing": self.is_syncing,
                "rag": {"state": "idle"}
            })
            status_text = 'Busy' if self.is_syncing else 'Idle'
            print(f"{Fore.MAGENTA}📡 [SQL Socket] Status: {status_text}{Style.RESET_ALL}")
        except Exception as e:
            # Quietly fail if socket is disconnected
            pass