import os
import duckdb
import threading
import hashlib
import re
import time
from pathlib import Path

import pandas as pd
from langchain_ollama import OllamaLLM
from colorama import Fore, Style, init
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

init(autoreset=True)
console = Console()

# Thread-safe lock
SQL_THREAD_LOCK = threading.Lock()


# ==================================================
# SQL QUERY AGENT
# ==================================================
class SQLQueryAgent:
    def __init__(self, db_path, model_name="qwen2.5-coder:7b-instruct"):
        self.db_path = str(db_path)
        # Isolate extensions within the database folder to prevent cross-contamination
        self.ext_dir = Path(db_path).parent / "sys_modules"
        self.ext_dir.mkdir(exist_ok=True)
        self.model_name = model_name

        # Optimized for 16GB VRAM (Higher ctx + Flash Attention if supported)
        self.llm = OllamaLLM(
            model=self.model_name,
            temperature=0,
        )

        # Persistence of extensions in the specific storage folder
        with duckdb.connect(self.db_path) as con:
            con.execute(f"SET extension_directory = '{self.ext_dir}';")
            # Auto-installing inside the isolated folder
            con.execute("INSTALL excel; INSTALL spatial; INSTALL sqlite;")

    def ask(self, query: str, chat_history: list = None):
        """
        History-aware SQL Agent.
        chat_history: list of {'role': 'user'|'assistant', 'content': str}
        """
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        # --- STEP 0: CONTEXTUALIZATION (STANDALONE QUERY GENERATION) ---
        # This prevents follow-up questions from breaking the SQL generator.
        formatted_history = []
        if chat_history:
            for msg in chat_history[-10:]:  # Last 10 messages for context
                if msg['role'] == 'user':
                    formatted_history.append(HumanMessage(content=msg['content']))
                else:
                    # Strip the heavy HTML table data from history to keep LLM focused
                    clean_content = msg['content'].split('### 📊 Data Records')[0].strip()
                    formatted_history.append(AIMessage(content=clean_content))

        if formatted_history:
            context_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a query refiner. Given the chat history and a follow-up question, rephrase the question into a STANDALONE query that includes all necessary context. Output ONLY the rephrased question."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            try:
                standalone_query = (context_prompt | self.llm).invoke({
                    "history": formatted_history,
                    "input": query
                }).strip()
                console.print(f"[dim cyan]Refined Query:[/dim cyan] [italic]{standalone_query}[/italic]")
            except Exception as e:
                standalone_query = query
        else:
            standalone_query = query

        # --- DATABASE EXECUTION ---
        with duckdb.connect(self.db_path) as con:
            con.execute(f"SET extension_directory = '{self.ext_dir}';")
            con.execute("LOAD excel; LOAD spatial; LOAD sqlite;")

            # 1. Get ALL table names for the "Router" phase
            tables_raw = con.execute("SHOW TABLES").fetchall()
            if not tables_raw:
                return "Database is empty."

            all_table_names = [t[0] for t in tables_raw]

            # --- STEP A: IDENTIFY RELEVANT TABLES ---
            # We use the standalone_query here for better routing accuracy
            router_prompt = (
                "You are a database router. Given these tables, identify which are needed to answer the question.\n"
                f"TABLES: {', '.join(all_table_names)}\n"
                f"USER QUESTION: {standalone_query}\n"
                "Output ONLY the table names as a comma-separated list. If none, say NONE."
            )

            router_output = self.llm.invoke(router_prompt).strip()

            found_words = re.findall(r'\b\w+\b', router_output)
            relevant_names = [name for name in found_words if name in all_table_names]

            if not relevant_names:
                relevant_names = all_table_names[:3]

            # --- STEP B: GRAB SPECIFIC SCHEMAS ---
            focused_schema = []
            for t_name in relevant_names:
                cols = con.execute(f"DESCRIBE {t_name}").fetchall()
                col_info = "\n      ".join([f"- {c[0]} ({c[1]})" for c in cols])
                focused_schema.append(f"TABLE: {t_name}\n      {col_info}")

            schema_context = "\n\n".join(focused_schema)

            # --- STEP C: GENERATE SQL WITH FOCUSED CONTEXT ---
            system_context = (
                "You are a Senior Data Analyst using DuckDB.\n"
                "Use the following FOCUSED SCHEMA to write your query:\n"
                f"{schema_context}\n\n"
                "INSTRUCTIONS:\n"
                "- Start with 'Thought: <reasoning>'.\n"
                "- Output exact SQL in a ```sql block.\n"
                "- If no data is needed (greeting), just reply normally."
            )

            try:
                console.print(f"\n[bold yellow]🤔 AI is thinking...[/bold yellow]")
                # We provide the original query context but emphasize the standalone_query intent
                initial_response = self.llm.invoke(f"{system_context}\n\nUser: {standalone_query}")

                # Extract Thought
                thought_match = re.search(r"Thought:(.*?)(?=```sql|$)", initial_response, re.DOTALL | re.IGNORECASE)
                thought_process = thought_match.group(1).strip() if thought_match else "Analyzing..."
                console.print(Panel(thought_process, title="AI Reasoning", border_style="blue"))

                # Extract SQL
                sql_match = re.search(r"```sql\n(.*?)\n```", initial_response, re.DOTALL)
                if not sql_match:
                    return initial_response

                sql_raw = sql_match.group(1).strip()
                console.print(
                    Panel(Syntax(sql_raw, "sql", theme="monokai"), title="Executing SQL", border_style="green"))

                # 3. Execution (Running on your RTX 5060 Ti via DuckDB)
                print(f"{Fore.BLUE}🖥️ Running on GPU...{Style.RESET_ALL}")

                sql_statements = [s.strip() for s in sql_raw.split(';') if s.strip()]

                all_results = []
                combined_df_html = ""

                for i, stmt in enumerate(sql_statements):
                    res_df = con.execute(stmt).df()

                    table_tag = re.search(r"FROM\s+(\w+)", stmt, re.I)
                    tag = table_tag.group(1) if table_tag else f"Result_{i + 1}"

                    all_results.append({
                        "source": tag,
                        "rows": len(res_df),
                        "data": res_df.head(5).to_dict()
                    })

                    table_md = res_df.to_markdown(index=False)
                    combined_df_html += f"#### Source table: {tag}\n<div class='df-scroll-container'>\n\n{table_md}\n\n</div>\n\n"

                # 4. Stage 2: The "Voice" Synthesis
                voice_prompt = (
                    f"User Query: {query}\n"
                    f"Refined Intent: {standalone_query}\n"
                    f"Context/Logic: {thought_process}\n"
                    f"Execution Results: {all_results}\n\n"
                    "As a Senior Data Analyst, interpret the multiple result sets provided to answer the User Query. "
                    "1. Synthesize the data from all tables into one cohesive answer.\n"
                    "2. If the user asked for a comparison or 'top rows from all', summarize the findings.\n"
                    "3. Do NOT mention technical SQL details."
                )

                final_chat = self.llm.invoke(voice_prompt)
                console.print(f"{Fore.GREEN}🤖 Response Ready.{Style.RESET_ALL}")

                # 5. Format for UI
                output = (
                    f"{final_chat}\n\n"
                    f"### 📊 Data Records\n"
                    f"{combined_df_html}"
                )

                return output

            except Exception as e:
                console.print(f"[bold red]⚠️ SQL Error: {e}[/bold red]")
                return f"I ran into an issue executing the SQL: {str(e)}"


# ==================================================
# FILE WATCHER (Ingestion Logic)
# ==================================================
class IngestionHandler(FileSystemEventHandler):
    def __init__(self, manager):
        self.manager = manager
        self._timer = None

    def process(self, event):
        if event.is_directory: return
        # Cleaned up extensions and added .sqlite
        valid_extensions = (".csv", ".xlsx", ".xls", ".parquet", ".db", ".sqlite", ".sqlite3")
        if event.src_path.lower().endswith(valid_extensions):
            if self._timer: self._timer.cancel()
            self._timer = threading.Timer(1.5, self.manager.sync_database)
            self._timer.start()

    def on_created(self, event):
        self.process(event)

    def on_modified(self, event):
        self.process(event)

    def on_deleted(self, event):
        self.process(event)


# ==================================================
# STRUCTURED DATA AGENT (The Core Manager)
# ==================================================
class StructuredDataAgent:
    def __init__(self, socketio=None, watch_dir=None):
        # 1. TOTAL ISOLATION LOGIC
        self.mode = os.getenv("APP_MODE", "development")

        if self.mode == "production":
            host_root = Path.home() / ".keovil_storage"
            db_suffix = "prod"
        else:
            host_root = Path.home() / ".keovil_storage_dev"
            db_suffix = "dev"

        storage_env = os.getenv("STORAGE_BASE", str(host_root))
        self.base_storage = Path(storage_env).absolute()
        self.socketio = socketio
        self.handler = IngestionHandler(self)

        # 2. ISOLATED PATHS
        self.watch_dir = Path(watch_dir or self.base_storage / "data").resolve()
        # Unique DB names for the specific mode
        self.db_path = (self.base_storage / "database" / f"cache_{db_suffix}.bin").resolve()
        self.state_db = (self.base_storage / "database" / f"state_{db_suffix}.bin").resolve()

        # Ensure everything is ready
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        # 3. INITIALIZE STATE DB
        with duckdb.connect(str(self.state_db)) as sc:
            sc.execute("""
                       CREATE TABLE IF NOT EXISTS file_history
                       (
                           path
                           TEXT
                           PRIMARY
                           KEY,
                           mtime
                           DOUBLE,
                           size
                           BIGINT
                       )
                       """)

        # 4. INITIALIZE AGENT (Passing the mode-specific path)
        self.agent = SQLQueryAgent(self.db_path)
        self.observer = Observer()
        self.is_syncing = False

        console.print(f"[bold cyan]🚀 SQL Agent Mode: {self.mode.upper()}[/bold cyan]")

    def _get_unique_name(self, fp: Path) -> str:
        """Creates a unique, SQL-safe name. Understandable for LLM via stems."""
        rel_path = fp.relative_to(self.watch_dir)
        # Unique 4-char hash based on the folder path
        path_hash = hashlib.md5(str(rel_path).encode()).hexdigest()[:4]
        # Clean the filename (e.g., "Sales Data 2026" -> "sales_data_2026")
        clean_stem = re.sub(r'[^a-zA-Z0-9]', '_', fp.stem).lower()
        return f"v_{clean_stem}_{path_hash}"

    def _needs_update(self, state_con, fp: Path) -> bool:
        """Checks if the file on disk differs from our last sync."""
        rel_path = str(fp.relative_to(self.watch_dir))
        stats = fp.stat()
        res = state_con.execute(
            "SELECT mtime, size FROM file_history WHERE path = ?", [rel_path]
        ).fetchone()
        return res is None or res[0] != stats.st_mtime or res[1] != stats.st_size

    def _cleanup_orphans(self, con, active_views):
        """The Janitor: Removes DuckDB views that no longer have physical files."""
        current_tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        for table in current_tables:
            if table not in active_views:
                con.execute(f"DROP VIEW IF EXISTS {table}")
                console.print(f"[red]🗑️ Dropped stale view: {table}[/red]")

    def _track_multi_table_names(self, con, fp, v_name, ext, active_views):
        """Purely tracks names for UNCHANGED files so they aren't deleted."""
        if ext in (".db", ".sqlite", ".sqlite3"):
            tables = con.execute(f"SELECT name FROM sqlite_scan('{fp}', 'sqlite_master') WHERE type='table'").fetchall()
            for (t_name,) in tables:
                active_views.add(f"{v_name}_{t_name.lower()}")
        elif ext in (".xlsx", ".xls"):
            # Use pandas just to get sheet names without loading data
            xls = pd.ExcelFile(fp)
            for sheet in xls.sheet_names:
                sheet_clean = re.sub(r'[^a-zA-Z0-9]', '_', sheet).lower()
                active_views.add(f"{v_name}_{sheet_clean}")

    def sync_database(self):
        if not SQL_THREAD_LOCK.acquire(blocking=False): return

        # Initial Signal: System is busy, progress is 0
        if self.socketio:
            self.socketio.emit('system_status', {
                'sql_syncing': True,
                'reason': 'processing',
                'rag': {'state': 'processing', 'current_file': 'Starting Sync...', 'progress': 0}
            })

        try:
            self.is_syncing = True
            active_views = set()
            HANDLERS = {".csv": "read_csv_auto", ".parquet": "parquet_scan"}

            # Get list of all potential files first to calculate progress
            all_files = []
            for root, _, files in os.walk(self.watch_dir):
                for f in files:
                    if not f.startswith("~$") and f.lower().endswith(
                            (".csv", ".parquet", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3")):
                        all_files.append(Path(root, f).resolve())

            total_files = len(all_files)

            with duckdb.connect(str(self.db_path)) as con, \
                    duckdb.connect(str(self.state_db)) as state_con:

                con.execute(f"SET extension_directory = '{self.agent.ext_dir}';")
                con.execute("INSTALL excel; INSTALL spatial; INSTALL sqlite; LOAD excel; LOAD spatial; LOAD sqlite;")

                for idx, fp in enumerate(all_files):
                    v_name = self._get_unique_name(fp)
                    ext = fp.suffix.lower()

                    # Update UI for every file processed
                    if self.socketio:
                        progress = int(((idx + 1) / total_files) * 100)
                        self.socketio.emit('system_status', {
                            'sql_syncing': True,
                            'reason': 'processing',
                            'rag': {
                                'state': 'processing',
                                'current_file': f"Indexing: {fp.name}",
                                'progress': progress
                            }
                        })

                    # --- FAST PATH: Unchanged Files ---
                    if not self._needs_update(state_con, fp):
                        if ext in (".db", ".sqlite", ".sqlite3", ".xlsx", ".xls"):
                            self._track_multi_table_names(con, fp, v_name, ext, active_views)
                        else:
                            active_views.add(v_name)
                        continue

                    # --- SLOW PATH: Ingestion ---
                    if ext in HANDLERS:
                        con.execute(f"CREATE OR REPLACE VIEW {v_name} AS SELECT * FROM {HANDLERS[ext]}('{fp}')")
                        active_views.add(v_name)
                    elif ext in (".xlsx", ".xls"):
                        xls = pd.ExcelFile(fp)
                        for sheet in xls.sheet_names:
                            sheet_clean = re.sub(r'[^a-zA-Z0-9]', '_', sheet).lower()
                            sub_v = f"{v_name}_{sheet_clean}"
                            con.execute(
                                f"CREATE OR REPLACE VIEW {sub_v} AS SELECT * FROM read_xlsx('{fp}', sheet='{sheet}')")
                            active_views.add(sub_v)
                    elif ext in (".db", ".sqlite", ".sqlite3"):
                        sqlite_tables = con.execute(
                            f"SELECT name FROM sqlite_scan('{fp}', 'sqlite_master') WHERE type='table'").fetchall()
                        for (t_name,) in sqlite_tables:
                            sub_v = f"{v_name}_{t_name.lower()}"
                            con.execute(
                                f"CREATE OR REPLACE VIEW {sub_v} AS SELECT * FROM sqlite_scan('{fp}', '{t_name}')")
                            active_views.add(sub_v)

                    # Record Success in State DB
                    stats = fp.stat()
                    state_con.execute(
                        "INSERT OR REPLACE INTO file_history VALUES (?, ?, ?)",
                        [str(fp.relative_to(self.watch_dir)), stats.st_mtime, stats.st_size]
                    )

                self._cleanup_orphans(con, active_views)

        except Exception as e:
            console.print(f"[bold red]✖ Sync Error:[/bold red] {e}")
            if self.socketio:
                self.socketio.emit('system_error', {'message': f"Sync failed: {str(e)}"})
        finally:
            self.is_syncing = False
            SQL_THREAD_LOCK.release()

            # Final Signal: Idle state
            if self.socketio:
                self.socketio.emit('system_status', {
                    'sql_syncing': False,
                    'reason': 'idle',
                    'rag': {'state': 'idle', 'progress': 100}
                })
            console.print(f"[bold green]✅ Sync Finished.[/bold green]")

    def start_monitoring(self):
        self.sync_database()
        self.observer.schedule(self.handler, str(self.watch_dir), recursive=True)
        self.observer.start()

    def query(self, text, chat_history=None):
        return self.agent.ask(text, chat_history)