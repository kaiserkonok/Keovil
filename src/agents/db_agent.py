import os
import duckdb
import threading
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
        self.ext_dir = Path(db_path).parent / "duckdb_extensions"
        self.ext_dir.mkdir(exist_ok=True)
        self.model_name = model_name

        # Optimized for 16GB VRAM (Higher ctx + Flash Attention if supported)
        self.llm = OllamaLLM(
            model=self.model_name,
            temperature=0,
            num_ctx=16384,
            # Ensuring fast response by keeping the model focused
        )

        # Persistence of extensions in the specific storage folder
        with duckdb.connect(self.db_path) as con:
            con.execute(f"SET extension_directory = '{self.ext_dir}';")
            # Auto-installing inside the isolated folder
            con.execute("INSTALL excel; INSTALL spatial; INSTALL sqlite;")

    def ask(self, query: str):
        with duckdb.connect(self.db_path) as con:
            con.execute(f"SET extension_directory = '{self.ext_dir}';")
            con.execute("LOAD excel; LOAD spatial; LOAD sqlite;")

            # 1. Get ALL table names for the "Router" phase
            tables_raw = con.execute("SHOW TABLES").fetchall()
            if not tables_raw:
                return "Database is empty."

            all_table_names = [t[0] for t in tables_raw]

            # --- STEP A: IDENTIFY RELEVANT TABLES ---
            router_prompt = (
                "You are a database router. Given these tables, identify which are needed to answer the question.\n"
                f"TABLES: {', '.join(all_table_names)}\n"
                f"USER QUESTION: {query}\n"
                "Output ONLY the table names as a comma-separated list. If none, say NONE."
            )

            router_output = self.llm.invoke(router_prompt).strip()

            # --- FIX STARTS HERE ---
            # This finds all words/names in the output and ignores junk like "Sure!" or "```"
            found_words = re.findall(r'\b\w+\b', router_output)
            relevant_names = [name for name in found_words if name in all_table_names]
            # --- FIX ENDS HERE ---

            print(relevant_names)

            # Fallback: If router fails, give the first 3 tables to prevent total failure
            if not relevant_names:
                relevant_names = all_table_names

            # --- STEP B: GRAB SPECIFIC SCHEMAS (THE FIX) ---
            focused_schema = []
            for t_name in relevant_names:
                cols = con.execute(f"DESCRIBE {t_name}").fetchall()
                col_info = "\n      ".join([f"- {c[0]} ({c[1]})" for c in cols])
                focused_schema.append(f"TABLE: {t_name}\n      {col_info}")

            schema_context = "\n\n".join(focused_schema)

            print(schema_context)

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
                initial_response = self.llm.invoke(f"{system_context}\n\nUser: {query}")

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

                # 3. Execution (The World-Class Batch Update)
                print(f"{Fore.BLUE}🖥️ Running on GPU...{Style.RESET_ALL}")

                # Split SQL by semicolon, filter out empty strings
                sql_statements = [s.strip() for s in sql_raw.split(';') if s.strip()]

                all_results = []
                combined_df_html = ""

                for i, stmt in enumerate(sql_statements):
                    res_df = con.execute(stmt).df()

                    # Store a preview for the Voice Agent
                    # We use a key like 'Table_N' or try to parse the table name from the SQL
                    table_tag = re.search(r"FROM\s+(\w+)", stmt, re.I)
                    tag = table_tag.group(1) if table_tag else f"Result_{i + 1}"

                    all_results.append({
                        "source": tag,
                        "rows": len(res_df),
                        "data": res_df.head(5).to_dict()  # Small preview per statement
                    })

                    # Build the HTML output for the UI
                    table_md = res_df.to_markdown(index=False)
                    combined_df_html += f"#### Source table: {tag}\n<div class='df-scroll-container'>\n\n{table_md}\n\n</div>\n\n"

                # 4. Stage 2: The "Voice" Synthesis
                # We send the aggregate results to the voice
                voice_prompt = (
                    f"User Query: {query}\n"
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
            host_root = Path.home() / ".kevil_krag_storage"
            db_suffix = "prod"
        else:
            host_root = Path.home() / ".k_rag_storage"
            db_suffix = "dev"

        storage_env = os.getenv("STORAGE_BASE", str(host_root))
        self.base_storage = Path(storage_env).absolute()
        self.socketio = socketio

        # 2. ISOLATED PATHS
        self.watch_dir = Path(watch_dir or self.base_storage / "data").resolve()
        # Unique DB names for the specific mode
        self.db_path = (self.base_storage / "database" / f"analyst_{db_suffix}.duckdb").resolve()
        self.state_db = (self.base_storage / "database" / f"sync_state_{db_suffix}.duckdb").resolve()

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

    def sync_database(self):
        if not SQL_THREAD_LOCK.acquire(blocking=False): return

        if self.socketio:
            self.socketio.emit('system_status', {'sql_syncing': True})

        try:
            self.is_syncing = True
            sync_table = Table(title="🔄 Hierarchical Smart Sync", show_header=True, header_style="bold cyan")
            sync_table.add_column("Resource Path")
            sync_table.add_column("Status", justify="right")

            # Inside StructuredDataAgent.sync_database
            with duckdb.connect(str(self.db_path)) as con, duckdb.connect(str(self.state_db)) as state_con:
                # --- ADD THESE LINES FIRST ---
                con.execute(f"SET extension_directory = '{self.agent.ext_dir}';")
                con.execute("INSTALL excel; INSTALL spatial; INSTALL sqlite;")
                # -----------------------------

                con.execute("LOAD excel; LOAD spatial; INSTALL sqlite; LOAD sqlite;")

                valid_exts = (".csv", ".xlsx", ".xls", ".parquet", ".db", ".sqlite", ".sqlite3")
                active_views = set()

                for root, _, files in os.walk(self.watch_dir):
                    for fname in files:
                        if fname.startswith("~$") or not fname.lower().endswith(valid_exts): continue

                        fp = Path(root, fname).resolve()
                        mtime = fp.stat().st_mtime
                        size = fp.stat().st_size

                        rel_path = fp.relative_to(self.watch_dir)
                        base_identity = str(rel_path).lower()
                        for char in [os.sep, ".", " ", "-"]:
                            base_identity = base_identity.replace(char, "_")

                        # --- GUARD 1: Safe History Check ---
                        rel_path_str = str(fp.relative_to(self.watch_dir))

                        # Change the query to look for rel_path_str
                        res = state_con.execute("SELECT mtime, size FROM file_history WHERE path = ?",
                                                [rel_path_str]).fetchall()  # <--- UPDATED
                        prev = res[0] if res else None

                        is_excel = fp.suffix.lower() in (".xlsx", ".xls")
                        is_unchanged = prev is not None and len(prev) >= 2 and prev[0] == mtime and prev[1] == size

                        if not is_excel and is_unchanged:
                            active_views.add(base_identity)
                            continue

                        # Ingestion Logic
                        ext = fp.suffix.lower()
                        if ext == ".csv":
                            con.execute(
                                f"CREATE OR REPLACE VIEW {base_identity} AS SELECT * FROM read_csv_auto('{fp}')")
                            active_views.add(base_identity)
                            sync_table.add_row(str(rel_path), "[green]UPDATED[/green]")

                        elif ext == ".parquet":
                            con.execute(f"CREATE OR REPLACE VIEW {base_identity} AS SELECT * FROM parquet_scan('{fp}')")
                            active_views.add(base_identity)
                            sync_table.add_row(str(rel_path), "[green]UPDATED[/green]")

                        elif is_excel:
                            xls = pd.ExcelFile(fp)
                            for sheet in xls.sheet_names:
                                sheet_clean = sheet.lower().replace(' ', '_').replace('-', '_')
                                view_name = f"{base_identity}_{sheet_clean}"
                                con.execute(
                                    f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_xlsx('{fp}', sheet='{sheet}')")
                                active_views.add(view_name)
                            sync_table.add_row(str(rel_path), f"[cyan]INDEXED ({len(xls.sheet_names)} sheets)[/cyan]")


                        elif ext in (".db", ".sqlite", ".sqlite3"):

                            # 1. Attach temporarily to see what's inside

                            alias = f"temp_{base_identity}"

                            con.execute(f"ATTACH '{fp}' AS {alias} (TYPE SQLITE, READ_ONLY)")

                            # 2. Get all table names from the attached DB

                            sqlite_tables = con.execute(
                                f"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()

                            for (st_name,) in sqlite_tables:
                                view_name = f"{base_identity}_{st_name}"

                                # 3. Create a persistent view in MAIN that points to the file

                                con.execute(
                                    f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM sqlite_scan('{fp}', '{st_name}')")

                                active_views.add(view_name)

                            con.execute(f"DETACH {alias}")

                            sync_table.add_row(str(rel_path),
                                               f"[magenta]SQLITE INDEXED ({len(sqlite_tables)} tables)[/magenta]")

                        state_con.execute("INSERT OR REPLACE INTO file_history VALUES (?, ?, ?)",
                                  [rel_path_str, mtime, size])

                # --- GUARD 2: Safe Database Detach ---
                db_list = con.execute("PRAGMA show_databases").fetchall()
                for row in db_list:
                    if len(row) > 1:  # Ensure the tuple actually has a second element
                        db_alias = row[1]
                        if db_alias not in ('main', 'temp') and db_alias not in active_views:
                            con.execute(f"DETACH {db_alias}")
                            sync_table.add_row(db_alias, "[red]DETACHED DB[/red]")

                # --- GUARD 3: Safe View Drop ---
                table_list = con.execute("SHOW TABLES").fetchall()
                for row in table_list:
                    if len(row) > 0:  # Ensure the tuple has at least one element
                        view = row[0]
                        if view not in active_views:
                            con.execute(f"DROP VIEW IF EXISTS {view}")
                            sync_table.add_row(view, "[red]DELETED VIEW[/red]")

            if sync_table.row_count > 0:
                console.print(sync_table)

        except Exception as e:
            # This will now print exactly WHICH line failed if it happens again
            import traceback
            console.print(f"[bold red]✖ Sync Error: {e}[/bold red]")
            console.print(traceback.format_exc())
        finally:
            self.is_syncing = False
            SQL_THREAD_LOCK.release()
            if self.socketio:
                self.socketio.emit('system_status', {'sql_syncing': False})
            print(f"{Fore.GREEN}✅ Sync Complete.{Style.RESET_ALL}")

    def start_monitoring(self):
        self.sync_database()
        self.observer.schedule(IngestionHandler(self), str(self.watch_dir), recursive=True)
        self.observer.start()
        console.print(Panel(f"Watching: {self.watch_dir}", title="Watcher Active", border_style="yellow"))

    def query(self, text):
        return self.agent.ask(text)