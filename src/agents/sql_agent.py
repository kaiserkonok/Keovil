import os
import duckdb
import threading
import re
import time
from pathlib import Path

import pandas as pd
from langchain_ollama import ChatOllama
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
        self.model_name = model_name

        # Optimized for RTX 5060 Ti 16GB VRAM
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0,
            num_ctx=16384,
        )

        # Install extensions ONCE
        duckdb.execute("INSTALL excel")
        duckdb.execute("INSTALL spatial")

    def ask(self, query: str):
        with duckdb.connect(self.db_path) as con:
            con.execute("LOAD excel")
            con.execute("LOAD spatial")

            # 1. Fetch Schema
            tables = con.execute("SHOW TABLES").fetchall()
            if not tables:
                return "Database is empty. Please upload files to the data folder."

            schema_info = []
            for (t_name,) in tables:
                cols = con.execute(f"DESCRIBE {t_name}").fetchall()
                # Use a vertical list for columns to make them distinct
                col_details = "\n      ".join([f"- {c[0]} ({c[1]})" for c in cols])
                schema_info.append(f"TABLE: {t_name}\n      {col_details}")

            schema = "\n\n".join(schema_info)

            # 2. Stage 1: The "Thought" and "SQL" Generation
            system_context = (
                "You are a Senior Data Analyst using DuckDB.\n"
                f"DATABASE SCHEMA:\n{schema}\n\n"
                "INSTRUCTIONS:\n"
                "- ALWAYS start with 'Thought: <reasoning>'. Use this to determine if a query is actually needed.\n"
                "- GREETINGS: If the user says 'hello' or provides a general greeting, respond politely WITHOUT a SQL block.\n"
                "- SQL USAGE: ONLY provide a ```sql block if the user's request requires data or metadata from the database.\n"
                "- NAMESPACING: Tables use a path-based naming convention (e.g., 'folder_subfolder_file_ext'). "
                "Use these prefixes to distinguish between different departments or versions of data.\n"
                "- METADATA: For questions about table/column counts, use system views:\n"
                "  * Count tables: `SELECT count(*) FROM information_schema.tables WHERE table_schema='main';`\n"
                "  * List tables: `SHOW TABLES;`\n"
                "- STANDARDS: Do NOT use dot-commands. Use standard SQL."
            )

            try:
                console.print(f"\n[bold yellow]🤔 AI is thinking...[/bold yellow]")
                initial_response = self.llm.invoke(f"{system_context}\n\nUser: {query}").content

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

                final_chat = self.llm.invoke(voice_prompt).content
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
        if event.src_path.lower().endswith((".csv", ".xlsx", ".xls", ".parquet")):
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
        home = Path.home()
        base = home / ".k_rag_storage"

        self.socketio = socketio
        self.watch_dir = Path(watch_dir or base / "data").resolve()
        self.db_path = (base / "database" / "analyst.duckdb").resolve()
        self.state_db = (base / "database" / "sync_state.duckdb").resolve()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        # Initialize State DB (Differential Tracking)
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

        self.agent = SQLQueryAgent(self.db_path)
        self.observer = Observer()
        self.is_syncing = False

    def sync_database(self):
        if not SQL_THREAD_LOCK.acquire(blocking=False): return

        # Trigger the amber pulse in UI
        if self.socketio:
            self.socketio.emit('system_status', {'sql_syncing': True})

        try:
            self.is_syncing = True
            sync_table = Table(title="🔄 Hierarchical Smart Sync", show_header=True, header_style="bold cyan")
            sync_table.add_column("Resource Path")
            sync_table.add_column("Status", justify="right")

            with duckdb.connect(str(self.db_path)) as con, duckdb.connect(str(self.state_db)) as state_con:
                con.execute("LOAD excel; LOAD spatial;")

                valid_exts = (".csv", ".xlsx", ".xls", ".parquet")
                active_views = set()

                for root, _, files in os.walk(self.watch_dir):
                    for fname in files:
                        if fname.startswith("~$") or not fname.lower().endswith(valid_exts): continue

                        fp = Path(root, fname).resolve()
                        mtime = fp.stat().st_mtime
                        size = fp.stat().st_size

                        # 1. World-Class Naming: Relative Path + Extension
                        # Example: 'finance/2024/sales.csv' -> 'finance_2024_sales_csv'
                        rel_path = fp.relative_to(self.watch_dir)
                        base_identity = str(rel_path).lower()
                        for char in [os.sep, ".", " ", "-"]:
                            base_identity = base_identity.replace(char, "_")

                        # Check differential state
                        prev = state_con.execute("SELECT mtime, size FROM file_history WHERE path = ?",
                                                 [str(fp)]).fetchone()
                        is_excel = fp.suffix.lower() in (".xlsx", ".xls")

                        # Skip if unchanged (Excel always re-indexed for sheet safety)
                        if not is_excel and prev and prev[0] == mtime and prev[1] == size:
                            # Still need to track this as active even if we don't re-ingest
                            if not is_excel: active_views.add(base_identity)
                            continue

                        # 2. Ingestion with Unique Names
                        if fp.suffix.lower() == ".csv":
                            con.execute(
                                f"CREATE OR REPLACE VIEW {base_identity} AS SELECT * FROM read_csv_auto('{fp}')")
                            active_views.add(base_identity)
                            sync_table.add_row(str(rel_path), "[green]UPDATED[/green]")

                        elif fp.suffix.lower() == ".parquet":
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

                        state_con.execute("INSERT OR REPLACE INTO file_history VALUES (?, ?, ?)",
                                          [str(fp), mtime, size])

                # 3. Precise Cleanup (Orphans)
                existing_tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
                for view in existing_tables - active_views:
                    con.execute(f"DROP VIEW IF EXISTS {view}")
                    sync_table.add_row(view, "[red]DELETED[/red]")

            if sync_table.row_count > 0:
                console.print(sync_table)

        except Exception as e:
            print(f"{Fore.RED}✖ Sync Error: {e}")
        finally:
            self.is_syncing = False
            SQL_THREAD_LOCK.release()

            # Turn off the pulse
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