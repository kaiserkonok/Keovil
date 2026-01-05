import os
import duckdb
import threading
import re
from pathlib import Path

import pandas as pd
from langchain_ollama import ChatOllama
from colorama import Fore, Style, init
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Added for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

init(autoreset=True)
console = Console()

# Thread-safe lock (same process)
SQL_THREAD_LOCK = threading.Lock()


# ==================================================
# SQL QUERY AGENT
# ==================================================
class SQLQueryAgent:
    def __init__(self, db_path, model_name="qwen2.5-coder:7b-instruct"):
        self.db_path = db_path
        self.model_name = model_name

        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0,
            num_ctx=16384,
        )

        # Install extensions ONCE (global)
        duckdb.execute("INSTALL excel")
        duckdb.execute("INSTALL spatial")

    def ask(self, query: str):
        with duckdb.connect(self.db_path) as con:
            con.execute("LOAD excel")
            con.execute("LOAD spatial")

            tables = con.execute("SHOW TABLES").fetchall()
            if not tables:
                return "Database is empty. Please upload files."

            schema_info = []
            for (t_name,) in tables:
                cols = con.execute(f"DESCRIBE {t_name}").fetchall()
                col_list = [f"{c[0]} ({c[1]})" for c in cols]
                schema_info.append(
                    f"Table '{t_name}': columns [{', '.join(col_list)}]"
                )

            schema = "\n".join(schema_info)

            # Updated prompt to force "Thought" visibility
            system_context = (
                "You are a Senior Data Analyst using DuckDB.\n"
                "1. Briefly explain your thought process.\n"
                "2. Provide the final SQL code inside a ```sql block.\n\n"
                f"Schema:\n{schema}"
            )

            try:
                # LOG: Start Thinking
                console.print(f"\n[bold yellow]🤔 AI is thinking...[/bold yellow]")

                response = self.llm.invoke(
                    f"{system_context}\n\nUser: {query}"
                ).content

                # LOG: Show the AI's reasoning panel
                thought_segment = response.split("```sql")[0].strip()
                console.print(Panel(thought_segment, title="AI Reasoning", border_style="blue"))

                sql_match = re.search(
                    r"```sql\n(.*?)\n```", response, re.DOTALL
                )

                if not sql_match:
                    return response

                sql = sql_match.group(1).strip()

                # LOG: Show the actual SQL being written
                console.print(Panel(Syntax(sql, "sql", theme="monokai"), title="Executing SQL", border_style="green"))

                print(f"{Fore.BLUE}🖥️ Running on GPU...{Style.RESET_ALL}")
                df = con.execute(sql).df()

                summary = self.llm.invoke(
                    f"Data preview: {df.head(5).to_dict()}\n"
                    "Summarize key findings."
                ).content

                return (
                    f"{summary}\n\n"
                    f"### 📊 Data\n{df.to_markdown(index=False)}"
                )

            except Exception as e:
                return f"⚠️ SQL Error: {e}"


# ==================================================
# FILE WATCHER
# ==================================================
class IngestionHandler(FileSystemEventHandler):
    def __init__(self, manager):
        self.manager = manager
        self._timer = None

    def process(self, event):
        if event.is_directory:
            return

        if event.src_path.lower().endswith(
                (".csv", ".xlsx", ".xls", ".parquet")
        ):
            if self._timer:
                self._timer.cancel()

            self._timer = threading.Timer(
                1.5, self.manager.sync_database
            )
            self._timer.start()

    def on_created(self, event):
        self.process(event)

    def on_modified(self, event):
        self.process(event)

    def on_deleted(self, event):
        self.process(event)


# ==================================================
# STRUCTURED DATA AGENT
# ==================================================
class StructuredDataAgent:
    def __init__(self, socketio=None, watch_dir=None):
        home = Path.home()
        base = home / ".k_rag_storage"

        self.socketio = socketio
        self.watch_dir = Path(
            watch_dir or base / "data"
        ).resolve()

        self.db_path = (base / "database" / "analyst.duckdb").resolve()
        self.state_db = (base / "database" / "sync_state.duckdb").resolve()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        # State DB (file tracking)
        with duckdb.connect(self.state_db) as sc:
            sc.execute(
                """
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
                """
            )

        self.agent = SQLQueryAgent(self.db_path)
        self.observer = Observer()
        self.is_syncing = False

    # ==================================================
    # SYNC LOGIC (With Visual Table Logs)
    # ==================================================
    def sync_database(self):
        if not SQL_THREAD_LOCK.acquire(blocking=False):
            return

        try:
            self.is_syncing = True

            # LOG: Visual Sync Table
            sync_table = Table(title="🔄 DuckDB Syncing", show_header=True, header_style="bold cyan")
            sync_table.add_column("File Name")
            sync_table.add_column("Status", justify="right")

            with (
                duckdb.connect(self.db_path) as con,
                duckdb.connect(self.state_db) as state_con,
            ):
                con.execute("LOAD excel")
                con.execute("LOAD spatial")

                valid_exts = (".csv", ".xlsx", ".xls", ".parquet")
                active_views = set()

                for root, _, files in os.walk(self.watch_dir):
                    for fname in files:
                        if fname.startswith("~$"):
                            continue
                        if not fname.lower().endswith(valid_exts):
                            continue

                        fp = Path(root, fname).resolve()
                        base_name = (
                            fp.stem.lower()
                            .replace(" ", "_")
                            .replace("-", "_")
                        )

                        mtime = fp.stat().st_mtime
                        size = fp.stat().st_size

                        prev = state_con.execute(
                            "SELECT mtime, size FROM file_history WHERE path = ?",
                            [str(fp)],
                        ).fetchone()

                        is_excel = fp.suffix.lower() in (".xlsx", ".xls")

                        if (
                                not is_excel
                                and prev
                                and prev[0] == mtime
                                and prev[1] == size
                        ):
                            active_views.add(base_name)
                            continue

                        # ---------------- CSV ----------------
                        if fp.suffix.lower() == ".csv":
                            con.execute(
                                f"""
                                CREATE OR REPLACE VIEW {base_name}
                                AS SELECT * FROM read_csv_auto('{fp}')
                                """
                            )
                            active_views.add(base_name)
                            sync_table.add_row(fname, "[bold green]SYNCED[/bold green]")

                        # ---------------- PARQUET ----------------
                        elif fp.suffix.lower() == ".parquet":
                            con.execute(
                                f"""
                                CREATE OR REPLACE VIEW {base_name}
                                AS SELECT * FROM parquet_scan('{fp}')
                                """
                            )
                            active_views.add(base_name)
                            sync_table.add_row(fname, "[bold green]SYNCED[/bold green]")

                        # ---------------- EXCEL (YOUR WORKING LOGIC) ----------------
                        elif is_excel:
                            try:
                                fn = (
                                    "read_xlsx"
                                    if fp.suffix.lower() == ".xlsx"
                                    else "read_xls"
                                )

                                xls = pd.ExcelFile(fp)
                                for sheet in xls.sheet_names:
                                    view_name = (
                                        f"{base_name}_"
                                        f"{sheet.lower().replace(' ', '_')}"
                                    )

                                    con.execute(
                                        f"""
                                        CREATE OR REPLACE VIEW {view_name}
                                        AS SELECT * FROM {fn}(
                                            '{fp}', sheet='{sheet}'
                                        )
                                        """
                                    )
                                    active_views.add(view_name)
                                sync_table.add_row(fname,
                                                   f"[bold green]INDEXED ({len(xls.sheet_names)} sheets)[/bold green]")

                            except Exception as e:
                                print(f"{Fore.YELLOW}⚠️ Excel Error ({fp.name}): {e}")

                        state_con.execute(
                            """
                            INSERT OR REPLACE INTO file_history
                            VALUES (?, ?, ?)
                            """,
                            [str(fp), mtime, size],
                        )

                # Cleanup removed views
                existing = {
                    r[0]
                    for r in con.execute("SHOW TABLES").fetchall()
                }

                for view in existing - active_views:
                    con.execute(f"DROP VIEW IF EXISTS {view}")
                    sync_table.add_row(view, "[bold red]DELETED[/bold red]")

            if sync_table.row_count > 0:
                console.print(sync_table)

        except Exception as e:
            print(f"{Fore.RED}✖ Sync Error: {e}{Style.RESET_ALL}")
        finally:
            self.is_syncing = False
            SQL_THREAD_LOCK.release()
            print(f"{Fore.GREEN}✅ Sync Complete.{Style.RESET_ALL}")

    # ==================================================
    # MONITOR
    # ==================================================
    def start_monitoring(self):
        self.sync_database()
        self.observer.schedule(
            IngestionHandler(self),
            str(self.watch_dir),
            recursive=True,
        )
        self.observer.start()
        console.print(Panel(f"Watching: {self.watch_dir}", title="Watcher Active", border_style="yellow"))

    # ==================================================
    # QUERY
    # ==================================================
    def query(self, text):
        return self.agent.ask(text)