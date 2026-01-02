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
        # Sanitize path for Windows/Linux compatibility
        self.db_uri = db_uri.replace('\\', '/')
        self.model_name = model_name
        self.engine = create_engine(self.db_uri)
        self.db = None
        self.llm = None
        self._lock = threading.Lock()
        self.refresh_agent()

    def refresh_agent(self):
        """Initializes the database and LLM connection."""
        with self._lock:
            try:
                self.db = SQLDatabase(self.engine)
                # Leveraging 16GB VRAM: High context window and 0 temperature for logic
                self.llm = ChatOllama(
                    model=self.model_name,
                    temperature=0,
                    num_ctx=16384,
                    timeout=180, # Increased timeout to prevent HTML error responses
                    verbose=True
                )
                print(f"{Fore.CYAN}🔄 SQL Engine Re-Initialized.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}✖ SQL Agent init failed: {e}{Style.RESET_ALL}")

    def ask(self, query: str):
        with self._lock:
            if not self.db: return "SQL system not initialized."
            try:
                # 1. Fetch Schema
                schema = self.db.get_table_info()

                # 2. Prompt for Reasoning & SQL
                prompt = f"""You are a SQLite Expert. Analyze the schema and answer the user.
                SCHEMA:
                {schema}

                USER QUESTION: {query}

                Respond in this EXACT format:
                Thought: <reasoning>
                SQL: ```sql
                <query>
                ```
                """

                raw_response = self.llm.invoke(prompt).content

                # --- Extract Thought & SQL ---
                thought_match = re.search(r"Thought:(.*?)SQL:", raw_response, re.DOTALL | re.IGNORECASE)
                if thought_match:
                    print(f"\n{Fore.MAGENTA}🤔 THOUGHT:{Style.RESET_ALL} {thought_match.group(1).strip()}")

                sql_match = re.search(r"```sql\n(.*?)\n```", raw_response, re.DOTALL)
                sql_query = sql_match.group(1).strip() if sql_match else None

                if not sql_query: return f"AI failed to generate SQL: {raw_response}"

                print(f"{Fore.BLUE}🖥️  EXECUTING SQL:{Style.RESET_ALL} {sql_query}")

                # 3. EXECUTION WITH HEADERS (The Fix)
                # We use SQLAlchemy directly to get column names so the AI doesn't get confused
                with self.engine.connect() as conn:
                    result_proxy = conn.execute(text(sql_query))
                    columns = result_proxy.keys()
                    rows = result_proxy.fetchall()

                    # Convert to a readable string with headers
                    data_with_headers = f"Columns: {list(columns)}\nData: {list(rows)}"
                    print(f"{Fore.CYAN}📊 DATA SECURED:{Style.RESET_ALL} {data_with_headers}")

                # 4. FINAL SUMMARY (With clear instructions not to hallucinate)
                summary_prompt = f"""
                USER QUESTION: {query}
                DATABASE RESULT:
                {data_with_headers}

                INSTRUCTION: 
                Summarize the data above to answer the user. 
                """

                final_answer = self.llm.invoke(summary_prompt).content
                print(f"{Fore.GREEN}🤖 FINAL ANSWER:{Style.RESET_ALL} {final_answer}\n")
                return final_answer

            except Exception as e:
                return f"⚠️ SQL Error: {str(e)}"


class IngestionHandler(FileSystemEventHandler):
    def __init__(self, manager):
        self.manager = manager
        self.valid_exts = (".csv", ".xlsx", ".xls", ".db", ".sqlite")

    def on_modified(self, event):
        if not event.is_directory and any(event.src_path.lower().endswith(x) for x in self.valid_exts):
            if "main.db" in event.src_path: return
            self.manager.sync_database(specific_file=event.src_path)

    def on_created(self, event):
        if not event.is_directory and any(event.src_path.lower().endswith(x) for x in self.valid_exts):
            self.manager.sync_database(specific_file=event.src_path)


class StructuredDataAgent:
    def __init__(self, db_path=None, watch_dir=None):
        home = str(Path.home())
        base_storage = os.path.join(home, ".k_rag_storage")
        if db_path is None: db_path = os.path.join(base_storage, "database", "main.db")
        if watch_dir is None: watch_dir = os.path.join(base_storage, "data")

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(watch_dir, exist_ok=True)

        self.db_uri = f"sqlite:///{os.path.abspath(db_path)}"
        self.watch_dir = os.path.abspath(watch_dir)
        self.agent = SQLQueryAgent(self.db_uri)
        self.observer = Observer()

    def sync_database(self, specific_file=None):
        """Processes files and updates the database."""
        files = [specific_file] if specific_file else [
            os.path.join(self.watch_dir, f) for f in os.listdir(self.watch_dir)
            if f.lower().endswith((".csv", ".xlsx", ".xls", ".db", ".sqlite"))
        ]

        for fp in files:
            t_name = os.path.splitext(os.path.basename(fp))[0].replace(" ", "_").replace("-", "_").lower()
            try:
                if fp.endswith(".csv"):
                    pd.read_csv(fp).to_sql(t_name, self.agent.engine, if_exists="replace", index=False)
                elif fp.endswith((".xlsx", ".xls")):
                    pd.read_excel(fp).to_sql(t_name, self.agent.engine, if_exists="replace", index=False)
                print(f"{Fore.BLUE}📦 Synced Table: {t_name}")
            except Exception as e:
                print(f"{Fore.RED}✖ Sync Error for {t_name}: {e}")

        # Always refresh the agent's schema view after syncing data
        self.agent.refresh_agent()

    # --- METHODS REQUIRED BY FLASK APP.PY ---

    def query(self, text_input: str):
        """Directly called by your Flask API endpoint."""
        return self.agent.ask(text_input)

    def start_monitoring(self):
        """Restores the missing method called by your startup script."""
        self.sync_database()
        event_handler = IngestionHandler(self)
        self.observer.schedule(event_handler, self.watch_dir, recursive=True)
        self.observer.start()
        print(f"{Fore.YELLOW}👀 SQL Data Lab monitoring: {self.watch_dir}{Style.RESET_ALL}")

    def stop(self):
        self.observer.stop()
        self.observer.join()