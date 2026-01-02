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
                schema = self.db.get_table_info()

                # --- 1. THE BRAIN: REASONING & PLAN ---
                # We combine your Senior Engineer logic with a conversational persona
                system_context = f"""
                You are a Senior Data Analyst.
                DATABASE SCHEMA:
                {schema}

                INSTRUCTIONS:
                - ALWAYS start with a 'Thought:' section explaining your logic.
                - If the request requires data, provide the SQLite query in a ```sql block.
                - If multiple tables are needed, separate queries with a semicolon.
                - If it's just a general chat (hello, etc.), respond naturally without SQL.
                """

                initial_response = self.llm.invoke(f"{system_context}\n\nUser Request: {query}").content

                # --- EXTRACT REASONING & SQL ---
                thought_process = ""
                thought_match = re.search(r"Thought:(.*?)SQL:", initial_response, re.DOTALL | re.IGNORECASE)
                if not thought_match:
                    thought_match = re.search(r"Thought:(.*)", initial_response, re.DOTALL | re.IGNORECASE)

                if thought_match:
                    thought_process = thought_match.group(1).strip()
                    print(f"\n{Fore.MAGENTA}🧠 THINKING: {thought_process}{Style.RESET_ALL}")

                sql_match = re.search(r"```sql\n(.*?)\n```", initial_response, re.DOTALL)

                # --- 2. THE ENGINE: BIG DATA EXECUTION ---
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
                                # We send a 'digest' of the data to the LLM so it doesn't choke on 1000s of rows
                                data_for_chat_summary.append({
                                    "total_rows": len(rows),
                                    "columns": cols,
                                    "sample_data": [dict(zip(cols, r)) for r in rows[:15]]
                                })

                                # We render the FULL data set into the Markdown table for the UI
                                md = "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n"
                                for r in rows:
                                    clean_row = [str(x).replace('|', '\\|') for x in r]
                                    md += "| " + " | ".join(clean_row) + " |\n"

                                # The container handles the "Thousands of Rows" scrolling
                                all_results_html.append(f'<div class="df-scroll-container">\n\n{md}\n\n</div>')

                # --- 3. THE VOICE: NATURAL CONVERSATION ---
                if not sql_match:
                    # Just return the natural chat if no database was needed
                    return initial_response

                # Final pass: The AI sees the real data and talks about it
                final_prompt = f"""
                User: {query}
                Your Logic: {thought_process}
                Data Found: {data_for_chat_summary}

                Based on the results above, give a natural, human-like response to the user.
                Explain what you found and any patterns you noticed. 
                If there are many rows, mention the total count.
                """

                final_chat = self.llm.invoke(final_prompt).content
                print(f"{Fore.GREEN}🤖 RESPONSE READY.{Style.RESET_ALL}")

                # --- 4. THE OUTPUT ---
                # We put the thought process in a quote block, then the chat, then the tables.
                output = f"{final_chat}\n\n"

                if all_results_html:
                    output += "### 📊 Data Records\n" + "\n\n".join(all_results_html)

                return output

            except Exception as e:
                print(f"{Fore.RED}⚠️ Error: {str(e)}{Style.RESET_ALL}")
                return f"I ran into an issue while processing that: {str(e)}"


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