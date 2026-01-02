import os
import pandas as pd
import sqlite3
import threading
import time
from pathlib import Path
from sqlalchemy import create_engine, inspect, text
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from colorama import Fore, Style, init
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

init(autoreset=True)


class SQLQueryAgent:
    def __init__(self, db_uri: str, model_name: str = 'qwen2.5:7b-instruct'):
        self.db_uri = db_uri
        self.model_name = model_name
        self.engine = create_engine(db_uri)
        self.agent_executor = None
        self._lock = threading.Lock()
        self.refresh_agent()

    def refresh_agent(self):
        with self._lock:
            try:
                db = SQLDatabase(self.engine)
                llm = ChatOllama(model=self.model_name, temperature=0.1)
                toolkit = SQLDatabaseToolkit(db=db, llm=llm)
                self.agent_executor = create_sql_agent(
                    toolkit=toolkit, llm=llm, agent_type='tool-calling', verbose=True
                )
                print(f"{Fore.CYAN}🔄 Agent schema refreshed.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}✖ Failed to refresh agent: {e}{Style.RESET_ALL}")

    def ask(self, query: str):
        with self._lock:
            if not self.agent_executor: return "Agent not ready."
            try:
                response = self.agent_executor.invoke({"input": query})
                return response["output"] if isinstance(response, dict) else str(response)
            except Exception as e:
                return f"Error: {str(e)}"


class IngestionHandler(FileSystemEventHandler):
    def __init__(self, system_manager):
        self.manager = system_manager
        self.valid_extensions = (".csv", ".xlsx", ".xls", ".db", ".sqlite")

    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith(self.valid_extensions):
            # Check if it's a system file
            if any(x in event.src_path for x in ["main.db", "manifest.db", "chat_history.db"]):
                return
            # ONLY sync the specific file that changed
            self.manager.sync_database(specific_file=event.src_path)

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(self.valid_extensions):
            self.manager.sync_database(specific_file=event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.lower().endswith(self.valid_extensions):
            self.manager.sync_database()  # Full sync on delete to remove orphans


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

    def _get_table_name(self, file_path):
        return os.path.splitext(os.path.basename(file_path))[0].replace(" ", "_").replace("-", "_").lower()

    def sync_database(self, specific_file=None):
        """Mirrors the database. If specific_file is provided, only that file is processed."""

        # IF we are saving a .txt or .pdf, this is never called now.
        # IF we provide a specific file, we ONLY process that one.

        if specific_file:
            print(f"{Fore.YELLOW}⚡ Incremental Sync: {os.path.basename(specific_file)}")
            files_to_process = [specific_file]
        else:
            print(f"\n{Fore.YELLOW}⚙️  Full Database Re-Sync...")
            main_db_abs = os.path.abspath(self.agent.engine.url.database)
            files_to_process = []
            valid_exts = (".csv", ".xlsx", ".xls", ".db", ".sqlite")
            for root, _, files in os.walk(self.watch_dir):
                for f in files:
                    fp = os.path.abspath(os.path.join(root, f))
                    if fp == main_db_abs or any(x in f for x in ["manifest.db", "chat_history.db"]): continue
                    if f.lower().endswith(valid_exts): files_to_process.append(fp)

        expected_tables = set() if not specific_file else None  # Only cleanup on full sync

        for file_path in files_to_process:
            base_t_name = self._get_table_name(file_path)
            ext = file_path.lower().split(".")[-1]
            try:
                if ext == "csv":
                    pd.read_csv(file_path).to_sql(base_t_name, self.agent.engine, if_exists="replace", index=False)
                elif ext in ["xls", "xlsx"]:
                    pd.read_excel(file_path).to_sql(base_t_name, self.agent.engine, if_exists="replace", index=False)
                elif ext in ["db", "sqlite"]:
                    src_conn = sqlite3.connect(file_path)
                    cursor = src_conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    for (internal_table,) in cursor.fetchall():
                        prefixed_name = f"{base_t_name}_{internal_table}"
                        df = pd.read_sql_query(f'SELECT * FROM "{internal_table}"', src_conn)
                        df.to_sql(prefixed_name, self.agent.engine, if_exists="replace", index=False)
                    src_conn.close()
                print(f"{Fore.BLUE}📦 Synced: {base_t_name}")
            except Exception as e:
                print(f"{Fore.RED}✖ Error processing {file_path}: {e}")

        # Only perform the expensive orphan-drop and agent-refresh on full sync or if actually changed
        if not specific_file or len(files_to_process) > 0:
            self.agent.refresh_agent()

    def start_monitoring(self):
        self.sync_database()  # Initial full sync
        event_handler = IngestionHandler(self)
        self.observer.schedule(event_handler, self.watch_dir, recursive=True)
        self.observer.start()
        print(f"{Fore.YELLOW}👀 Monitoring folder for changes...{Style.RESET_ALL}")

    def stop(self):
        self.observer.stop()
        self.observer.join()

    def query(self, text_input: str):
        return self.agent.ask(text_input)


if __name__ == "__main__":
    system = StructuredDataAgent()
    system.start_monitoring()
    try:
        while True:
            user_input = input(f"\n{Fore.YELLOW}❓ Query (or 'exit'): {Style.RESET_ALL}")
            if user_input.lower() in ['exit', 'quit']: break
            print(f"{Fore.CYAN}🤖 Analyzing data...{Style.RESET_ALL}")
            response = system.query(user_input)
            print(f"{Fore.GREEN}{response}{Style.RESET_ALL}")
    except KeyboardInterrupt:
        pass
    finally:
        system.stop()