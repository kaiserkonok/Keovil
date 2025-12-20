import os
import pandas as pd
import sqlite3
import threading
from pathlib import Path
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from colorama import Fore, Style, init
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Initialize colorama
init(autoreset=True)


# --- The Brain: Agent Logic ---

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
                # Re-initializing SQLDatabase ensures new tables are detected
                db = SQLDatabase(self.engine)
                llm = ChatOllama(model=self.model_name, temperature=0.1)
                toolkit = SQLDatabaseToolkit(db=db, llm=llm)
                self.agent_executor = create_sql_agent(
                    toolkit=toolkit,
                    llm=llm,
                    agent_type='tool-calling',
                    verbose=True
                )
                print(f"{Fore.CYAN}🔄 Agent schema refreshed.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}✖ Failed to refresh agent: {e}{Style.RESET_ALL}")

    def ask(self, query: str):
        with self._lock:
            if not self.agent_executor: return "Agent not ready."
            try:
                # LangChain SQL agents usually expect a dictionary input
                response = self.agent_executor.invoke({"input": query})
                return response["output"] if isinstance(response, dict) else str(response)
            except Exception as e:
                return f"Error: {str(e)}"


# --- The Watcher: File System Logic ---

class IngestionHandler(FileSystemEventHandler):
    def __init__(self, system_manager):
        self.manager = system_manager

    def on_modified(self, event):
        if not event.is_directory: self.manager.ingest_single_file(event.src_path)

    def on_created(self, event):
        if not event.is_directory: self.manager.ingest_single_file(event.src_path)


# --- The Manager: Everything Orchestrator ---

class StructuredDataAgent:
    def __init__(self, db_path=None, watch_dir=None):
        # --- CENTRALIZED STORAGE LOGIC ---
        home = str(Path.home())
        base_storage = os.path.join(home, ".k_rag_storage")

        # Default to ~/.k_rag_storage/database/main.db
        if db_path is None:
            db_path = os.path.join(base_storage, "database", "main.db")

        # Default to ~/.k_rag_storage/data (Same as RAG)
        if watch_dir is None:
            watch_dir = os.path.join(base_storage, "data")

        # Ensure directories exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(watch_dir, exist_ok=True)

        self.db_uri = f"sqlite:///{os.path.abspath(db_path)}"
        self.watch_dir = os.path.abspath(watch_dir)
        self.agent = SQLQueryAgent(self.db_uri)
        self.observer = Observer()

    def ingest_single_file(self, file_path: str):
        """Internal method to process data and update the agent."""
        ext = file_path.lower().split(".")[-1]
        if ext not in ["csv", "xlsx", "xls", "db", "sqlite"]:
            return

        table_name = os.path.splitext(os.path.basename(file_path))[0].replace(" ", "_").replace("-", "_").lower()

        try:
            if ext == "csv":
                print(f"{Fore.BLUE}📦 Ingesting CSV: {table_name}")
                first = True
                for chunk in pd.read_csv(file_path, chunksize=10000):
                    chunk.to_sql(table_name, self.agent.engine, if_exists="replace" if first else "append", index=False)
                    first = False
            elif ext in ["xls", "xlsx"]:
                print(f"{Fore.BLUE}📊 Ingesting Excel: {table_name}")
                pd.read_excel(file_path).to_sql(table_name, self.agent.engine, if_exists="replace", index=False)
            elif ext in ["db", "sqlite"]:
                self._copy_db_tables(file_path, table_name)

            print(f"{Fore.GREEN}✔ Processed {table_name}")
            self.agent.refresh_agent()
        except Exception as e:
            print(f"{Fore.RED}✖ Ingestion error: {e}")

    def _copy_db_tables(self, src_db_path, prefix):
        """Helper to merge external SQLite tables."""
        src_conn = sqlite3.connect(src_db_path)
        dest_conn = self.agent.engine.raw_connection()
        try:
            cursor = src_conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            for (table,) in cursor.fetchall():
                new_name = f"{prefix}_{table}"
                df = pd.read_sql_query(f"SELECT * FROM {table}", src_conn)
                df.to_sql(new_name, self.agent.engine, if_exists="replace", index=False)
            dest_conn.commit()
        finally:
            src_conn.close()
            dest_conn.close()

    def start_monitoring(self):
        """Bootstraps the existing files and starts the watcher."""
        print(f"{Fore.YELLOW}🚀 Initializing data sync from {self.watch_dir}...")
        for root, _, files in os.walk(self.watch_dir):
            for f in files:
                self.ingest_single_file(os.path.join(root, f))

        event_handler = IngestionHandler(self)
        self.observer.schedule(event_handler, self.watch_dir, recursive=True)
        self.observer.start()
        print(f"{Fore.YELLOW}👀 System is live and watching for changes.{Style.RESET_ALL}")

    def stop(self):
        self.observer.stop()
        self.observer.join()

    def query(self, text: str):
        return self.agent.ask(text)


# --- Clean Execution ---

if __name__ == "__main__":
    # If you leave these as None, it uses the fixed Project Storage paths
    # Or you can override them here:
    DB_FILE = None
    DATA_DIR = None

    system = StructuredDataAgent(DB_FILE, DATA_DIR)
    system.start_monitoring()

    try:
        while True:
            user_input = input(f"\n{Fore.YELLOW}❓ Query: {Style.RESET_ALL}")
            if user_input.lower() in ['exit', 'quit']: break

            print(f"{Fore.CYAN}🤖 Thinking...{Style.RESET_ALL}")
            response = system.query(user_input)
            print(f"{Fore.GREEN}{response}{Style.RESET_ALL}")
    except KeyboardInterrupt:
        pass
    finally:
        print(f"{Fore.RED}Shutting down...")
        system.stop()