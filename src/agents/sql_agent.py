import os
import pandas as pd
import sqlite3
import threading
from pathlib import Path
from sqlalchemy import create_engine, inspect, text
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
        """Re-initializes the SQL agent to detect schema changes (new/deleted tables)."""
        with self._lock:
            try:
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
                response = self.agent_executor.invoke({"input": query})
                return response["output"] if isinstance(response, dict) else str(response)
            except Exception as e:
                return f"Error: {str(e)}"


# --- The Watcher: File System Logic ---

class IngestionHandler(FileSystemEventHandler):
    def __init__(self, system_manager):
        self.manager = system_manager

    def on_modified(self, event):
        if not event.is_directory: self.manager.sync_database()

    def on_created(self, event):
        if not event.is_directory: self.manager.sync_database()

    def on_deleted(self, event):
        if not event.is_directory: self.manager.sync_database()


# --- The Manager: Orchestrator ---

class StructuredDataAgent:
    def __init__(self, db_path=None, watch_dir=None):
        home = str(Path.home())
        base_storage = os.path.join(home, ".k_rag_storage")

        # Set default paths
        if db_path is None:
            db_path = os.path.join(base_storage, "database", "main.db")
        if watch_dir is None:
            watch_dir = os.path.join(base_storage, "data")

        # Ensure directories exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(watch_dir, exist_ok=True)

        self.db_uri = f"sqlite:///{os.path.abspath(db_path)}"
        self.watch_dir = os.path.abspath(watch_dir)
        self.agent = SQLQueryAgent(self.db_uri)
        self.observer = Observer()

    def _get_table_name(self, file_path):
        """Standardizes file names to valid SQL table names."""
        return os.path.splitext(os.path.basename(file_path))[0].replace(" ", "_").replace("-", "_").lower()

    def sync_database(self):
        """Mirrors the database state to the current folder state."""
        print(f"\n{Fore.YELLOW}⚙️  Syncing database state with folder...")

        current_files = []
        valid_extensions = (".csv", ".xlsx", ".xls", ".db", ".sqlite")
        for root, _, files in os.walk(self.watch_dir):
            for f in files:
                if f.lower().endswith(valid_extensions):
                    current_files.append(os.path.join(root, f))

        expected_tables = set()

        # 1. Ingest existing files (Update/Create)
        for file_path in current_files:
            base_t_name = self._get_table_name(file_path)
            ext = file_path.lower().split(".")[-1]

            try:
                if ext == "csv":
                    pd.read_csv(file_path).to_sql(base_t_name, self.agent.engine, if_exists="replace", index=False)
                    expected_tables.add(base_t_name)
                elif ext in ["xls", "xlsx"]:
                    pd.read_excel(file_path).to_sql(base_t_name, self.agent.engine, if_exists="replace", index=False)
                    expected_tables.add(base_t_name)
                elif ext in ["db", "sqlite"]:
                    src_conn = sqlite3.connect(file_path)
                    cursor = src_conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    for (internal_table,) in cursor.fetchall():
                        prefixed_name = f"{base_t_name}_{internal_table}"
                        df = pd.read_sql_query(f'SELECT * FROM "{internal_table}"', src_conn)
                        df.to_sql(prefixed_name, self.agent.engine, if_exists="replace", index=False)
                        expected_tables.add(prefixed_name)
                    src_conn.close()

                print(f"{Fore.BLUE}📦 Synced: {base_t_name}")
            except Exception as e:
                print(f"{Fore.RED}✖ Error processing {file_path}: {e}")

        # 2. Drop orphaned tables (Cleanup)
        inspector = inspect(self.agent.engine)
        existing_tables = inspector.get_table_names()

        with self.agent.engine.begin() as conn:
            for table in existing_tables:
                if table not in expected_tables:
                    print(f"{Fore.RED}🗑️  Removing orphaned data: {table}")
                    # Using text() for SQLAlchemy 2.0 compatibility
                    conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))

        # 3. Refresh Agent
        self.agent.refresh_agent()

    def start_monitoring(self):
        """Bootstrap sync and start the directory watcher."""
        self.sync_database()
        event_handler = IngestionHandler(self)
        self.observer.schedule(event_handler, self.watch_dir, recursive=True)
        self.observer.start()
        print(f"{Fore.YELLOW}👀 Monitoring folder for changes...{Style.RESET_ALL}")

    def stop(self):
        self.observer.stop()
        self.observer.join()

    def query(self, text_input: str):
        return self.agent.ask(text_input)


# --- Execution Entry Point ---

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
        print(f"{Fore.RED}Shutting down...")
        system.stop()