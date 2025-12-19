import os
import pandas as pd
import sqlite3
import threading
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from colorama import Fore, Style, init
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Initialize colorama
init(autoreset=True)

# Global threading lock
db_lock = threading.Lock()


def copy_external_db_tables(src_db_path: str, dest_conn, prefix: str):
    try:
        src_conn = sqlite3.connect(src_db_path)
        src_cursor = src_conn.cursor()
        src_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in src_cursor.fetchall()]

        for table in tables:
            new_table_name = f"{prefix}_{table}"
            print(f"{Fore.CYAN}{Style.BRIGHT}→ Copying table {Fore.YELLOW}{table}{Style.RESET_ALL} "
                  f"from {Fore.MAGENTA}{src_db_path}{Style.RESET_ALL} "
                  f"into {Fore.GREEN}{new_table_name}{Style.RESET_ALL}")

            src_cursor.execute(f"SELECT * FROM {table}")
            rows = src_cursor.fetchall()
            src_cursor.execute(f"PRAGMA table_info({table})")
            cols = [col[1] for col in src_cursor.fetchall()]
            col_defs = ", ".join([f'"{c}" TEXT' for c in cols])

            dest_cursor = dest_conn.cursor()
            dest_cursor.execute(f'DROP TABLE IF EXISTS "{new_table_name}"')
            dest_cursor.execute(f'CREATE TABLE "{new_table_name}" ({col_defs});')

            placeholders = ", ".join(["?"] * len(cols))
            for i in range(0, len(rows), 10000):
                batch = rows[i:i+10000]
                dest_cursor.executemany(
                    f'INSERT INTO "{new_table_name}" VALUES ({placeholders});',
                    batch
                )
                dest_conn.commit()

        src_conn.close()
    except Exception as e:
        print(f"{Fore.RED}✖ Failed to copy tables from {src_db_path}: {e}{Style.RESET_ALL}")


def ingest_file(file_path: str, engine, chunksize: int = 10000):
    ext = file_path.lower().split(".")[-1]
    try:
        if ext == "csv":
            table_name = os.path.splitext(os.path.basename(file_path))[0].replace(" ", "_").replace("-", "_").lower()
            print(f"{Fore.BLUE}{Style.BRIGHT}📦 Ingesting CSV {Fore.MAGENTA}{file_path}{Style.RESET_ALL} "
                  f"into table {Fore.GREEN}{table_name}{Style.RESET_ALL}")
            first_chunk = True
            for chunk in pd.read_csv(file_path, chunksize=chunksize):
                chunk.to_sql(
                    table_name,
                    engine,
                    if_exists="replace" if first_chunk else "append",
                    index=False,
                )
                first_chunk = False
            print(f"{Fore.GREEN}✔ Finished ingesting {file_path}{Style.RESET_ALL}")

        elif ext in ["xls", "xlsx"]:
            table_name = os.path.splitext(os.path.basename(file_path))[0].replace(" ", "_").replace("-", "_").lower()
            print(f"{Fore.BLUE}{Style.BRIGHT}📊 Ingesting Excel {Fore.MAGENTA}{file_path}{Style.RESET_ALL} "
                  f"into table {Fore.GREEN}{table_name}{Style.RESET_ALL}")
            df = pd.read_excel(file_path)
            df.to_sql(table_name, engine, if_exists="replace", index=False)
            print(f"{Fore.GREEN}✔ Finished ingesting {file_path}{Style.RESET_ALL}")

        elif ext in ["db", "sqlite"]:
            prefix = os.path.splitext(os.path.basename(file_path))[0].replace(" ", "_").replace("-", "_").lower()
            print(f"{Fore.BLUE}{Style.BRIGHT}🗄 Copying tables from external DB {Fore.MAGENTA}{file_path}{Style.RESET_ALL} "
                  f"with prefix {Fore.GREEN}{prefix}{Style.RESET_ALL}")
            dest_conn = engine.raw_connection()
            copy_external_db_tables(file_path, dest_conn, prefix)
            dest_conn.close()
            print(f"{Fore.GREEN}✔ Finished copying DB {file_path}{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}✖ Failed to ingest {file_path}: {e}{Style.RESET_ALL}")


def build_agent(engine):
    """Rebuild the SQL agent with fresh schema reflection."""
    db = SQLDatabase(engine)
    llm = ChatOllama(model='qwen2.5:7b', temperature=0.7)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return create_sql_agent(toolkit=toolkit, llm=llm, agent_type='tool-calling', verbose=True)


class IngestionHandler(FileSystemEventHandler):
    def __init__(self, engine):
        self.engine = engine
        self.agent = build_agent(engine)

    def on_modified(self, event):
        if not event.is_directory:
            with db_lock:
                ingest_file(event.src_path, self.engine)
                self.agent = build_agent(self.engine)  # refresh agent

    def on_created(self, event):
        if not event.is_directory:
            with db_lock:
                ingest_file(event.src_path, self.engine)
                self.agent = build_agent(self.engine)  # refresh agent


if __name__ == "__main__":
    db_uri = 'sqlite:////home/kaiserkonok/computer_programming/K_RAG/database/main.db'
    engine = create_engine(db_uri)

    # Initial ingestion of all files
    for dirpath, _, filenames in os.walk("/home/kaiserkonok/computer_programming/K_RAG/test_data/"):
        for filename in filenames:
            ingest_file(os.path.join(dirpath, filename), engine)

    # Setup watchdog observer
    path = "/home/kaiserkonok/computer_programming/K_RAG/test_data/"
    event_handler = IngestionHandler(engine)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    print(f"{Fore.YELLOW}{Style.BRIGHT}👀 Watching {path} for changes...{Style.RESET_ALL}")

    try:
        while True:
            q = input(f"{Fore.YELLOW}{Style.BRIGHT}❓ Query:{Style.RESET_ALL} ")
            print(f"{Fore.CYAN}{Style.BRIGHT}🤖 Agent says:{Style.RESET_ALL}")
            with db_lock:
                response = event_handler.agent.invoke(q)
            if isinstance(response, dict) and "output" in response:
                print(f"{Fore.GREEN}{response['output']}{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}{response}{Style.RESET_ALL}")
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
