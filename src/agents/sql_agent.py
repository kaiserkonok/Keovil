import os
import glob
import re
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_ollama import ChatOllama

# 1. PERSISTENT STORAGE
DB_PATH = "universal_data.duckdb"
engine_path = f"duckdb:///{DB_PATH}"

print(engine_path)

engine = create_engine(engine_path)


def ingest_dynamic_csvs(directory="."):
    # Recursively find every CSV
    csv_files = glob.glob(f"{directory}/**/*.csv", recursive=True)
    print(csv_files)

    with engine.connect() as conn:
        # Default performance tuning (works on any RAM size)
        conn.execute(text("SET preserve_insertion_order = false;"))

        table_counts = {}

        for file_path in csv_files:
            # Create a clean SQL-safe name
            raw_name = os.path.basename(file_path).replace('.csv', '')
            clean_name = re.sub(r'[^a-zA-Z0-9]', '_', raw_name).lower()

            # Handle duplicate filenames in different folders
            if clean_name in table_counts:
                table_counts[clean_name] += 1
                clean_name = f"{clean_name}_{table_counts[clean_name]}"
            else:
                table_counts[clean_name] = 1

            # Use DuckDB's auto-sniffer to handle large/messy files
            try:
                conn.execute(
                    text(f"CREATE OR REPLACE TABLE {clean_name} AS SELECT * FROM read_csv_auto('{file_path}');"))
                print(f"✅ Ingested {clean_name}")
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")

        conn.commit()


# Run once to sync files to storage
ingest_dynamic_csvs("/home/kaiserkonok/computer_programming/K_RAG/test_data/")

# 2. LANGCHAIN AGENT
db = SQLDatabase(engine)
# llm = ChatOllama(model='qwen2.5:7b', temperature=0)
#
# agent = create_sql_agent(
#     llm=llm,
#     db=db,
#     verbose=True,
#     allow_dangerous_code=True
# )