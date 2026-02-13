import sys
import time
import duckdb
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

# 1. BOOTSTRAP ENVIRONMENT
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.agents.db_agent import StructuredDataAgent

console = Console()


# 2. WEB-READY SIGNAL VALIDATOR
class MockSocket:
    def __init__(self):
        self.emissions = []

    def emit(self, event, data, namespace=None):
        rag = data.get('rag', {})
        state = rag.get('state', 'unknown')
        prog = rag.get('progress', 0)
        file = rag.get('current_file', 'No File')

        # Colorful Visual Feedback for Web Readiness
        status_color = "green" if state == "processing" else "yellow"
        if state == "idle": status_color = "blue"

        console.print(f"[{status_color}]📡 WEB SIGNAL:[/][bold white] {state.upper()}[/] | {file} | [cyan]{prog}%[/]")

        # VALIDATION FOR chat.html
        if 'sql_syncing' in data and 'rag' in data:
            # This confirms the frontend JS will pick it up
            pass
        else:
            console.print("[bold red]⚠️  WARNING: Payload format missing keys for frontend animation![/]")

        self.emissions.append(data)


# 3. HIGH-SPEED DATA FACTORY
def generate_complex_datasets(data_dir):
    paths = []
    with console.status("[bold green]🏗️  Manufacturing Stress Test Data...") as status:
        # A. SMALL CSVs (Rapid Burst)
        for i in range(5):
            p = data_dir / f"burst_{i}.csv"
            pd.DataFrame({'id': range(100)}).to_csv(p, index=False)
            paths.append(p)

        # B. MEDIUM PARQUET (Efficient Format)
        p_med = data_dir / "performance_metrics.parquet"
        pd.DataFrame({'metric': range(50000), 'val': [i * 1.1 for i in range(50000)]}).to_parquet(p_med)
        paths.append(p_med)

        # C. THE BEAST: 10M Row CSV (~1.2GB)
        p_heavy = data_dir / "enterprise_load.csv"
        con = duckdb.connect()
        con.execute(
            f"COPY (SELECT range as id, random() as val, 'DATA_BLOCK_' || (range % 100) as tag FROM range(10000000)) TO '{p_heavy}' (HEADER, DELIMITER ',');")
        paths.append(p_heavy)

    console.print(Panel(f"Generated {len(paths)} files (Total ~1.3GB)", title="Factory Success", border_style="green"))
    return paths


# 4. MAIN TEST RUNNER
def run_ultra_test():
    base_path = Path.home() / ".keovil_storage_dev"
    data_dir = base_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    mock_socket = MockSocket()
    agent = StructuredDataAgent(socketio=mock_socket, watch_dir=str(data_dir))

    test_files = []
    try:
        test_files = generate_complex_datasets(data_dir)

        console.print("\n[bold magenta]🚀 STARTING MULTI-FILE INGESTION ENGINE[/]")
        start_time = time.time()

        # RUN ENGINE
        agent.sync_database()

        end_time = time.time()
        console.print(f"\n[bold green]🏁 INGESTION COMPLETE IN {end_time - start_time:.2f}s[/]")

        # 5. SCALE-QUERY ACCURACY
        console.print("\n[bold yellow]🔎 PERFORMANCE QUERY (10M Rows Scan)...[/]")
        response = agent.query("Calculate the sum of 'val' in enterprise_load where id < 1000000")

        console.print(Panel(response, title="RTX 5060 Ti Output", border_style="cyan"))

    except Exception as e:
        console.print(f"[bold red]FATAL ERROR: {e}[/]")
    finally:
        console.print("\n[bold red]🧹 TEARDOWN: NO GARBAGE LEFT BEHIND[/]")
        for f in test_files:
            if f.exists(): f.unlink()
        agent.sync_database()
        console.print("[bold green]✨ DEV ENVIRONMENT RESET.[/]")


if __name__ == "__main__":
    run_ultra_test()