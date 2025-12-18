import sys
from pathlib import Path
from typing import List

from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.theme import Theme

# Customizing colors: Neon cyan for AI, Magenta for user, Green for files
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "user": "bold orchid",
    "bot": "bold turquoise2",
    "filepath": "italic green"
})

console = Console(theme=custom_theme)


class CSVAgent:
    """
    Natural language querying over one or more CSV files.
    """

    def __init__(self, llm: ChatOllama, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose

    def run(self, csv_files: List[Path], query: str) -> str:
        if not csv_files:
            return "No CSV files provided to CSVAgent."

        agent = create_csv_agent(
            llm=self.llm,
            path=[str(p) for p in csv_files],
            verbose=self.verbose,
            allow_dangerous_code=True,
        )
        return agent.invoke(query)["output"]


def run_terminal_chat():
    # 1. Setup Data
    # Assuming your local path remains the same
    DATA_DIR = Path("/home/kaiserkonok/computer_programming/K_RAG/test_data/")

    # Mocking your file collector (ensure your import works)
    try:
        from src.utils.file_collector import collect_tabular_files
        csv_files, _ = collect_tabular_files(DATA_DIR)
    except ImportError:
        # Fallback for demonstration if import fails
        csv_files = list(DATA_DIR.glob("*.csv"))

    # 2. Initialize LLM
    llm = ChatOllama(model="qwen2.5:7b", temperature=0)
    agent = CSVAgent(llm, verbose=True)  # Verbose False keeps terminal clean

    # 3. UI Header
    console.print(Panel.fit(
        "[bold cyan]CSV Data Chatbot[/bold cyan]\n[dim]Powered by LangChain & Qwen2.5[/dim]",
        border_style="turquoise2"
    ))

    if not csv_files:
        console.print("[danger]No CSV files found in the directory![/danger]")
        return

    console.print("\n[bold]📂 Loaded Files:[/bold]")
    for f in csv_files:
        console.print(f"  [filepath]✔ {f.name}[/filepath]")

    console.print(
        "\n[info]Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to stop the session.[/info]\n")

    # 4. Chat Loop
    while True:
        try:
            user_input = Prompt.ask("[user]You[/user]")

            if user_input.lower() in ["exit", "quit"]:
                console.print("[warning]Goodbye! 👋[/warning]")
                break

            with console.status("[bold blue]Thinking...[/bold blue]", spinner="dots"):
                response = agent.run(csv_files, user_input)

            # Output the response in a nice panel
            console.print(Panel(Text(response, style="bot"), title="[bot]Agent[/bot]", title_align="left",
                                border_style="turquoise2"))
            console.print("")  # Whitespace

        except KeyboardInterrupt:
            console.print("\n[warning]Session ended by user.[/warning]")
            break
        except Exception as e:
            console.print(f"[danger]An error occurred: {e}[/danger]")


if __name__ == "__main__":
    run_terminal_chat()