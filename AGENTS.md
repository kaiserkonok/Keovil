# Keovil Agent Guidelines

This file provides instructions for AI agents working in the Keovil repository.
It covers build/test commands, code style, and project conventions.

## Project Overview

Keovil is a private RAG system for documents & structured data that:
- Runs entirely locally on GPU (RTX 8GB+ VRAM recommended)
- Provides both SDK (`pip install`) and web application interfaces
- Uses ColBERT for document Q&A (Qdrant + Torch)
- Uses DuckDB for SQL querying of structured data
- Built with Flask, Ollama, Watchdog, and Rich/Colorama for terminal output

## Build/Run Commands

### Installation
```bash
# Install from GitHub (SDK + web app)
pip install git+https://github.com/kaiserkonok/Keovil.git@crazy

# For development install
git clone https://github.com/kaiserkonok/Keovil.git
cd Keovil
pip install -e .

# Install with dev dependencies (for testing)
pip install -e ".[dev]"
```

### Running the Application
```bash
# Web app (uses ~/.keovil storage)
python -m keovil_web.app

# Custom storage location
STORAGE_BASE=/path/to/storage python -m keovil_web.app

# Production mode (uses ~/.keovil)
APP_MODE=production python -m keovil_web.app

# Development mode (uses ~/.keovil_storage_dev - default)
APP_MODE=development python -m keovil_web.app
```

### SDK Usage Example
```python
from keovil import KeovilRAG

# Initialize with data directory
rag = KeovilRAG(data_dir="/path/to/files")

# Ingest documents
rag.ingest(["file1.pdf", "data.csv", "notes.txt"])

# Query documents
answer = rag.query("What are the main findings?")
```

## Test Commands

### Run All Tests
```bash
# Install dev dependencies first
pip install -e ".[dev]"

# Run test suite
python -m pytest tests/ -v
```

### Run Specific Tests
```bash
# Run individual test files directly
python tests/test_rag_agent.py
python tests/test_sql_agent.py

# Run with explicit test discovery
python -m pytest tests/test_rag_agent.py -v
```

**Note:** Tests are standalone scripts that manually adjust sys.path to import from src/
They may require setting environment variables:
- `APP_MODE=test` (for isolated test storage)
- `STORAGE_BASE=/path/to/test/storage`

## Code Style Guidelines

### Python Version
- **Target:** Python 3.12+
- **Required:** Use type hints from `typing` module
- **Minimum:** Python 3.12 specified in pyproject.toml

### Import Order
1. Standard library imports (alphabetical)
2. Third-party imports (alphabetical)
3. Local application imports (relative when in same package)
4. Blank line between sections

Example:
```python
import os
import json
import threading
from pathlib import Path
from typing import List, Dict, Any

import torch
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient

from .chunker import IntelligentChunker
from .colbert import ColBERTEngine
```

### Naming Conventions
- **Classes:** `PascalCase` (e.g., `KeovilRAG`, `IntelligentChunker`)
- **Functions/Methods:** `snake_case` (e.g., `ingest()`, `query()`, `_get_file_hash()`)
- **Constants:** `UPPERCASE_WITH_UNDERSCORES` (e.g., `SUPPORTED_EXTENSIONS`, `TOKEN_LIMIT`)
- **Variables:** `snake_case` (e.g., `data_dir`, `storage_base`, `chat_history`)
- **Private methods:** Prefix with single underscore `_` (e.g., `_initial_sync()`, `_get_stored_hashes()`)

### Type Hints
- **Required** for all public function parameters and return values
- Use built-in collection types: `list`, `dict`, `tuple` (Python 3.9+)
- Import specific types when needed: `Any`, `Union`, `Optional`, `Callable`
- For complex return types, use `->` syntax

Example:
```python
def ingest(self, new_files: List[str] = None) -> None:
    """Index files into the vector store."""
    ...

def query(self, question: str, chat_history: List = None) -> str:
    """Ask a question and get an answer."""
    ...

def _get_file_hash(self, filepath: str) -> Optional[str]:
    """Calculate MD5 hash of file."""
    ...
```

### Error Handling
- Use `try/except` blocks for I/O operations, external service calls
- Log errors using print statements with colorama for visibility
- Catch specific exceptions when possible, avoid bare `except:`
- Re-raise or return sensible defaults after logging

Example:
```python
def _get_file_hash(self, filepath: str) -> Optional[str]:
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"{Colors.FAIL}Error hashing {filepath}: {e}{Colors.ENDC}")
        return None
```

### Docstrings
- Use Google-style docstrings for public APIs
- Include Args, Returns, and Raises sections when applicable
- Keep descriptions concise but informative
- Match the style of existing code in the repository

Example:
```python
def ingest(self, new_files: List[str] = None) -> None:
    """Index files into the vector store.
    
    Args:
        new_files: List of file paths to ingest. If None, does nothing.
        
    Returns:
        None
        
    Side effects:
        Updates vector store and manifest database
    """
```

### Formatting
- **Line length:** Aim for 88-100 characters (follow existing code)
- **Indentation:** 4 spaces (no tabs)
- **Blank lines:** 
  - 2 blank lines between top-level functions/classes
  - 1 blank line between method definitions in a class
  - Use blank lines to separate logical sections within functions
- **Quotes:** Use double quotes `"` for strings (consistent with existing code)
- **Trailing commas:** Use in multi-line tuples/lists/dicts for easier diffs

### Specific Conventions in Keovil

#### Storage Paths
- Production: `~/.keovil` (when `APP_MODE=production`)
- Development: `~/.keovil_storage_dev` (default when `APP_MODE=development` or unset)
- Override with `STORAGE_BASE` environment variable
- Always use `Path` objects for path manipulation

#### Terminal Output
- Use `colorama.Fore` and `colorama.Style` for colored output
- Use `rich.console.Console` for rich output (tables, panels, etc.)
- Define color constants at top of file (like `Colors` class)
- Reset colors after use with `Style.RESET_ALL` or `Colors.ENDC`

Example from existing code:
```python
from colorama import Fore, Style, init
init(autoreset=True)
print(f"{Fore.CYAN}[Sync] Reconciling Store...{Style.RESET_ALL}")
```

#### Dependencies
- All dependencies listed in `pyproject.toml`
- Dev dependencies in `[project.optional-dependencies] dev`
- Critical runtime deps: torch, transformers, qdrant-client, langchain-ollama, flask
- GPU-dependent packages: torch, pylate (for ColBERT)

#### Environment Variables
Check for these in code:
- `APP_MODE`: "development" or "production" 
- `STORAGE_BASE`: Custom storage path
- `OLLAMA_HOST`: Ollama server address (default: 127.0.0.1:11434)
- `QDRANT_HOST`: Qdrant server address (default: localhost)
- `MODE`: Used in KeovilRAG constructor ("development", "production", "sdk")

#### Threading
- Use `threading.Lock()` for shared resources
- Lock acquisition with `with self.lock:` context manager
- Background workers as daemon threads

Example:
```python
self.lock = threading.Lock()
with self.lock:
    # critical section
```

## Key Architecture Points

### SDK vs App Layer
- **SDK (`src/keovil/`):** Pure RAG logic (`KeovilRAG` class)
  - No UI, no file watching, no SocketIO
  - Import with `from keovil import KeovilRAG`
- **App layer (`src/keovil_web/` and `src/college_rag.py`):**
  - Adds file watching, background processing, web UI
  - `CollegeRAG` extends `KeovilRAG` with additional features

### Storage Organization
When using `~/.keovil` (or `STORAGE_BASE`):
```
~/.keovil/
├── data/           # Original files (symlinks or copies)
├── database/
│   └── manifest.db # SQLite tracking file hashes
└── qdrant/         # Vector database storage (managed by Qdrant)
```

### Testing Approach
- Tests use isolated storage directories
- Set `APP_MODE=test` and `STORAGE_BASE` to temporary paths
- Tests clean up after themselves (remove test directories)
- Tests are run directly as Python scripts, not through pytest discovery alone

## Things to Avoid

1. **Hardcoded paths** - Always use `Path.home() / ".keovil"` or `storage_dir` parameter
2. **Missing type hints** - Add them for all public APIs
3. **Inconsistent naming** - Follow existing patterns in the file you're modifying
4. **Long functions** - Break into smaller helper methods when >50 lines
5. **Broad exception handling** - Catch specific exceptions when possible
6. **Print statements without color** - Use colorama for visibility in terminal
7. **Missing docstrings** - Public methods/classes should have docstrings

## When in Doubt

1. Look at existing code in the same file/module
2. Follow the patterns established in `src/keovil/rag.py` (core SDK code)
3. Check how similar functionality is implemented elsewhere
4. Run existing tests to ensure your changes don't break anything
5. Verify storage paths respect `STORAGE_BASE` environment variable

This guide reflects the current state of the Keovil codebase. As the project evolves, these guidelines may be updated.