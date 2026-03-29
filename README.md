<img src="src/keovil_web/static/images/keo.png" width="120" align="left">

# Keovil

> Private Query Interface for Documents & Structured Data

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Platform: Linux](https://img.shields.io/badge/Platform-Linux-green.svg)](https://linux.org)
[![Hardware: RTX 8GB+](https://img.shields.io/badge/Hardware-RTX%208GB%2B-red.svg)](#requirements)
[![GitHub Stars](https://img.shields.io/github/stars/kaiserkonok/Keovil?style=social)](https://github.com/kaiserkonok/Keovil)

**Ask questions in plain English. Keovil queries your files.**

Upload PDFs, text files, code — or connect CSV, Excel, SQLite databases. Keovil generates the queries and returns results. Everything runs locally on your GPU.

## Installation

### Option 1: Install from GitHub (Recommended)

```bash
# Install the SDK and web app
pip install git+https://github.com/kaiserkonok/Keovil.git

# For specific branch (e.g., development)
pip install git+https://github.com/kaiserkonok/Keovil.git@crazy
```

### Option 2: Local Development Install

```bash
# Clone the repository
git clone https://github.com/kaiserkonok/Keovil.git
cd Keovil

# Install in development mode
pip install -e .
```

### Option 3: Web App Only (Quick Start)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull model
ollama pull qwen2.5-coder:7b-instruct

# 3. Start Qdrant (macOS)
brew install qdrant && brew services start qdrant

# 4. Install Keovil
pip install git+https://github.com/kaiserkonok/Keovil.git

# 5. Run the web app
keovil-web

# Or alternatively
python -m keovil_web
```

---

## Quick Usage Examples

### Using the SDK in your Python projects

```python
from keovil import KeovilRAG

# Initialize RAG system
rag = KeovilRAG(data_dir="/path/to/your/files")

# Index your files (PDF, text, CSV, Excel, etc.)
rag.ingest(["document1.pdf", "data.csv", "notes.txt"])

# Ask questions in natural language
answer = rag.query("What are the main findings in the documents?")
print(answer)

# For chat-like conversations
history = []
answer = rag.query("What was the revenue last year?", history)
history.extend([("You", "What was the revenue last year?"), ("AI", answer)])
answer = rag.query("How does it compare to this year?", history)
```

### Using the Full Web Application

```bash
# After installation, run the web interface
python -m keovil_web.app

# Then visit http://localhost:5000 in your browser
# Features:
# - Drag & drop files for instant indexing
# - Natural language querying of documents
# - SQL querying of structured data (CSV, Excel, etc.)
# - File system explorer
# - Secure hardware-based authentication
```

---

## Quick Start

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull model
ollama pull qwen2.5-coder:7b-instruct

# 3. Start Qdrant (macOS)
brew install qdrant && brew services start qdrant

# 4. Install Keovil
pip install git+https://github.com/kaiserkonok/Keovil.git

# 5. Run the web app
keovil-web
```

Open [http://localhost:5000](http://localhost:5000)

---

## Features

| | |
|:---|:---|
| 🗄️ **Structured Data Analysis** | Query CSV, Excel, SQLite, Parquet via natural language. Keovil generates and executes SQL via DuckDB. |
| 📄 **Document Q&A** | Ask questions about PDFs, text files, code. Built on ColBERT retrieval with Qdrant. |
| 🔄 **Automatic Indexing** | Drop files in a folder — Keovil syncs and indexes them automatically. |
| 🔒 **Total Privacy** | Everything runs locally. No cloud, no subscriptions, no data leaves your machine. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              KEOVIL INTERFACE                                │
│                    Flask Web UI (localhost:5000)                            │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│    KNOWLEDGE LAB    │ │   STRUCTURED DATA   │ │   CMS EXPLORER      │
│   (Document Q&A)    │ │       LAB           │ │   (File Manager)    │
│                     │ │   (SQL Queries)      │ │                     │
└──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
           │                        │                        │
           ▼                        ▼                        │
┌─────────────────────┐ ┌─────────────────────┐            │
│    COLBERT ENGINE   │ │   SQL QUERY AGENT   │            │
│   (Qdrant + Torch)  │ │     (DuckDB)        │            │
└──────────┬──────────┘ └──────────┬──────────┘            │
           │                        │                        │
           ▼                        ▼                        │
┌─────────────────────┐ ┌─────────────────────┐            │
│       QDRANT       │ │       DUCKDB        │            │
│  Vector Database   │ │  SQL Engine (GPU)   │            │
└─────────────────────┘ └─────────────────────┘            │
           │                        │                        │
           ▼                        ▼                        │
┌─────────────────────────────────────────────────────────────┘
│                           OLLAMA
│              Local LLM Inference (qwen2.5-coder)
│                     NVIDIA RTX GPU
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
DOCUMENTS                          STRUCTURED DATA
─────────                          ───────────────

User drops files ────────────────── User drops files
     │                                    │
     ▼                                    ▼
DocumentProcessor ────────────────── FileWatcher
(Extract text)                     (Monitor folder)
     │                                    │
     ▼                                    ▼
IntelligentChunker                  Auto-Sync to
(Smart splitting)                   DuckDB Views
     │                                    │
     ▼                                    ▼
ColBERT Embeddings                  SQL Tables
(Torch GPU)                         (CSV/XLSX/etc)
     │                                    │
     ▼                                    ▼
Qdrant VectorDB ────────────────── DuckDB
(Index & store)                    (Execute SQL)
     │                                    │
     │                                    │
     └──────────────┬───────────────────┘
                    │
                    ▼
              ┌───────────┐
              │   OLLAMA  │
              │  (LLM)    │
              └─────┬─────┘
                    │
                    ▼
              User Answer
```

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX 8GB VRAM | NVIDIA RTX 12GB+ VRAM |
| RAM | 16GB | 32GB |
| Storage | 10GB | 50GB |
| Supported GPUs | RTX 30, 40, 50 series | RTX 40/50 series |

### Software

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.12+ | Runtime |
| CUDA | 12.4+ (12.8 for RTX 50) | GPU acceleration |
| Ollama | Latest | Local LLM inference |
| Qdrant | v1.7.4+ | Vector database |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `127.0.0.1:11434` | Ollama server address |
| `QDRANT_HOST` | `localhost` | Qdrant server address |
| `STORAGE_BASE` | `~/.keovil` | Custom storage path |

### Storage Locations

```
~/.keovil/                 ← Default storage for both SDK and web app
├── data/                  # Source files
├── database/              # SQLite manifest + chat history
└── qdrant/               # Vector embeddings
```

---

## Installation (Linux)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull qwen2.5-coder:7b-instruct

# Install Qdrant
curl -L https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-linux-amd64.tar.gz -o qdrant.tar.gz
tar -xzf qdrant.tar.gz
./qdrant &

# Install Keovil
pip install git+https://github.com/kaiserkonok/Keovil.git

# Run the web app
keovil-web
```

---

## Usage Modes

### Web Application

```bash
# After pip install, run:
keovil-web

# Or use python module
python -m keovil_web
```

- Uses `~/.keovil` for storage
- Collection: `keovil_app`
- Visit http://localhost:5000

### SDK (For Developers)

```python
from keovil import KeovilRAG

rag = KeovilRAG(data_dir="/path/to/files")
rag.ingest(["file.pdf"])
answer = rag.query("Your question?")
```

- Uses `~/.keovil` for storage
- Collection: `keovil`

### Custom Storage

```bash
# Override storage location
export STORAGE_BASE=/path/to/custom/storage
keovil-web
```

---

## Troubleshooting

### Ollama Issues

```bash
# Check installation
ollama --version

# Check running
ollama serve

# Pull model
ollama pull qwen2.5-coder:7b-instruct

# List models
ollama list
```

### Qdrant Issues

```bash
# Check health
curl http://localhost:6333/healthz

# Response should be: {"status":"ok"}
```

### GPU Issues

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check VRAM
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <a href="https://github.com/kaiserkonok/Keovil">Star</a> ·
  <a href="https://github.com/kaiserkonok/Keovil/issues">Issues</a> ·
  <a href="https://kevil.io">Website</a>
</p>
