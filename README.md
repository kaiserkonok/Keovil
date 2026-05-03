<img src="src/keovil_web/static/images/keo.png" width="120" align="left">

# Keovil

> Private Query Interface for Documents & Structured Data

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Platform: Linux](https://img.shields.io/badge/Platform-Linux-green.svg)](https://linux.org)
[![Hardware: Flexible](https://img.shields.io/badge/Hardware-Flexible-green)](#requirements)
[![PyPI](https://img.shields.io/pypi/v/keovil.svg)](https://pypi.org/project/keovil/)
[![GitHub Stars](https://img.shields.io/github/stars/kaiserkonok/Keovil?style=social)](https://github.com/kaiserkonok/Keovil)

**Ask questions in plain English. Keovil queries your files.**

Upload PDFs, text files, code, and more. Or query CSV, Excel, SQLite databases via the web app. Keovil generates the queries and returns results. Document processing runs locally on your GPU, with flexible LLM options (cloud or local).

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install keovil
```

### Option 2: Install from GitHub (Latest Development)

```bash
pip install git+https://github.com/kaiserkonok/Keovil.git
```

### Option 3: Local Development Install

```bash
# Clone the repository
git clone https://github.com/kaiserkonok/Keovil.git
cd Keovil

# Install in development mode
pip install -e .
```

### Quick Start

```bash
# 1. Install Keovil
pip install keovil

# 2. Run the web app (Qdrant runs automatically!)
python -m keovil_web
```

That's it! Qdrant runs in embedded mode automatically.

---

### Optional: Run Qdrant Manually (For Better Performance)

If you want to run Qdrant externally (e.g., via Docker) for better performance:

```bash
# macOS
brew install qdrant && brew services start qdrant

# Linux (Docker)
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant_storage qdrant/qdrant
```

Then set `QDRANT_HOST=your-server` if not on localhost.

---

## Quick Start

### Web App

```bash
# 1. Install Keovil
pip install keovil

# 2. Run the web app (Qdrant runs automatically!)
python -m keovil_web
```

Open [http://localhost:5000](http://localhost:5000)

> **That's it!** Qdrant runs in embedded mode automatically. No Docker needed.
>
> For local LLM, also install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`

### SDK (For Developers)

```python
from keovil import KeovilRAG
from keovil.utils.llm_config import LLMConfig

# Initialize with default Ollama
rag = KeovilRAG(data_dir="/path/to/your/files")

# Or use a specific LLM provider
config = LLMConfig(provider="openai", model="gpt-4o", openai_api_key="sk-...")
rag = KeovilRAG(data_dir="/path/to/files", llm_config=config)

# Index your files (PDF, text, code, etc.)
rag.ingest(["document1.pdf", "notes.txt"])

# Ask questions in natural language
answer = rag.query("What is the recommended dosage for adults over 65?")
print(answer)

# For chat-like conversations
history = []
answer = rag.query("What are the API rate limits for the Pro plan?", history)
history.extend([("You", "What are the API rate limits for the Pro plan?"), ("AI", answer)])
answer = rag.query("Does the Enterprise plan include SSO?", history)
```

---

## Linux Installation

```bash
# Option 1: Just Keovil (Qdrant auto-runs in embedded mode)
pip install keovil
python -m keovil_web

# Option 2: With local Ollama (if using local LLM)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:7b-instruct
pip install keovil
python -m keovil_web

# Option 3: With external Qdrant (for better performance)
pip install keovil
curl -L https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-linux-amd64.tar.gz -o qdrant.tar.gz
tar -xzf qdrant.tar.gz
./qdrant &
python -m keovil_web
```

---

## Features

| | |
|:---|:---|
| 🗄️ **Structured Data Analysis (Web App)** | Query CSV, Excel, SQLite, Parquet via natural language. Keovil generates and executes SQL via DuckDB. |
| 📄 **Document Q&A** | Ask questions about PDFs, text files, code. Built on ColBERT retrieval with Qdrant. Docling parses documents locally. |
| 🔄 **Automatic Indexing** | Drop files in a folder — Keovil syncs and indexes them automatically. |
| 🔒 **Total Privacy** | Document processing runs locally. Cloud LLMs are optional - use local Ollama for full privacy. |
| 🌐 **Multi-LLM Support** | Use Ollama (local), OpenAI, Anthropic, OpenRouter, or Gemini. Change anytime without restart. |
| ⚡ **Flexible Hardware** | Full GPU not required - can use cloud LLMs with local document processing. |

---

## Supported LLM Providers

### Web App
Change providers anytime via Settings page - no restart needed!

| Provider | Description | API Key Required |
|----------|-------------|------------------|
| **Ollama** | Local models running on your machine | No |
| **OpenAI** | GPT-4o, GPT-4o-mini, etc. | Yes |
| **Anthropic** | Claude 3.5 Sonnet, Haiku, etc. | Yes |
| **OpenRouter** | Access 100+ models via single API | Yes |
| **Gemini** | Google Gemini 2.0, 1.5 Pro, etc. | Yes |

### SDK (For Developers)

```python
from keovil import KeovilRAG
from keovil.utils.llm_config import LLMConfig

# Ollama (default)
rag = KeovilRAG(data_dir="/path", llm_config=LLMConfig(provider="ollama", model="qwen2.5-coder:7b"))

# OpenAI
config = LLMConfig(provider="openai", model="gpt-4o", openai_api_key="sk-...")
rag = KeovilRAG(data_dir="/path", llm_config=config)

# Anthropic
config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022", anthropic_api_key="sk-ant-...")
rag = KeovilRAG(data_dir="/path", llm_config=config)

# OpenRouter
config = LLMConfig(provider="openrouter", model="openai/gpt-4o-mini", openrouter_api_key="sk-or-...")
rag = KeovilRAG(data_dir="/path", llm_config=config)

# Gemini
config = LLMConfig(provider="gemini", model="gemini-2.0-flash", gemini_api_key="AIza...")
rag = KeovilRAG(data_dir="/path", llm_config=config)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              KEOVIL INTERFACE                                │
│                    Flask Web UI (localhost:5000)                            │
│                    └── Settings: Switch LLM providers                       │
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
┌─────────────────────────────────────────────────────────────┐
│                        LLM PROVIDERS                         │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │
│   │ Ollama  │ │ OpenAI  │ │Anthropic│ │OpenRouter│ │ Gemini │ │
│   │ (Local) │ │  GPT-4  │ │Claude-3 │ │ 100+ LMs │ │Gemini  │ │
│   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └────────┘ │
│              Dynamic switching - no restart needed!          │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
DOCUMENTS                          STRUCTURED DATA (Web App Only)
─────────                          ──────────────────────────────

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
              │   LLM    │
              │(Any Prov)│
              └─────┬─────┘
                    │
                    ▼
              User Answer
```

---

## Requirements

### Hardware

#### With Cloud LLM (OpenAI, Anthropic, Gemini, OpenRouter)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 6GB (RTX 3060) | 8GB+ (RTX 4060/4070) |
| RAM | 16GB | 32GB |

> Cloud LLMs don't use local GPU - GPU only needed for Docling (document parsing) + ColBERT (embeddings). 6GB minimum tested and working; 8GB recommended.

#### With Local LLM (Ollama)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 10GB+ | 16GB+ |
| RAM | 16GB | 32GB |

> Local Ollama + ColBERT + Docling all need GPU. 10GB minimum for 7B models + embeddings. 16GB recommended for smooth operation.

#### No GPU (CPU Only)

| Component | Minimum |
|-----------|---------|
| RAM | 16GB |

> Works but extremely slow. Document ingestion minutes vs seconds. Not recommended for production use.

### Software

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.12+ | Runtime |
| CUDA | 12.4+ (12.8 for RTX 50) | GPU acceleration (if using GPU) |
| Ollama | Latest | Local LLM (only if using local) |
| Qdrant | Optional | Vector database (auto-embedded if not running) |

> **No Docker needed!** Qdrant runs automatically in embedded mode if no external server is available. Everything works out of the box.

> **No GPU?** Works with cloud LLMs only. Document processing will be slower but functional.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KEOVIL_PROVIDER` | `ollama` | LLM provider (ollama, openai, anthropic, openrouter, gemini) |
| `KEOVIL_MODEL` | provider-specific | Model name |
| `OLLAMA_HOST` | `127.0.0.1:11434` | Ollama server address |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `OPENROUTER_API_KEY` | - | OpenRouter API key |
| `GEMINI_API_KEY` | - | Google Gemini API key |
| `QDRANT_HOST` | `localhost` | Qdrant server address |
| `STORAGE_BASE` | `~/.keovil` | Custom storage path |

### Storage Locations

```
~/.keovil/                 ← Default storage for both SDK and web app
├── data/                  # Source files
├── database/              # SQLite manifest + chat history
├── config.json            # LLM provider settings
└── qdrant/               # Vector embeddings
```

---

## Usage Modes

### Web Application

```bash
# After pip install, run:
python -m keovil_web
```

- Uses `~/.keovil` for storage
- Collection: `keovil_app`
- Visit http://localhost:5000
- Go to Settings to change LLM provider anytime

### SDK (For Developers)

```python
from keovil import KeovilRAG

rag = KeovilRAG(data_dir="/path/to/files")
rag.ingest(["file.pdf"])
answer = rag.query("Your question?")
```

- Uses `~/.keovil` for storage
- Collection: `keovil`
- Pass `llm_config` to use different providers

### Custom Storage

```bash
# Override storage location
export STORAGE_BASE=/path/to/custom/storage
python -m keovil_web
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