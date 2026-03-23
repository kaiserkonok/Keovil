![Keovil Logo](src/keovil_web/static/images/keo.png)

# Keovil

> Private Query Interface for Documents & Structured Data

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Platform: Linux](https://img.shields.io/badge/Platform-Linux-green.svg)](https://linux.org)
[![Hardware: RTX 8GB+](https://img.shields.io/badge/Hardware-RTX%208GB%2B-red.svg)](#requirements)
[![GitHub Stars](https://img.shields.io/github/stars/kaiserkonok/Keovil?style=social)](https://github.com/kaiserkonok/Keovil)

**Ask questions in plain English. Keovil queries your files.**

Upload PDFs, text files, code — or connect CSV, Excel, SQLite databases. Keovil generates the queries and returns results. Everything runs locally on your GPU.

---

## Quick Start

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull model
ollama pull qwen2.5-coder:7b-instruct

# 3. Start Qdrant (macOS)
brew install qdrant && brew services start qdrant

# 4. Clone & install
git clone https://github.com/kaiserkonok/Keovil.git
cd Keovil
pip install -r requirements.txt

# 5. Run
python src/keovil_web/app.py
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
| `APP_MODE` | `development` | `development` or `production` |
| `OLLAMA_HOST` | `127.0.0.1:11434` | Ollama server address |
| `QDRANT_HOST` | `localhost` | Qdrant server address |
| `STORAGE_BASE` | `~/.keovil_storage(_dev)` | Custom storage path |

### Storage Locations

```
~/.keovil_storage_dev/     ← Development mode (default)
~/.keovil_storage/         ← Production mode
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

# Clone & install
git clone https://github.com/kaiserkonok/Keovil.git
cd Keovil
pip install -r requirements.txt

# Run
python src/keovil_web/app.py
```

---

## Usage Modes

### Development Mode (Default)

```bash
export APP_MODE=development
python src/keovil_web/app.py
```

- Isolated storage: `~/.keovil_storage_dev`
- Separate collections for testing

### Production Mode

```bash
export APP_MODE=production
python src/keovil_web/app.py
```

- Unified storage: `~/.keovil_storage`
- All data in one place

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
