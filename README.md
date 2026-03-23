# Keovil

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub Stars](https://img.shields.io/github/stars/kaiserkonok/Keovil?style=social)](https://github.com/kaiserkonok/Keovil)

A private AI-powered query interface for your documents and structured data. Upload files, connect databases, and ask questions in plain English - Keovil generates the queries and returns results.

Works fully offline on your own GPU.

## Features

- **Structured Data Analysis**: Query CSV, Excel, SQLite, and Parquet files using natural language. Keovil generates and executes SQL via DuckDB.
- **Document Q&A**: Ask questions about your PDFs, text files, and code. Built on ColBERT retrieval with Qdrant.
- **Automatic Indexing**: Drop files in a folder and Keovil automatically syncs and indexes them.
- **Total Privacy**: Everything runs locally. No cloud, no subscriptions, no data leaves your machine.

## Requirements

### Hardware
- **GPU**: NVIDIA RTX series (30, 40, or 50 series) with **8GB+ VRAM**
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for models and data

### Software
- **Python**: 3.12+
- **CUDA**: 12.8+ (for RTX 50 series Blackwell) or 12.4+ (for RTX 30/40 series)
- **Ollama**: For running LLM models locally
- **Qdrant**: Vector database for similarity search

## Installation

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull Required Model

```bash
ollama pull qwen2.5-coder:7b-instruct
```

### 3. Install Qdrant

**macOS:**
```bash
brew install qdrant
brew services start qdrant
```

**Linux:**
```bash
curl -L https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-linux-amd64.tar.gz -o qdrant.tar.gz
tar -xzf qdrant.tar.gz
./qdrant
```

Or use Docker:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.7.4
```

### 4. Clone and Install Keovil

```bash
git clone https://github.com/kaiserkonok/Keovil.git
cd Keovil
pip install -r requirements.txt
```

## Usage

### Start Qdrant

Make sure Qdrant is running on `localhost:6333`

### Start Ollama

In a separate terminal:
```bash
ollama serve
```

### Run Keovil

```bash
python src/keovil_web/app.py
```

Open your browser to **[http://localhost:5000](http://localhost:5000)**

### Development Mode

By default, Keovil runs in development mode with separate storage:
```bash
export APP_MODE=development  # default
python src/keovil_web/app.py
```

### Production Mode

For production with unified storage:
```bash
export APP_MODE=production
python src/keovil_web/app.py
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_MODE` | `development` | `development` or `production` |
| `OLLAMA_HOST` | `127.0.0.1:11434` | Ollama server address |
| `QDRANT_HOST` | `localhost` | Qdrant server address |

### Storage Locations

- **Development**: `~/.keovil_storage_dev`
- **Production**: `~/.keovil_storage`

## Architecture

- **Ollama**: Local LLM inference
- **Qdrant**: Vector database for document similarity search
- **DuckDB**: SQL engine for structured data analysis (CSV, Excel, SQLite, Parquet)
- **Keovil**: Flask web interface combining document RAG with SQL query generation

## Troubleshooting

### Ollama Not Found
```bash
# Check Ollama is installed
ollama --version

# Check Ollama is running
ollama serve

# Pull the model
ollama pull qwen2.5-coder:7b-instruct
```

### Qdrant Not Running
```bash
# Check Qdrant is running
curl http://localhost:6333/healthz

# Should return: {"status":"ok"}
```

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
