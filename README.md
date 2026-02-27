# Keovil - Private AI Data Agent

A high-performance interface for document query, structured data analysis, and real-time file engineering.

## Quick Start

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull the Model

```bash
ollama pull qwen2.5-coder:7b-instruct
```

### 3. Run Docker

```bash
docker run --network=host \
  -v ~/.keovil_storage:/data \
  -p 5000:5000 \
  keovil-local
```

## Access

Open http://localhost:5000

## GPU Support

Ensure NVIDIA Container Toolkit is installed.

## What This Does

- Uses your host's Ollama (no need to install inside container)
- Shares host network so container can reach localhost:11434
- All data stored in ~/.keovil_storage
