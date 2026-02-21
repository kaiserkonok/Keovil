# --- STAGE 0: BINARY SOURCES ---
FROM ollama/ollama:0.5.7 AS ollama_source
FROM qdrant/qdrant:v1.16.2 AS qdrant_source

# --- STAGE 1: THE FORGE (BUILDER) ---
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

# 1. System Setup
RUN apt-get update && apt-get install -y \
    software-properties-common curl binutils \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv python3-pip build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. VENV SETUP
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 3. THE "PERMANENT" CACHE LAYER (Ollama & Torch)
COPY --from=ollama_source /bin/ollama /usr/bin/ollama

RUN nohup ollama serve > /dev/null 2>&1 & \
    echo "Waiting for Ollama to ignite..." && \
    while ! curl -s http://localhost:11434/api/tags > /dev/null; do sleep 1; done && \
    ollama pull qwen2.5-coder:7b-instruct

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# 4. INSTALL REQUIREMENTS
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. DOWNLOAD COLBERT
RUN python3 -c "from pylate import models; models.ColBERT(model_name_or_path='lightonai/GTE-ModernColBERT-v1', device='cpu')"

# 6. COMPILE CODE
COPY . .
RUN NPROC=$(nproc) python3 compile.py build_ext --inplace

# --- STAGE 2: THE VAULT (PRODUCTION) ---
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y software-properties-common curl \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy binaries and venv
COPY --from=ollama_source /bin/ollama /usr/bin/ollama
COPY --from=ollama_source /lib/ollama /usr/lib/ollama
COPY --from=qdrant_source /qdrant /qdrant

# Setup Qdrant links and paths
RUN ln -s /qdrant/qdrant /usr/bin/qdrant && \
    ln -s /qdrant/static /app/static && \
    mkdir -p /qdrant/storage

# Copy heavy data
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.ollama /root/.ollama
COPY --from=builder /root/.cache /root/.cache

# --- 🛡️ THE ATOMIC VAULT (STRICT PURGE) ---
# We copy everything FIRST to ensure templates, static, and assets are preserved.
COPY --from=builder /build/src ./src

# We run a multi-targeted purge to delete code but keep the engine running.
# 1. Delete all .c files (Cython leftovers)
# 2. Delete all __pycache__ folders
# 3. Delete all .py files EXCEPT the entry point (app.py) and __init__.py files
# 4. TRUNCATE __init__.py files to 0 bytes (Keeps them for imports but hides code)
RUN find ./src -name "*.c" -delete && \
    find ./src -name "__pycache__" -type d -exec rm -rf {} + && \
    find ./src -name "*.py" ! -path "*/keovil_web/app.py" ! -name "__init__.py" -delete && \
    find ./src -name "__init__.py" -exec truncate -s 0 {} +

# Verification Check: This will fail the build if a secret .py file is still there
RUN if [ $(find ./src -name "neural_db.py" | wc -l) -ne 0 ]; then echo "SECURITY BREACH: .py files still exist!"; exit 1; fi

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    OLLAMA_HOST=127.0.0.1:11434 \
    QDRANT_HOST=localhost \
    OLLAMA_KEEP_ALIVE=-1 \
    CUDA_MODULE_LOADING=LAZY \
    PYTHONDONTWRITEBYTECODE=1

ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]