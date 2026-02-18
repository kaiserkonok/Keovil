# --- STAGE 0: QDRANT SOURCE ---
FROM qdrant/qdrant:v1.16.2 AS qdrant_source

# --- STAGE 1: APP BUILDER ---
FROM keovil-base:latest AS builder
WORKDIR /build

# Re-activate venv from base
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🧠 Compile your custom C-extensions (Accuracy/Speed)
COPY . .
RUN python3.12 compile.py build_ext --inplace

# --- STAGE 2: FINAL RUNTIME ---
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Minimal Runtime Libs
RUN apt-get update && apt-get install -y \
    software-properties-common curl wget libgomp1 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y python3.12 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 🔗 COPY FROM STAGES
COPY --from=qdrant_source /qdrant /qdrant
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/src ./src

# Link Qdrant
RUN ln -s /qdrant/qdrant /usr/bin/qdrant && \
    ln -s /qdrant/static /app/static && \
    mkdir -p /qdrant/storage

# 🚀 GPU BRIDGE
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    GGML_CUDA=1

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]