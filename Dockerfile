# --- STAGE 0: QDRANT SOURCE ---
FROM qdrant/qdrant:v1.16.2 AS qdrant_source

# --- STAGE 1: APP BUILDER ---
FROM keovil-base:latest AS builder
WORKDIR /build

# Re-activate venv from your custom base
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🧠 RESTORED: Compile your custom C-extensions for maximum speed
COPY . .
RUN python3.12 compile.py build_ext --inplace

# --- STAGE 2: FINAL RUNTIME ---
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 🛠️ SYSTEM FIXES: Install Python + EasyOCR/OpenCV Dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common curl wget libgomp1 \
    # These 3 lines are the "EasyOCR Fix"
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y python3.12 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 🔗 COPY FROM PREVIOUS STAGES
COPY --from=qdrant_source /qdrant /qdrant
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/src ./src

# Link Qdrant & Create Storage
RUN ln -s /qdrant/qdrant /usr/bin/qdrant && \
    ln -s /qdrant/static /app/static && \
    mkdir -p /qdrant/storage

# 🚀 GPU BRIDGE - Ensuring EasyOCR & Llama-CPP see the 5060 Ti
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    GGML_CUDA=1

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]