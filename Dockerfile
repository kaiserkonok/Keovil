# --- STAGE 1: THE FORGE (BUILDER) ---
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

# 1. Install Python 3.12 and Build Tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3-pip build-essential curl binutils \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# 2. Compile Phase
# 🛡️ FIX: Added flags to bypass the 'blinker' and system package lock
COPY requirements.txt .
RUN pip3 install --no-cache-dir Cython && \
    pip3 install --no-cache-dir --ignore-installed --break-system-packages -r requirements.txt

COPY . .
# Run secure compile.py (it now auto-deletes .c files locally)
RUN python3 compile.py build_ext --inplace

# 🛡️ THE STRIP: Remove debug symbols from the binaries to hide logic details
RUN find ./src -name "*.so" -exec strip --strip-unneeded {} +


# --- STAGE 2: THE VAULT (PRODUCTION) ---
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 1. Install Runtime Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 python3-pip curl libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# 2. Re-install Heavy AI Libraries (Aligned to CUDA 12.4 for your RTX 5060 Ti)
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Copy only necessary artifacts
COPY --from=builder /build/src ./src
COPY requirements.txt .
# 🛡️ FIX: Added same flags here to ensure smooth runtime setup
RUN pip3 install --no-cache-dir --ignore-installed --break-system-packages -r requirements.txt

# 4. 🛡️ THE FINAL PURGE: Delete all source, and BLANK all __init__ files
# This ensures that even if someone gets into the container, they find NO code.
RUN find ./src -name "*.py" ! -name "app.py" ! -name "__init__.py" -delete && \
    find ./src -name "__init__.py" -exec truncate -s 0 {} +

# 5. Environment for Performance
ENV PYTHONPATH=/app/src
ENV DOCLING_DEVICE=cuda
ENV CUDA_MODULE_LOADING=LAZY

EXPOSE 5000
CMD ["python3", "src/rag_chat/app.py"]