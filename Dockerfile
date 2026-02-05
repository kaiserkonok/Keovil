# --- STAGE 1: THE FORGE (BUILDER) ---
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

# 1. Install Python 3.12 and Build Tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv python3-pip build-essential curl binutils \
    && rm -rf /var/lib/apt/lists/*

# 2. VENV SETUP
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 3. Cache Heavy Libraries (Torch/CUDA)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir Cython && \
    rm requirements.txt

# 5. Compile & Strip
COPY . .
# ⚡ Speed tweak: NPROC uses all available CPU cores for Cython compilation
RUN NPROC=$(nproc) python3 compile.py build_ext --inplace && \
    find ./src -name "*.so" -exec strip --strip-unneeded {} +


# --- STAGE 2: THE VAULT (PRODUCTION) ---
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 1. Minimal Runtime Environment
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. THE LIFT & SHIFT
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 3. Copy Compiled Binaries
COPY --from=builder /build/src ./src

# 4. 🛡️ THE FINAL PURGE (Hardened)
# We delete ALL .py files, then specifically restore the ONE entry point we need.
# This ensures zero source-code leakage of your HWID/Registry logic.
RUN find ./src -name "*.py" ! -path "*/keovil_web/app.py" ! -name "__init__.py" -delete && \
    find ./src -name "__init__.py" -exec truncate -s 0 {} +

# 5. Performance Tweaks (RTX 5060 Ti Optimized)
ENV PYTHONPATH=/app
ENV DOCLING_DEVICE=cuda
ENV CUDA_MODULE_LOADING=LAZY

EXPOSE 5000
# Matches your verified tree path
CMD ["python3", "src/keovil_web/app.py"]