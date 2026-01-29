# 1. Official NVIDIA CUDA runtime
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 2. Install Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --set python3 /usr/bin/python3.12

# Ensure pip is up to date for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

WORKDIR /app

# 3. UNIVERSAL GPU BRIDGE
# Optimization: Pre-installing heavy AI libraries to leverage Docker caching
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# 4. Install requirements - FIX for 'uninstall-distutils-installed-package'
COPY requirements.txt .
# We use --ignore-installed to bypass the 'blinker' and 'distro' system-package errors
RUN pip3 install --no-cache-dir --ignore-installed -r requirements.txt

COPY . .

# Environment setup
ENV PYTHONPATH=/app/src
ENV DOCLING_DEVICE=cuda

# RTX 5060 Ti Optimization: Saves VRAM by loading CUDA kernels on demand
ENV CUDA_MODULE_LOADING=LAZY

EXPOSE 5000
CMD ["python3", "src/rag_chat/app.py"]