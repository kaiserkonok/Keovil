#!/bin/bash

MODEL_DIR="/opt/.vault/binaries"
MODEL_PATH="$MODEL_DIR/evil.gguf"
DOWNLOAD_URL="https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q8_0.gguf"

# Force CUDA Backend
export GGML_CUDA=1
export CUDA_VISIBLE_DEVICES=0

if [ ! -f "$MODEL_PATH" ]; then
    echo "🚨 Model not found! Downloading..."
    mkdir -p "$MODEL_DIR"
    wget "$DOWNLOAD_URL" -O "$MODEL_PATH"
fi

if nvidia-smi &> /dev/null; then
    echo "🖥️ NVIDIA GPU Detected. Offloading to VRAM..."
    NGPU="-1" # -1 automatically offloads ALL layers
    DEVICE="cuda"
    KEO_CTX=32768
    KEO_BATCH=512
else
    echo "⚠️ GPU Failed. Falling back to CPU."
    NGPU="0"
    DEVICE="cpu"
    KEO_CTX=4096
    KEO_BATCH=128
fi

/usr/bin/qdrant --uri http://0.0.0.0:6333 &

echo "🚀 Igniting Llama-CPP..."
python3 -m llama_cpp.server \
  --model "$MODEL_PATH" \
  --n_gpu_layers -1 \
  --n_ctx "$KEO_CTX" \
  --flash_attn True \
  --port 7977 \
  --host 0.0.0.0 &

while ! curl -s http://localhost:6333/health > /dev/null; do sleep 1; done
while ! curl -s http://localhost:7977/v1/models > /dev/null; do sleep 2; done

echo "🔍 Checking ColBERT on $DEVICE..."
python3 -c "from pylate import models; models.ColBERT(model_name_or_path='lightonai/GTE-ModernColBERT-v1', device='$DEVICE')"

echo "✅ Keovil is ready."
exec python3 src/keovil_web/app.py