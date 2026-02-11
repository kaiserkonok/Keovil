#!/bin/bash

# 1. Start Qdrant
export QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
export QDRANT__SERVICE__ENABLE_STATIC_CONTENT=true
/usr/bin/qdrant --uri http://0.0.0.0:6333 &

# 2. Start Ollama
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/ollama
export OLLAMA_INTEL_GPU=false
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_MODELS=/root/.ollama/models
/usr/bin/ollama serve &

# 3. Wait for Services (No more manual sleep!)
echo "Waiting for services to ignite..."
while ! curl -s http://localhost:6333/health > /dev/null; do sleep 0.2; done
while ! curl -s http://localhost:11434/api/tags > /dev/null; do sleep 0.2; done

# 4. Pre-warm the Model into VRAM
# This loads the weights into your 5060 Ti NOW so the first query is instant.
echo "Pre-loading model into 16GB VRAM..."
curl -s -X POST http://localhost:11434/api/generate \
     -d "{\"model\": \"qwen2.5-coder:7b-instruct\", \"keep_alive\": -1}" > /dev/null

echo "Keovil is hot and ready."

# 5. Start the engine
exec python3 src/keovil_web/app.py