#!/bin/bash
set -e

# --- CONFIG ---
PRODUCT="KEOVIL"
IMAGE_NAME="kevilai/keovil:latest"
CONTAINER_NAME="keovil_container"
BIN_DIR="$HOME/.local/bin"
LAUNCHER="$BIN_DIR/keovil"

echo "------------------------------------------------"
echo "  🌐 KEVIL.IO - $PRODUCT AI Engine"
echo "------------------------------------------------"

# 1. PLATFORM & DOCKER CHECKS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "❌ Error: $PRODUCT requires an NVIDIA GPU. macOS is not supported."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker not found. Please install Docker Desktop."
    exit 1
fi

# 2. PREPARE DIRECTORIES
mkdir -p "$BIN_DIR"
mkdir -p "${HOME}/.keovil"
mkdir -p "${HOME}/.keovil_storage/qdrant_storage"
mkdir -p "${HOME}/.cache/huggingface"

# 3. CREATE THE SMART LAUNCHER
echo "⚙️ Registering $PRODUCT command (Password-free)..."

cat <<EOF > "$LAUNCHER"
#!/bin/bash
# Auto-update on every launch
echo "📥 Checking for Keovil Engine updates..."
docker pull $IMAGE_NAME

echo "🔥 Igniting RTX Hardware..."
docker run -it --rm \\
  --name $CONTAINER_NAME \\
  --network=host \\
  --gpus all \\
  --shm-size=4gb \\
  -e APP_MODE=production \\
  -e OLLAMA_HOST=localhost:11434 \\
  -v "${HOME}/.keovil:/root/.keovil" \\
  -v "${HOME}/.keovil_storage:/root/.keovil_storage" \\
  -v "${HOME}/.keovil_storage/qdrant_storage:/qdrant/storage" \\
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \\
  $IMAGE_NAME
EOF

chmod +x "$LAUNCHER"

# 4. PATH VERIFICATION
# Ensure ~/.local/bin is in the user's PATH
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    SHELL_RC="$HOME/.bashrc"
    [[ "$SHELL" == *"zsh"* ]] && SHELL_RC="$HOME/.zshrc"
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$SHELL_RC"
    echo "⚠️ Added $BIN_DIR to your PATH. Please run: source $SHELL_RC"
fi

echo "✅ Success! From now on, just type: keovil"
echo "------------------------------------------------"

# Start it now
exec "$LAUNCHER"
