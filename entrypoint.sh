#!/bin/bash

# --- 0. UI Essentials ---
export TERM=xterm-256color
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# CLEANUP TRAP: Kill all background processes on exit (Ollama, Qdrant, etc)
cleanup() {
    echo -e "\n${RED}Stopping Keovil Engine...${NC}"
    # Kill background jobs started by this script
    kill $(jobs -p) 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM

# Spinner: Docker-safe version (no tput)
spinner() {
    local pid=$1
    local msg=$2
    local delay=0.1
    local spinstr='|/-\'
    while ps -p $pid > /dev/null; do
        local temp=${spinstr#?}
        printf "\r${BLUE}${BOLD}[%c]${NC} %s" "$spinstr" "$msg"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
    done
    printf "\r\033[K" # Clear the line after finish
}

# Robust checker: Waits for 200 OK or process death
wait_for_200() {
    local url=$1
    local target_pid=$2
    while true; do
        local status=$(curl -s -o /dev/null -w "%{http_code}" "$url" || echo "000")
        if [ "$status" -eq 200 ]; then exit 0; fi
        if ! ps -p $target_pid > /dev/null; then exit 1; fi
        sleep 0.5
    done
}

clear
echo -e "${BLUE}${BOLD}🚀 KEVIL.IO | Keovil v1.0${NC}"
echo -e "-------------------------------------------"

# --- 1. Background Services ---
export QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
/usr/bin/qdrant --uri http://0.0.0.0:6333 > /dev/null 2>&1 &
QDRANT_PID=$!

export OLLAMA_MODELS=/root/.ollama/models
/usr/bin/ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!

# --- 2. Professional Wait Sequence ---

wait_for_200 "http://localhost:6333/healthz" "$QDRANT_PID" &
spinner $! "Syncing Neural Database..."
echo -e "${GREEN}✅ Database: READY${NC}"

wait_for_200 "http://localhost:11434/api/tags" "$OLLAMA_PID" &
spinner $! "Warming GPU Cores..."
echo -e "${GREEN}✅ AI Service: ONLINE${NC}"

# --- 3. Pre-warm Model into VRAM ---
( curl -s -X POST http://localhost:11434/api/generate \
     -d "{\"model\": \"qwen2.5-coder:7b-instruct\", \"keep_alive\": -1}" > /dev/null ) &
spinner $! "Charging Model into VRAM..."
echo -e "${GREEN}✅ GPU Cache: LOADED${NC}"

# --- 4. Flask Server Ignition ---
# NO REDIRECTION HERE: Let Python's _shield() handle the terminal
python3 src/keovil_web/app.py &
APP_PID=$!

wait_for_200 "http://localhost:5000" "$APP_PID" &
spinner $! "Launching Web Interface..."

# Final check if app is actually alive
if ! ps -p $APP_PID > /dev/null; then
    echo -e "${RED}❌ Interface: CRASHED${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Interface: ACTIVE${NC}"

# --- 5. Final Clean Dashboard ---
echo -e "-------------------------------------------"
echo -e "${GREEN}${BOLD}✔ ALL SYSTEMS OPERATIONAL${NC}"
echo -e "🔗 ACCESS LINK: ${BLUE}${BOLD}http://localhost:5000${NC}"
echo -e "-------------------------------------------"
echo -e "Press [Ctrl+C] to shutdown engine safely."

# Bring the python process to the foreground management
wait $APP_PID