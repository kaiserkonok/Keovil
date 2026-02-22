#!/bin/bash

# --- 0. UI Essentials ---
export TERM=xterm-256color
GREEN='\033[38;5;82m'   # Neon Green
BLUE='\033[38;5;39m'    # Deep Sky Blue
GRAY='\033[38;5;244m'   # Slate Gray
RED='\033[38;5;196m'    # Vivid Red
BOLD='\033[1m'
NC='\033[0m'

# Professional Symbols
SYMB_PULSE="[•]"
SYMB_OK="✔"

cleanup() {
    echo -e "\n${RED}Stopping Keovil Engine...${NC}"
    kill $(jobs -p) 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM

# Original Dash Spinner with refined colors
spinner() {
    local pid=$1
    local msg=$2
    local delay=0.1
    local spinstr='|/-\'
    while ps -p $pid > /dev/null; do
        local temp=${spinstr#?}
        printf "\r${BLUE}${BOLD}[%c]${NC} ${GRAY}%s...${NC}" "$spinstr" "$msg"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
    done
    printf "\r\033[K"
}

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

# New helper to ensure model is actually in VRAM
load_model_sync() {
    # stream: false makes curl wait for the full load and a response
    curl -s -X POST http://localhost:11434/api/generate \
         -d "{\"model\": \"qwen2.5-coder:7b-instruct\", \"prompt\": \"hi\", \"stream\": false}" > /dev/null
}

clear
echo -e "${BLUE}${BOLD}${SYMB_PULSE} KEVIL.IO | Keovil v1.0${NC}"

# --- 1. Background Services ---
export QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
/usr/bin/qdrant --uri http://0.0.0.0:6333 > /dev/null 2>&1 &
QDRANT_PID=$!

export OLLAMA_MODELS=/root/.ollama/models
/usr/bin/ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!

# --- 2. Professional Wait Sequence ---

wait_for_200 "http://localhost:6333/healthz" "$QDRANT_PID" &
spinner $! "Syncing Neural Database"
echo -e "${BLUE}◈${NC}  Database      ${GREEN}${BOLD}Ready${NC}"

wait_for_200 "http://localhost:11434/api/tags" "$OLLAMA_PID" &
spinner $! "Warming GPU Cores"
echo -e "${BLUE}⌬${NC}  AI Service    ${GREEN}${BOLD}Online${NC}"

# --- 3. Pre-warm Model into VRAM (FIXED) ---
# We now wait for the model to actually respond before moving on.
load_model_sync &
spinner $! "Charging VRAM Cache"
echo -e "${BLUE}⚡${NC}  GPU Cache     ${GREEN}${BOLD}Charged${NC}"

# --- 4. Flask Server Ignition ---
python3 src/keovil_web/app.py &
APP_PID=$!

wait_for_200 "http://localhost:5000" "$APP_PID" &
spinner $! "Launching Interface"

if ! ps -p $APP_PID > /dev/null; then
    echo -e "${RED}✖  Interface     Crashed${NC}"
    exit 1
fi
echo -e "${BLUE}🌐${NC}  Interface     ${GREEN}${BOLD}Active${NC}"

# --- 5. Final Clean Dashboard ---
echo -e "${GREEN}${BOLD}${SYMB_OK} ALL SYSTEMS OPERATIONAL${NC}"
echo -e "🔗 ACCESS LINK: ${BLUE}${BOLD}http://localhost:5000${NC}"
echo -e "${GRAY}Press [Ctrl+C] to shutdown engine safely.${NC}"

wait $APP_PID