#!/bin/bash

# --- 0. UI Essentials ---
export TERM=xterm-256color
GREEN='\033[38;5;82m'   # Neon Green
BLUE='\033[38;5;39m'    # Deep Sky Blue
GRAY='\033[38;5;244m'   # Slate Gray
RED='\033[38;5;196m'    # Vivid Red
BOLD='\033[1m'
NC='\033[0m'

SYMB_PULSE="[•]"
SYMB_OK="✔"

cleanup() {
    echo -e "\n${RED}Stopping Keovil Engine...${NC}"
    kill $(jobs -p) 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM

spinner() {
    local pid=$1
    local msg=$2
    local delay=0.1
    local spinstr='|/-\'
    while ps -p $pid > /dev/null 2>/dev/null; do
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
    local max_attempts=${3:-30}
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        local status=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
        if [ "$status" -eq 200 ]; then return 0; fi
        if [ -n "$target_pid" ] && ! ps -p $target_pid > /dev/null 2>/dev/null; then return 1; fi
        attempt=$((attempt + 1))
        sleep 1
    done
    return 1
}

clear
echo -e "${BLUE}${BOLD}${SYMB_PULSE} KEVIL.IO | Keovil v1.0${NC}"

# --- 1. Background Services ---
export QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
/usr/bin/qdrant --uri http://0.0.0.0:6333 > /dev/null 2>&1 &
QDRANT_PID=$!

wait_for_200 "http://localhost:6333/healthz" "$QDRANT_PID" 20
if [ $? -eq 0 ]; then
    echo -e "${BLUE}◈${NC}  Database      ${GREEN}${BOLD}Ready${NC}"
else
    echo -e "${RED}✖  Database      Failed${NC}"
    exit 1
fi

# --- 2. Detect and connect to Ollama ---
# Try multiple methods to find host's Ollama
echo -e "${BLUE}${BOLD}[*]${NC} Detecting Ollama..."

OLLAMA_URLS=(
    "http://host.docker.internal:11434"  # Docker Desktop (Mac/Windows)
    "http://172.17.0.1:11434"            # Linux docker bridge
    "http://127.0.0.1:11434"             # Localhost / --network=host
    "http://localhost:11434"              # Localhost fallback
)

OLLAMA_FOUND=0
OLLAMA_URL=""

for url in "${OLLAMA_URLS[@]}"; do
    echo -e "${GRAY}  Trying $url...${NC}"
    if wait_for_200 "$url/api/tags" "" 3 2>/dev/null; then
        OLLAMA_URL="$url"
        OLLAMA_FOUND=1
        break
    fi
done

if [ $OLLAMA_FOUND -eq 1 ]; then
    echo -e "${BLUE}⌬${NC}  Ollama       ${GREEN}${BOLD}Online ($OLLAMA_URL)${NC}"
    export OLLAMA_HOST=$OLLAMA_URL
else
    echo -e "${RED}✖${NC}  Ollama       ${RED}Not Found${NC}"
    echo ""
    echo -e "${GRAY}Please ensure Ollama is running on your host machine.${NC}"
    echo -e "${GRAY}Install: https://ollama.com${NC}"
    echo -e "${GRAY}Then run: ollama pull qwen2.5-coder:7b-instruct${NC}"
    echo ""
    echo -e "${YELLOW}If using Docker, ensure --add-host is set or use --network=host${NC}"
    exit 1
fi

# --- 3. Flask Server Ignition ---
python3 src/keovil_web/app.py &
APP_PID=$!

wait_for_200 "http://localhost:5000" "$APP_PID" 30
if [ $? -eq 0 ]; then
    echo -e "${BLUE}🌐${NC}  Interface     ${GREEN}${BOLD}Active${NC}"
else
    if ! ps -p $APP_PID > /dev/null 2>/dev/null; then
        echo -e "${RED}✖  Interface     Crashed${NC}"
    else
        echo -e "${RED}✖  Interface     Timeout${NC}"
    fi
    exit 1
fi

# --- 4. Final Clean Dashboard ---
echo -e "${GREEN}${BOLD}${SYMB_OK} ALL SYSTEMS OPERATIONAL${NC}"
echo -e "🔗 ACCESS LINK: ${BLUE}${BOLD}http://localhost:5000${NC}"
echo -e "${GRAY}Press [Ctrl+C] to shutdown engine safely.${NC}"

wait $APP_PID
