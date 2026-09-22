#!/usr/bin/env bash
# ==============================================================================
# BCSE306L DA-1: Multimodal Cobot Trust Prediction & Calibration System
# Interactive Startup Script
# Authors: Ayushi Singh (24BRS1369), Rakshith Ganjimut (24BRS1301)
# ==============================================================================

set -e

# Terminal Colors
CYAN='\033[0;36m'
TEAL='\033[38;5;43m'
GREEN='\033[0;32m'
GOLD='\033[38;5;220m'
AMBER='\033[0;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

# Default parameters
HOST="0.0.0.0"
PORT="8000"
RELOAD=false
RUN_TESTS=false
OPEN_BROWSER=false

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Print banner
print_banner() {
  clear 2>/dev/null || true
  echo -e "${TEAL}${BOLD}"
  cat << 'EOF'
  ██████╗ ██████╗ ██████╗  ██████╗ ████████╗    ████████╗██████╗ ██╗   ██╗███████╗████████╗
 ██╔════╝██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝    ╚══██╔══╝██╔══██╗██║   ██║██╔════╝╚══██╔══╝
 ██║     ██║   ██║██████╔╝██║   ██║   ██║          ██║   ██████╔╝██║   ██║███████╗   ██║   
 ██║     ██║   ██║██╔══██╗██║   ██║   ██║          ██║   ██╔══██╗██║   ██║╚════██║   ██║   
 ╚██████╗╚██████╔╝██████╔╝╚██████╔╝   ██║          ██║   ██║  ██║╚██████╔╝███████║   ██║   
  ╚═════╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝          ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝   
EOF
  echo -e "${NC}"
  echo -e " ${BOLD}Multimodal Machine Learning for Human Trust Calibration in Cobots${NC}"
  echo -e " ${DIM}Course: BCSE306L (DA-1) | VIT Chennai | Dr. Vijayprabhakaran${NC}"
  echo -e " ${DIM}Authors: Ayushi Singh (24BRS1369) & Rakshith Ganjimut (24BRS1301)${NC}"
  echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────────${NC}"
}

# Parse Command-Line Arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --port) PORT="$2"; shift ;;
    --host) HOST="$2"; shift ;;
    --reload) RELOAD=true ;;
    --test) RUN_TESTS=true ;;
    --open) OPEN_BROWSER=true ;;
    -h|--help)
      echo -e "${BOLD}Usage:${NC} ./start.sh [options]"
      echo ""
      echo -e "${BOLD}Options:${NC}"
      echo -e "  ${CYAN}--port <port>${NC}        Specify port (default: 8000)"
      echo -e "  ${CYAN}--host <host>${NC}        Specify host interface (default: 0.0.0.0)"
      echo -e "  ${CYAN}--reload${NC}             Enable FastAPI hot-reloading for development"
      echo -e "  ${CYAN}--test${NC}               Run pipeline & baseline verification suite before start"
      echo -e "  ${CYAN}--open${NC}               Attempt to automatically open default web browser"
      echo -e "  ${CYAN}-h, --help${NC}           Show this help message and exit"
      echo ""
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}. Use --help for usage details."
      exit 1
      ;;
  esac
  shift
done

print_banner

# Step 1: Detect Python Environment
echo -e "${CYAN}[1/4]${NC} Detecting Python 3 environment..."
if command -v python3 &>/dev/null; then
  PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
  PYTHON_CMD="python"
else
  echo -e "${RED}✘ Error: Python 3 not found! Please install python 3.10+ to proceed.${NC}"
  exit 1
fi

PY_VER=$($PYTHON_CMD --version 2>&1)
echo -e "      ${GREEN}✔${NC} Found ${PY_VER} (${PYTHON_CMD})"

# Step 2: Check Virtual Environment (if exists, activate)
if [[ -f ".venv/bin/activate" ]]; then
  echo -e "${CYAN}[2/4]${NC} Activating local virtual environment (.venv)..."
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo -e "      ${GREEN}✔${NC} Activated virtual environment"
else
  echo -e "${CYAN}[2/4]${NC} Using system Python environment..."
fi

# Step 3: Verify Dependencies
echo -e "${CYAN}[3/4]${NC} Checking core scientific & web dependencies..."
REQUIRED_PKGS=("fastapi" "uvicorn" "torch" "numpy" "scipy" "sklearn")
MISSING_PKGS=()

for pkg in "${REQUIRED_PKGS[@]}"; do
  if ! $PYTHON_CMD -c "import $pkg" &>/dev/null; then
    MISSING_PKGS+=("$pkg")
  fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
  echo -e "      ${AMBER}⚠ Missing packages:${NC} ${MISSING_PKGS[*]}"
  echo -e "      ${CYAN}→ Installing required packages from requirements.txt...${NC}"
  $PYTHON_CMD -m pip install -r requirements.txt
  echo -e "      ${GREEN}✔${NC} All dependencies successfully installed!"
else
  echo -e "      ${GREEN}✔${NC} All dependencies verified (PyTorch, FastAPI, NumPy, SciPy, Scikit-Learn)"
fi

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
  echo -e "\n${PURPLE}${BOLD}[TEST]${NC} Running verification tests..."
  "$SCRIPT_DIR/test.sh" || {
    echo -e "${RED}✘ Tests failed! Aborting server start.${NC}"
    exit 1
  }
fi

# Step 4: Check if port is already in use
echo -e "${CYAN}[4/4]${NC} Checking port availability on ${PORT}..."
PID_ON_PORT=$(lsof -ti:"$PORT" 2>/dev/null || fuser "$PORT/tcp" 2>/dev/null || true)

if [ -n "$PID_ON_PORT" ]; then
  echo -e "      ${AMBER}⚠ Port $PORT is already in use by process PID $PID_ON_PORT.${NC}"
  read -r -p "      Terminate existing process on port $PORT? [Y/n]: " CONFIRM_KILL
  CONFIRM_KILL=${CONFIRM_KILL:-Y}
  if [[ "$CONFIRM_KILL" =~ ^[Yy]$ ]]; then
    kill -9 $PID_ON_PORT 2>/dev/null || true
    sleep 0.8
    echo -e "      ${GREEN}✔${NC} Process terminated. Port $PORT released."
  else
    echo -e "${RED}✘ Cannot start server while port $PORT is occupied. Try --port <another_port>.${NC}"
    exit 1
  fi
else
  echo -e "      ${GREEN}✔${NC} Port $PORT is free and ready."
fi

# Display launch info
echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────────${NC}"
echo -e "${GOLD}${BOLD}  🚀 LAUNCHING 3D COBOT COCKPIT & REST API SERVER${NC}"
echo -e "  ${BOLD}Local URL:${NC}    ${TEAL}${BOLD}http://localhost:${PORT}${NC}"
echo -e "  ${BOLD}Network URL:${NC}  ${TEAL}http://${HOST}:${PORT}${NC}"
echo -e "  ${BOLD}API Docs:${NC}     ${DIM}http://localhost:${PORT}/docs${NC}"
echo -e "  ${BOLD}Press Ctrl+C to safely terminate server.${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────────${NC}\n"

# Optional auto-open browser in background
if [ "$OPEN_BROWSER" = true ]; then
  (
    sleep 1.5
    if command -v xdg-open &>/dev/null; then
      xdg-open "http://localhost:${PORT}" &>/dev/null
    elif command -v open &>/dev/null; then
      open "http://localhost:${PORT}" &>/dev/null
    fi
  ) &
fi

# Handle SIGINT and SIGTERM gracefully
cleanup() {
  echo -e "\n${AMBER}Stopping Cobot Trust Server...${NC}"
  exit 0
}
trap cleanup SIGINT SIGTERM

# Launch uvicorn
if [ "$RELOAD" = true ]; then
  exec $PYTHON_CMD -m uvicorn server:app --host "$HOST" --port "$PORT" --reload
else
  exec $PYTHON_CMD -m uvicorn server:app --host "$HOST" --port "$PORT"
fi
