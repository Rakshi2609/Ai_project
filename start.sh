#!/usr/bin/env bash
# ==============================================================================
# BCSE306L DA-1: Universal Robots UR5 Multimodal Trust Prediction System
# Startup Controller (FastAPI Backend + Next.js App Router Dashboard)
# ==============================================================================

set -e

# Default Configuration
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
PORT_NEXT="${PORT_NEXT:-3000}"
RELOAD=false
RUN_TESTS=false
OPEN_BROWSER=false
BACKEND_ONLY=false

# ANSI Terminal Colors
CYAN='\033[0;36m'
TEAL='\033[38;5;45m'
GREEN='\033[0;32m'
AMBER='\033[0;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
GOLD='\033[38;5;220m'
NC='\033[0m' # No Color

print_banner() {
  echo -e "${TEAL}"
  cat << "EOF"
  ██████╗ ██████╗ ██████╗  ██████╗ ████████╗    ████████╗██████╗ ██╗   ██╗███████╗████████╗
 ██╔════╝██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝    ╚══██╔══╝██╔══██╗██║   ██║██╔════╝╚══██╔══╝
 ██║     ██║   ██║██████╔╝██║   ██║   ██║          ██║   ██████╔╝██║   ██║███████╗   ██║   
 ██║     ██║   ██║██╔══██╗██║   ██║   ██║          ██║   ██╔══██╗██║   ██║╚════██║   ██║   
 ╚██████╗╚██████╔╝██████╔╝╚██████╔╝   ██║          ██║   ██║  ██║╚██████╔╝███████║   ██║   
  ╚═════╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝          ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝   
EOF
  echo -e "${CYAN}${BOLD}     MULTIMODAL HUMAN-ROBOT COLLABORATION TRUST PREDICTION SYSTEM (BCSE306L DA-1)${NC}"
  echo -e "${DIM}     VIT Chennai • Collaborative Robotics & Multimodal Machine Learning Laboratory${NC}\n"
}

# Parse Command-Line Arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --port) PORT="$2"; shift ;;
    --host) HOST="$2"; shift ;;
    --port-next) PORT_NEXT="$2"; shift ;;
    --backend-only) BACKEND_ONLY=true ;;
    --reload) RELOAD=true ;;
    --test) RUN_TESTS=true ;;
    --open) OPEN_BROWSER=true ;;
    -h|--help)
      echo -e "${BOLD}Usage:${NC} ./start.sh [options]"
      echo ""
      echo -e "${BOLD}Options:${NC}"
      echo -e "  ${CYAN}--port <port>${NC}        Specify FastAPI port (default: 8000)"
      echo -e "  ${CYAN}--port-next <port>${NC}   Specify Next.js port (default: 3000)"
      echo -e "  ${CYAN}--host <host>${NC}        Specify host interface (default: 0.0.0.0)"
      echo -e "  ${CYAN}--backend-only${NC}       Launch only the FastAPI backend service"
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
  $PYTHON_CMD -m pip install -q -r requirements.txt
  echo -e "      ${GREEN}✔${NC} All dependencies installed successfully."
else
  echo -e "      ${GREEN}✔${NC} All Python dependencies verified (PyTorch, FastAPI, NumPy, SciPy, Scikit-learn)."
fi

# Step 4: Optional Pre-Flight Test Suite
if [ "$RUN_TESTS" = true ]; then
  echo -e "\n${CYAN}[4/4]${NC} Running pre-flight verification test suite..."
  if ./test.sh; then
    echo -e "      ${GREEN}✔${NC} Pre-flight verification passed 100%."
  else
    echo -e "      ${RED}✘ Error: Verification tests failed. Aborting startup.${NC}"
    exit 1
  fi
else
  echo -e "${CYAN}[4/4]${NC} Pre-flight test skipped (pass ${BOLD}--test${NC} to run before start)."
fi

# Check for Port Collisions on port 8000
echo -e "\n${CYAN}[Diagnostics]${NC} Checking network interfaces..."
OCCUPIED=$(lsof -ti:"$PORT" 2>/dev/null || fuser "$PORT/tcp" 2>/dev/null || true)
if [ -n "$OCCUPIED" ]; then
  echo -e "      ${AMBER}⚠ Notice: Port $PORT is occupied by PID(s): $OCCUPIED${NC}"
  echo -e "      ${CYAN}→ Terminating stale process on port $PORT...${NC}"
  kill -15 $OCCUPIED 2>/dev/null || kill -9 $OCCUPIED 2>/dev/null || true
  sleep 1
fi
echo -e "      ${GREEN}✔${NC} FastAPI port $PORT is free and ready."

# Display launch info
echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────────${NC}"
echo -e "${GOLD}${BOLD}  🚀 LAUNCHING MULTIMODAL COBOT TRUST SYSTEM${NC}"
echo -e "  ${BOLD}Next.js Modern Dashboard:${NC}  ${TEAL}${BOLD}http://localhost:${PORT_NEXT}${NC}"
echo -e "  ${BOLD}FastAPI Backend / REST API:${NC} ${TEAL}http://localhost:${PORT}${NC}"
echo -e "  ${BOLD}FastAPI Swagger Docs:${NC}       ${DIM}http://localhost:${PORT}/docs${NC}"
echo -e "  ${BOLD}Press Ctrl+C to safely terminate all services.${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────────${NC}\n"

# Cleanup trap for graceful shutdown
API_PID=""
NEXT_PID=""

cleanup() {
  echo -e "\n${AMBER}Shutting down Cobot Trust System services...${NC}"
  if [ -n "$API_PID" ]; then kill -15 "$API_PID" 2>/dev/null || true; fi
  if [ -n "$NEXT_PID" ]; then kill -15 "$NEXT_PID" 2>/dev/null || true; fi
  exit 0
}
trap cleanup SIGINT SIGTERM

# 1. Start FastAPI Backend in background
if [ "$RELOAD" = true ]; then
  $PYTHON_CMD -m uvicorn server:app --host "$HOST" --port "$PORT" --reload &
else
  $PYTHON_CMD -m uvicorn server:app --host "$HOST" --port "$PORT" &
fi
API_PID=$!

# 2. Start Next.js Frontend
if [ "$BACKEND_ONLY" = false ] && [ -d "dashboard" ]; then
  (
    cd dashboard
    if [ -d ".next" ]; then
      npm run start -- -p "$PORT_NEXT"
    else
      npm run dev -- -p "$PORT_NEXT"
    fi
  ) &
  NEXT_PID=$!
fi

# Optional auto-open browser in background
if [ "$OPEN_BROWSER" = true ]; then
  (
    sleep 2
    URL="http://localhost:${PORT_NEXT}"
    if command -v xdg-open &>/dev/null; then
      xdg-open "$URL" &>/dev/null
    elif command -v open &>/dev/null; then
      open "$URL" &>/dev/null
    fi
  ) &
fi

# Wait for both processes
if [ -n "$NEXT_PID" ]; then
  wait "$API_PID" "$NEXT_PID"
else
  wait "$API_PID"
fi
