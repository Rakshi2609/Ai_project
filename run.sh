#!/usr/bin/env bash
# ==============================================================================
# BCSE306L DA-1: Universal Robots UR5 Multimodal Trust Prediction System
# All-In-One Unified Controller Script (start, stop, restart, test, status)
# ==============================================================================

set -e

# Port Defaults
PORT_API="${PORT_API:-8000}"
PORT_NEXT="${PORT_NEXT:-3000}"
HOST="${HOST:-0.0.0.0}"
DEV_MODE=false
OPEN_BROWSER=false

# Terminal Formatting & Colors
CYAN='\033[0;36m'
TEAL='\033[38;5;45m'
GREEN='\033[0;32m'
AMBER='\033[0;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
GOLD='\033[38;5;220m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
  echo -e "${CYAN}${BOLD}     ALL-IN-ONE MULTIMODAL HUMAN-ROBOT TRUST SYSTEM (BCSE306L DA-1)${NC}"
  echo -e "${DIM}     VIT Chennai • Real-Time Face Expression, Voice Prosody & 3D UR5 Cobot${NC}\n"
}

# ------------------------------------------------------------------------------
# Action: STOP
# ------------------------------------------------------------------------------
stop_services() {
  echo -e "${CYAN}${BOLD}[Cobot Trust System]${NC} Stopping active services..."
  
  # Terminate FastAPI (port 8000)
  PIDS_API=$(lsof -ti:"$PORT_API" 2>/dev/null || fuser "$PORT_API/tcp" 2>/dev/null || pgrep -f "server:app" || true)
  if [ -n "$PIDS_API" ]; then
    echo -e "  ${AMBER}Stopping FastAPI backend on port ${PORT_API} (PID: ${PIDS_API})...${NC}"
    kill -15 $PIDS_API 2>/dev/null || kill -9 $PIDS_API 2>/dev/null || true
    echo -e "  ${GREEN}✔ Backend stopped.${NC}"
  else
    echo -e "  ${DIM}FastAPI backend not running.${NC}"
  fi

  # Terminate Next.js (port 3000)
  PIDS_NEXT=$(lsof -ti:"$PORT_NEXT" 2>/dev/null || fuser "$PORT_NEXT/tcp" 2>/dev/null || pgrep -f "next-server" || true)
  if [ -n "$PIDS_NEXT" ]; then
    echo -e "  ${AMBER}Stopping Next.js dashboard on port ${PORT_NEXT} (PID: ${PIDS_NEXT})...${NC}"
    kill -15 $PIDS_NEXT 2>/dev/null || kill -9 $PIDS_NEXT 2>/dev/null || true
    echo -e "  ${GREEN}✔ Next.js dashboard stopped.${NC}"
  else
    echo -e "  ${DIM}Next.js dashboard not running.${NC}"
  fi
}

# ------------------------------------------------------------------------------
# Action: STATUS
# ------------------------------------------------------------------------------
check_status() {
  echo -e "${CYAN}${BOLD}[Service Status Check]${NC}"
  
  # FastAPI
  if lsof -ti:"$PORT_API" &>/dev/null || curl -s "http://127.0.0.1:${PORT_API}/docs" &>/dev/null; then
    echo -e "  FastAPI Backend (port ${PORT_API}):   ${GREEN}${BOLD}● ONLINE${NC} (http://localhost:${PORT_API})"
  else
    echo -e "  FastAPI Backend (port ${PORT_API}):   ${RED}${BOLD}○ OFFLINE${NC}"
  fi

  # Next.js
  if lsof -ti:"$PORT_NEXT" &>/dev/null || curl -s "http://127.0.0.1:${PORT_NEXT}" &>/dev/null; then
    echo -e "  Next.js Dashboard (port ${PORT_NEXT}): ${GREEN}${BOLD}● ONLINE${NC} (http://localhost:${PORT_NEXT})"
  else
    echo -e "  Next.js Dashboard (port ${PORT_NEXT}): ${RED}${BOLD}○ OFFLINE${NC}"
  fi

  # Hardware Diagnostics
  echo -e "  -------------------------------------------------"
  if ls /dev/video* 1>/dev/null 2>&1; then
    VIDEOS=$(ls /dev/video* 2>/dev/null | tr '\n' ' ')
    echo -e "  Video Stream Hardware:                ${GREEN}${BOLD}✔ DETECTED${NC} (${VIDEOS})"
  else
    echo -e "  Video Stream Hardware:                ${AMBER}○ NO CAMERA DETECTED (/dev/video*)${NC}"
  fi
}

# ------------------------------------------------------------------------------
# Action: TEST
# ------------------------------------------------------------------------------
run_tests() {
  echo -e "${CYAN}${BOLD}[Verification Test Suite]${NC} Executing automated test suite..."
  if [ -f "tests/test_routes.py" ]; then
    echo -e "  ${CYAN}Running Multi-Route Verification...${NC}"
    python3 tests/test_routes.py || true
  fi
  if [ -f "./test.sh" ]; then
    ./test.sh
  else
    python3 -c "import trust_ai_pipeline; trust_ai_pipeline.demo_system_run()"
    python3 evaluation.py
    python3 retrainer.py
  fi
}

# ------------------------------------------------------------------------------
# Action: START (Default)
# ------------------------------------------------------------------------------
start_services() {
  print_banner

  # 1. Check Python 3
  echo -e "${CYAN}[1/5]${NC} Checking Python 3..."
  if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
  elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
  else
    echo -e "${RED}✘ Error: Python 3 is required. Please install python 3.10+${NC}"
    exit 1
  fi
  echo -e "      ${GREEN}✔${NC} Found $($PYTHON_CMD --version 2>&1)"

  # Check Virtual Environment
  if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo -e "      ${GREEN}✔${NC} Activated .venv"
  fi

  # 2. Check Python Dependencies
  echo -e "${CYAN}[2/5]${NC} Verifying scientific AI dependencies..."
  REQUIRED_PKGS=("fastapi" "uvicorn" "torch" "numpy" "scipy" "sklearn")
  MISSING=()
  for pkg in "${REQUIRED_PKGS[@]}"; do
    if ! $PYTHON_CMD -c "import $pkg" &>/dev/null; then
      MISSING+=("$pkg")
    fi
  done
  if [ ${#MISSING[@]} -gt 0 ]; then
    echo -e "      ${AMBER}Installing missing packages:${NC} ${MISSING[*]}"
    $PYTHON_CMD -m pip install -q -r requirements.txt
  fi
  echo -e "      ${GREEN}✔${NC} PyTorch, FastAPI, NumPy, SciPy & Scikit-learn verified."

  # 3. Check Node.js and Dashboard Build
  echo -e "${CYAN}[3/5]${NC} Checking Node.js and Next.js frontend..."
  if ! command -v node &>/dev/null; then
    echo -e "${RED}✘ Error: Node.js is required for Next.js dashboard.${NC}"
    exit 1
  fi
  echo -e "      ${GREEN}✔${NC} Found Node $(node -v)"

  if [ -d "dashboard" ]; then
    if [ ! -d "dashboard/node_modules" ]; then
      echo -e "      ${AMBER}Installing dashboard dependencies...${NC}"
      (cd dashboard && npm install --silent)
    fi
    if [ ! -d "dashboard/.next" ]; then
      echo -e "      ${AMBER}Building Next.js production bundle...${NC}"
      (cd dashboard && npm run build)
    fi
    echo -e "      ${GREEN}✔${NC} Next.js dashboard ready."
  fi

  # 4. Clean up any existing port collisions
  echo -e "${CYAN}[4/5]${NC} Preparing network ports..."
  PIDS_OLD=$(lsof -ti:"$PORT_API","$PORT_NEXT" 2>/dev/null || fuser "$PORT_API/tcp" "$PORT_NEXT/tcp" 2>/dev/null || true)
  if [ -n "$PIDS_OLD" ]; then
    echo -e "      ${AMBER}Terminating stale processes on ports ${PORT_API}/${PORT_NEXT}...${NC}"
    kill -15 $PIDS_OLD 2>/dev/null || kill -9 $PIDS_OLD 2>/dev/null || true
    sleep 0.8
  fi
  echo -e "      ${GREEN}✔${NC} Ports $PORT_API (FastAPI) and $PORT_NEXT (Next.js) are clear."

  # 5. Launch Both Servers
  echo -e "${CYAN}[5/5]${NC} Starting servers..."

  API_PID=""
  NEXT_PID=""

  cleanup_all() {
    echo -e "\n${AMBER}Gracefully terminating services...${NC}"
    if [ -n "$API_PID" ]; then kill -15 "$API_PID" 2>/dev/null || true; fi
    if [ -n "$NEXT_PID" ]; then kill -15 "$NEXT_PID" 2>/dev/null || true; fi
    exit 0
  }
  trap cleanup_all SIGINT SIGTERM

  # Start FastAPI Backend
  if [ "$DEV_MODE" = true ]; then
    $PYTHON_CMD -m uvicorn server:app --host "$HOST" --port "$PORT_API" --reload &
  else
    $PYTHON_CMD -m uvicorn server:app --host "$HOST" --port "$PORT_API" &
  fi
  API_PID=$!

  # Start Next.js Frontend
  if [ -d "dashboard" ]; then
    (
      cd dashboard
      if [ "$DEV_MODE" = true ]; then
        npm run dev -- -p "$PORT_NEXT"
      else
        npm run start -- -p "$PORT_NEXT"
      fi
    ) &
    NEXT_PID=$!
  fi

  # Display Dashboard URLs
  echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────────${NC}"
  echo -e "${GOLD}${BOLD}  ✨ COBOT TRUST SYSTEM ONLINE & READY${NC}"
  echo -e ""
  echo -e "  ${BOLD}Next.js Full Dashboard:${NC}     ${TEAL}${BOLD}http://localhost:${PORT_NEXT}${NC}"
  echo -e "    ├─ Live Cockpit:          ${DIM}http://localhost:${PORT_NEXT}/${NC}"
  echo -e "    ├─ Real Biometric Capture: ${DIM}http://localhost:${PORT_NEXT}/capture${NC}"
  echo -e "    ├─ 3D Cobot Digital Twin: ${DIM}http://localhost:${PORT_NEXT}/robot${NC}"
  echo -e "    └─ Benchmarks & Studio:   ${DIM}http://localhost:${PORT_NEXT}/analytics${NC}"
  echo -e ""
  echo -e "  ${BOLD}FastAPI Backend / REST API:${NC}  ${TEAL}http://localhost:${PORT_API}${NC}"
  echo -e "  ${BOLD}Interactive Swagger Docs:${NC}    ${DIM}http://localhost:${PORT_API}/docs${NC}"
  echo -e ""
  echo -e "  ${DIM}Press Ctrl+C to terminate both servers safely.${NC}"
  echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────────${NC}\n"

  # Optional auto open browser
  if [ "$OPEN_BROWSER" = true ]; then
    (
      sleep 1.5
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
}

# ------------------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------------------
COMMAND="${1:-start}"

case "$COMMAND" in
  stop)
    stop_services
    ;;
  restart)
    stop_services
    sleep 1
    start_services
    ;;
  status)
    check_status
    ;;
  test)
    run_tests
    ;;
  dev)
    DEV_MODE=true
    start_services
    ;;
  start)
    shift || true
    while [[ "$#" -gt 0 ]]; do
      case $1 in
        --open) OPEN_BROWSER=true ;;
        --dev) DEV_MODE=true ;;
        --port-api) PORT_API="$2"; shift ;;
        --port-next) PORT_NEXT="$2"; shift ;;
      esac
      shift
    done
    start_services
    ;;
  help|-h|--help)
    echo -e "${BOLD}Usage:${NC} ./run.sh [command] [options]"
    echo ""
    echo -e "${BOLD}Commands:${NC}"
    echo -e "  ${CYAN}./run.sh${NC}          Start everything (FastAPI backend + Next.js dashboard)"
    echo -e "  ${CYAN}./run.sh dev${NC}      Start everything in hot-reloading development mode"
    echo -e "  ${CYAN}./run.sh stop${NC}     Stop all running services (ports 8000 & 3000)"
    echo -e "  ${CYAN}./run.sh restart${NC}  Stop and restart all services"
    echo -e "  ${CYAN}./run.sh status${NC}   Check health status of both services"
    echo -e "  ${CYAN}./run.sh test${NC}     Run complete verification test suite"
    echo ""
    echo -e "${BOLD}Options:${NC}"
    echo -e "  ${CYAN}--open${NC}            Automatically open browser to Next.js dashboard"
    echo -e "  ${CYAN}--port-api <p>${NC}    Set custom port for FastAPI (default: 8000)"
    echo -e "  ${CYAN}--port-next <p>${NC}   Set custom port for Next.js (default: 3000)"
    echo ""
    ;;
  *)
    # Default to start
    start_services
    ;;
esac
