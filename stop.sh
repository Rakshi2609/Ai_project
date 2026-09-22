#!/usr/bin/env bash
# ==============================================================================
# BCSE306L DA-1: Stop Cobot Trust Prediction Server
# ==============================================================================

PORT="${1:-8000}"

# Colors
GREEN='\033[0;32m'
AMBER='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}[Cobot Trust System]${NC} Searching for processes on port ${PORT}..."

PIDS=$(lsof -ti:"$PORT" 2>/dev/null || fuser "$PORT/tcp" 2>/dev/null || true)

if [ -z "$PIDS" ]; then
  # Also check by process name in case
  PIDS=$(pgrep -f "server:app" || true)
fi

if [ -n "$PIDS" ]; then
  echo -e "${AMBER}Found active server process(es): ${PIDS}${NC}"
  kill -15 $PIDS 2>/dev/null || kill -9 $PIDS 2>/dev/null || true
  sleep 0.5
  echo -e "${GREEN}✔ Server successfully stopped.${NC}"
else
  echo -e "${GREEN}✔ No active server running on port ${PORT}.${NC}"
fi
