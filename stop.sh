#!/usr/bin/env bash
# ==============================================================================
# BCSE306L DA-1: Stop Cobot Trust Prediction Servers (FastAPI & Next.js)
# ==============================================================================

PORT_API="${1:-8000}"
PORT_NEXT="${2:-3000}"

# Colors
GREEN='\033[0;32m'
AMBER='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}[Cobot Trust System]${NC} Terminating active services..."

# Terminate FastAPI Backend
PIDS_API=$(lsof -ti:"$PORT_API" 2>/dev/null || fuser "$PORT_API/tcp" 2>/dev/null || pgrep -f "server:app" || true)
if [ -n "$PIDS_API" ]; then
  echo -e "${AMBER}Stopping FastAPI backend on port ${PORT_API} (PIDs: ${PIDS_API})...${NC}"
  kill -15 $PIDS_API 2>/dev/null || kill -9 $PIDS_API 2>/dev/null || true
  echo -e "${GREEN}✔ Backend stopped.${NC}"
else
  echo -e "${GREEN}✔ No backend running on port ${PORT_API}.${NC}"
fi

# Terminate Next.js Dashboard
PIDS_NEXT=$(lsof -ti:"$PORT_NEXT" 2>/dev/null || fuser "$PORT_NEXT/tcp" 2>/dev/null || pgrep -f "next-server" || true)
if [ -n "$PIDS_NEXT" ]; then
  echo -e "${AMBER}Stopping Next.js dashboard on port ${PORT_NEXT} (PIDs: ${PIDS_NEXT})...${NC}"
  kill -15 $PIDS_NEXT 2>/dev/null || kill -9 $PIDS_NEXT 2>/dev/null || true
  echo -e "${GREEN}✔ Next.js dashboard stopped.${NC}"
else
  echo -e "${GREEN}✔ No Next.js server running on port ${PORT_NEXT}.${NC}"
fi
