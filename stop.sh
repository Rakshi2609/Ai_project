#!/usr/bin/env bash
# Wrapper delegating to unified runner: ./run.sh stop
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run.sh" stop "$@"
