#!/usr/bin/env bash
#
# DOVA Local Server Launcher
#
# Frees ports 8082/8083 (stops any existing listeners), then starts the DOVA
# API server and the DOVA MCP (HTTP) server together, streams both logs, and
# shuts both down cleanly on Ctrl+C or if either process exits.
#
#   dova serve --port 8082
#   dova mcp serve --transport http --port 8083 --allowed-host mcp1.cavatar.info
#
# Override defaults via env vars, e.g.:
#   API_PORT=9000 MCP_PORT=9001 MCP_ALLOWED_HOST='mcp.example.com' ./scripts/run_servers.sh
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Log functions
log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# Configuration (override via environment)
API_PORT="${API_PORT:-8082}"
MCP_PORT="${MCP_PORT:-8083}"
MCP_TRANSPORT="${MCP_TRANSPORT:-http}"
MCP_ALLOWED_HOST="${MCP_ALLOWED_HOST:-mcp1.cavatar.info}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"

# Stop any process currently LISTENing on a TCP port (graceful TERM, then KILL).
free_port() {
    local port="$1" pids
    pids="$(lsof -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -z "$pids" ]]; then
        log_info "Port ${port} is free."
        return
    fi
    log_info "Stopping existing process(es) on port ${port}: ${pids//$'\n'/ }"
    kill $pids 2>/dev/null || true
    for _ in 1 2 3 4 5 6; do
        sleep 0.5
        pids="$(lsof -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null || true)"
        [[ -z "$pids" ]] && break
    done
    if [[ -n "$pids" ]]; then
        log_info "Force-killing on port ${port}: ${pids//$'\n'/ }"
        kill -9 $pids 2>/dev/null || true
    fi
    log_ok "Port ${port} freed."
}

# Preflight: dova must be on PATH (activate the right venv/conda env first).
if ! command -v dova >/dev/null 2>&1; then
    log_error "'dova' not found on PATH. Activate the environment that has DOVA installed, then re-run."
    exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

API_LOG="$LOG_DIR/dova-api.log"
MCP_LOG="$LOG_DIR/dova-mcp.log"

# Clean shutdown: kill the servers (and the log tail) on exit. Idempotent.
_cleaned=0
cleanup() {
    [[ "$_cleaned" -eq 1 ]] && return
    _cleaned=1
    log_info "Shutting down servers..."
    for pid in "${API_PID:-}" "${MCP_PID:-}" "${TAIL_PID:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    log_ok "Stopped."
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Stop anything already bound to our ports before (re)starting.
free_port "$API_PORT"
free_port "$MCP_PORT"

log_info "Starting DOVA API server on port ${API_PORT}..."
dova serve --port "$API_PORT" >"$API_LOG" 2>&1 &
API_PID=$!

log_info "Starting DOVA MCP server (transport=${MCP_TRANSPORT}, port=${MCP_PORT}, allowed-host='${MCP_ALLOWED_HOST}')..."
dova mcp serve --transport "$MCP_TRANSPORT" --port "$MCP_PORT" --allowed-host "$MCP_ALLOWED_HOST" >"$MCP_LOG" 2>&1 &
MCP_PID=$!

log_ok "API server  PID=${API_PID}  log=${API_LOG}"
log_ok "MCP server  PID=${MCP_PID}  log=${MCP_LOG}"
log_info "Streaming logs — press Ctrl+C to stop both."
echo

# Live combined logs in the foreground.
tail -n +1 -F "$API_LOG" "$MCP_LOG" &
TAIL_PID=$!

# Wait until either server exits, then cleanup() (via EXIT trap) stops the rest.
wait -n "$API_PID" "$MCP_PID"
log_error "A server process exited — shutting the other one down."
