#!/usr/bin/env bash
# llm-litmus: single entry point. Sets up backend + frontend and opens the dashboard.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_PORT=8000
FRONTEND_PORT=3000

RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m'

err() { echo -e "${RED}error:${NC} $1" >&2; }
warn() { echo -e "${YELLOW}warning:${NC} $1" >&2; }
info() { echo -e "${GREEN}==>${NC} $1"; }

hash_of() {
    # portable sha256 of the given file(s)
    if command -v sha256sum >/dev/null 2>&1; then
        cat "$@" | sha256sum | cut -d' ' -f1
    else
        cat "$@" | shasum -a 256 | cut -d' ' -f1
    fi
}

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        version="$("$candidate" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
        major="${version%%.*}"
        minor="${version##*.}"
        if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    err "Python 3.11+ is required but was not found."
    echo "  Install it from https://www.python.org/downloads/ or via pyenv, then re-run ./run.sh." >&2
    exit 1
fi
info "Using $($PYTHON_BIN --version) ($PYTHON_BIN)"

if ! command -v node >/dev/null 2>&1; then
    err "Node.js was not found."
    echo "  Install Node.js 18+ from https://nodejs.org, then re-run ./run.sh." >&2
    exit 1
fi
NODE_MAJOR="$(node -e 'console.log(process.versions.node.split(".")[0])')"
if [ "$NODE_MAJOR" -lt 18 ]; then
    err "Node.js 18+ is required (found $(node --version))."
    echo "  Upgrade from https://nodejs.org, then re-run ./run.sh." >&2
    exit 1
fi
info "Using Node $(node --version)"

if ! command -v npm >/dev/null 2>&1; then
    err "npm was not found (usually ships with Node.js)."
    exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
    err "Ollama was not found."
    echo "  Install it from https://ollama.com, pull a model (e.g. 'ollama pull llama3'), then re-run ./run.sh." >&2
    exit 1
fi
if ! curl -s -m 2 "http://localhost:11434/api/tags" >/dev/null 2>&1; then
    warn "Ollama is installed but doesn't seem to be running. Start it with 'ollama serve' before running a benchmark."
else
    info "Ollama is running"
fi

# ---------------------------------------------------------------------------
# Backend setup
# ---------------------------------------------------------------------------

if [ ! -d "$BACKEND_DIR/.venv" ]; then
    info "Creating Python virtual environment..."
    "$PYTHON_BIN" -m venv "$BACKEND_DIR/.venv"
fi

# shellcheck disable=SC1091
source "$BACKEND_DIR/.venv/bin/activate"

REQ_HASH_FILE="$BACKEND_DIR/.venv/.requirements.hash"
CURRENT_HASH="$(hash_of "$BACKEND_DIR/requirements.txt")"
if [ ! -f "$REQ_HASH_FILE" ] || [ "$(cat "$REQ_HASH_FILE")" != "$CURRENT_HASH" ]; then
    info "Installing backend dependencies..."
    pip install -q -r "$BACKEND_DIR/requirements.txt"
    echo "$CURRENT_HASH" > "$REQ_HASH_FILE"
else
    info "Backend dependencies already up to date"
fi

# ---------------------------------------------------------------------------
# Frontend setup
# ---------------------------------------------------------------------------

PKG_HASH_FILE="$FRONTEND_DIR/node_modules/.package.hash"
CURRENT_PKG_HASH="$(hash_of "$FRONTEND_DIR/package.json")"
if [ ! -d "$FRONTEND_DIR/node_modules" ] || [ ! -f "$PKG_HASH_FILE" ] || [ "$(cat "$PKG_HASH_FILE" 2>/dev/null || true)" != "$CURRENT_PKG_HASH" ]; then
    info "Installing frontend dependencies..."
    (cd "$FRONTEND_DIR" && npm install --silent)
    echo "$CURRENT_PKG_HASH" > "$PKG_HASH_FILE"
else
    info "Frontend dependencies already up to date"
fi

# ---------------------------------------------------------------------------
# Start services
# ---------------------------------------------------------------------------

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    info "Shutting down..."
    [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null || true
    [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

info "Starting backend on :$BACKEND_PORT..."
(cd "$BACKEND_DIR" && exec uvicorn app.main:app --host 127.0.0.1 --port "$BACKEND_PORT") &
BACKEND_PID=$!

info "Waiting for backend to become healthy..."
for _ in $(seq 1 30); do
    if curl -s -m 1 "http://localhost:$BACKEND_PORT/health" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
if ! curl -s -m 1 "http://localhost:$BACKEND_PORT/health" >/dev/null 2>&1; then
    err "Backend failed to start. Check the output above."
    exit 1
fi
info "Backend is healthy"

info "Starting frontend on :$FRONTEND_PORT..."
(cd "$FRONTEND_DIR" && exec env NEXT_PUBLIC_BACKEND_URL="http://localhost:$BACKEND_PORT" npm run dev -- --port "$FRONTEND_PORT") &
FRONTEND_PID=$!

info "Waiting for frontend to become ready..."
for _ in $(seq 1 60); do
    if curl -s -m 1 "http://localhost:$FRONTEND_PORT" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
if ! curl -s -m 1 "http://localhost:$FRONTEND_PORT" >/dev/null 2>&1; then
    err "Frontend failed to start. Check the output above."
    exit 1
fi

URL="http://localhost:$FRONTEND_PORT"
info "Opening $URL"
if command -v open >/dev/null 2>&1; then
    open "$URL"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL"
else
    warn "Could not detect a way to open a browser automatically. Visit $URL manually."
fi

info "llm-litmus is running. Press Ctrl+C to stop."
wait "$FRONTEND_PID"
