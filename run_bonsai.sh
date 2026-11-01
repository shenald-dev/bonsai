#!/usr/bin/env bash
# Bonsai Micro-AI - Local Web Chat & API Launcher for Linux & macOS

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "===================================================================="
echo "   ____   ____  _   _  _____   ___   ___ "
echo "  (  _ \ / ___)( ) ( )( ____) / _ \ / _ \\"
echo "   ) _ ( \___ \| \_/ | |__   ( (_) | (_) )"
echo "  (____/ (____/ \___/ (____)  \___/ \___/ "
echo ""
echo "  The art of cultivating massive foundation models into pocket-sized"
echo "  intelligences on consumer silicon."
echo "===================================================================="
echo ""

echo "[1/3] Checking Python environment..."
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "[ERROR] Python 3 not found on PATH! Please install Python 3.10+"
    exit 1
fi

PY_CMD="python3"
if ! command -v python3 &>/dev/null; then
    PY_CMD="python"
fi

echo "[2/3] Initializing Bonsai Micro-AI..."
echo "      Output Target: ./bonsai_pruned"
echo ""
echo "[3/3] Launching Web UI & OpenAI API at http://localhost:7860 ..."
echo "      (Press Ctrl+C anytime to stop the server)"
echo ""

$PY_CMD -m bonsai.web_server
