#!/usr/bin/env bash
set -e

# ── Run the Event Extraction System ──────────────────────────────────────────
#
# Prerequisites:
#   - Python 3.12
#   - uv  (install: pip install uv  or  curl -Lsf https://astral.sh/uv/install.sh | sh)
#   - A .env file in this directory containing: GOOGLE_API_KEY=your_key
#     (copy env.example → .env and fill in your key)
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for .env
if [ ! -f "$ROOT_DIR/.env" ]; then
  echo "ERROR: .env file not found."
  echo "  Copy env.example to .env and add your GOOGLE_API_KEY."
  exit 1
fi

# Check for uv
if ! command -v uv &>/dev/null; then
  echo "ERROR: 'uv' is not installed."
  echo "  Install it with:  pip install uv"
  exit 1
fi

echo "Starting backend + UI on http://localhost:8000 ..."
echo "Press Ctrl+C to stop."
echo ""

uv run --with-requirements "$ROOT_DIR/backend_service/requirements.txt" \
  uvicorn backend_service.app.main:app --reload --port 8000
