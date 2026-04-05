#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "Removing steady outputs..."
rm -rf data_steady
rm -f logs/steady_*.log

echo "Removed:"
echo "  data_steady/"
echo "  logs/steady_*.log"
