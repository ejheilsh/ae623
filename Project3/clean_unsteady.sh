#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "Removing unsteady outputs..."
rm -rf unsteady_data
rm -f data/results_*.bin
rm -f data/results_*_dg.bin
rm -f results.bin results_dg.bin residual.bin cell_res.bin
rm -f logs/unsteady_*.log

echo "Removed:"
echo "  unsteady_data/"
echo "  data/results_*.bin"
echo "  data/results_*_dg.bin"
echo "  results.bin results_dg.bin residual.bin cell_res.bin"
echo "  logs/unsteady_*.log"
