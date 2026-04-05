#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <grid_name_without_extension> <iters> [ic_file]"
  echo "Example: $0 8k 50000"
  echo "Example: $0 8k 50000 data_steady/steady_8k_p1_results.bin"
  exit 1
fi

GRID="$1"
ITERS="$2"
MESH="grids/${GRID}.gri"

ORDER="${ORDER:-1}"
CFL="${CFL:-0.1}"
FLUX="${FLUX:-hlle}"

STEADY_ITERS="${STEADY_ITERS:-10000}"
STEADY_CFL="${STEADY_CFL:-0.5}"

if [ "$#" -ge 3 ]; then
  IC_FILE="$3"
else
  IC_FILE="data_steady/steady_${GRID}_p${ORDER}_results.bin"
fi

if [ ! -f "$MESH" ]; then
  echo "Error: mesh not found: $MESH"
  exit 1
fi

if [ ! -f "$IC_FILE" ]; then
  echo "Initial condition not found: $IC_FILE"
  echo "Running steady solve first to create it..."
  ./euler_solver "$MESH" "$ORDER" "$STEADY_CFL" "$FLUX" "$STEADY_ITERS" steady
fi

./euler_solver "$MESH" "$ORDER" "$CFL" "$FLUX" "$ITERS" unsteady "$IC_FILE"
RESULTS_DIR="unsteady_data/${GRID}_p${ORDER}"
python3 postproc/plot_unsteady.py "$MESH" "$RESULTS_DIR" --latest 3
