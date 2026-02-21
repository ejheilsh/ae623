#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <grid_name_without_extension> <iters>"
  echo "Example: $0 8k 50000"
  exit 1
fi

GRID="$1"
ITERS="$2"
ORDER="${ORDER:-1}"
CFL="${CFL:-1}"
FLUX="${FLUX:-hlle}"

./euler_solver "grids/${GRID}.gri" "$ORDER" "$CFL" "$FLUX" "$ITERS" steady
python3 postproc/plot_results.py "grids/${GRID}.gri" \
  "data_steady/steady_${GRID}_results.bin" \
  "data_steady/steady_${GRID}_residual.bin" \
  "data_steady/steady_${GRID}_cell_res.bin"
