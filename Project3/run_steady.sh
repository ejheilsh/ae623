#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <grid_name_without_extension> <iters>"
  echo "Example: $0 8k 50000"
  exit 1
fi

GRID="$1"
ITERS="$2"
shift 2
declare -a EXTRA_ARGS=("$@")
ORDER="${ORDER:-1}"
CFL="${CFL:-1}"
FLUX="${FLUX:-hlle}"

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  ./euler_solver "grids/${GRID}.gri" "$ORDER" "$CFL" "$FLUX" "$ITERS" steady "${EXTRA_ARGS[@]}"
else
  ./euler_solver "grids/${GRID}.gri" "$ORDER" "$CFL" "$FLUX" "$ITERS" steady
fi
python3 postproc/plot_results.py "grids/${GRID}.gri" \
  "data_steady/steady_${GRID}_results.bin" \
  "data_steady/steady_${GRID}_residual.bin" \
  "data_steady/steady_${GRID}_cell_res.bin"
