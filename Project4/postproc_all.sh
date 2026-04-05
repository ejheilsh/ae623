#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.mplconfig}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
mkdir -p "$MPLCONFIGDIR"

MESH="${MESH:-grids/2k_q3.gri}"
GRID_TAG="${GRID_TAG:-2k_q3}"
P_ORDERS_STR="${P_ORDERS:-0 1 2 3}"
read -r -a P_ORDER_LIST <<< "$P_ORDERS_STR"
if [[ ${#P_ORDER_LIST[@]} -eq 0 ]]; then
  P_ORDER_LIST=(0 1 2 3)
fi

echo "Task 3 Mach contour postprocessing"
echo "  mesh: $MESH"
echo "  grid tag: $GRID_TAG"
echo "  orders: ${P_ORDER_LIST[*]}"

mkdir -p postproc_out/task3

RESULT_FILES=()
LABELS=()

for p in "${P_ORDER_LIST[@]}"; do
  results_file="data_steady/steady_${GRID_TAG}_p${p}_results.bin"

  if [[ ! -f "$results_file" ]]; then
    echo "Skipping p=${p}: missing ${results_file}"
    continue
  fi

  echo "Plotting Mach contour for p=${p}"
  python3 postproc/plot_mach_contour.py \
    "$MESH" \
    "$results_file" \
    --cmap turbo \
    --vmin 0.0 \
    --vmax 0.85 \
    --no-show

  RESULT_FILES+=("$results_file")
  LABELS+=("p=${p}")
done

if [[ ${#RESULT_FILES[@]} -gt 0 ]]; then
  echo "Plotting combined Cp distribution"
  python3 postproc/plot_cp_series.py \
    "$MESH" \
    "${RESULT_FILES[@]}" \
    --labels "${LABELS[@]}" \
    --output "postproc_out/task3/cp_series_${GRID_TAG}.png" \
    --no-show
fi

echo "Finished. Outputs are in postproc_out/task3/"
