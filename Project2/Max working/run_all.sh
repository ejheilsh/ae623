#!/usr/bin/env bash
set -euo pipefail

ITERS_MAX="${ITERS_MAX:-1000000}"
ORDER="${ORDER:-2}"
CFL="${CFL:-0.5}"
FLUX="${FLUX:-hlle}"
GRIDS_STR="${GRIDS:-8k}"
read -r -a GRIDS_ARR <<< "$GRIDS_STR"
OUTPUT_DIR="${OUTPUT_DIR:-output_final}"

mkdir -p "$OUTPUT_DIR"

if [ ! -x "./euler_solver" ]; then
  echo "euler_solver not found; building first..."
  ./build.sh
fi

for GRID in "${GRIDS_ARR[@]}"; do
  echo "======================================================================"
  echo "Running steady solve: grid=${GRID}, order=${ORDER}, cfl=${CFL}, flux=${FLUX}, iters=${ITERS_MAX}"
  echo "======================================================================"

  ORDER="$ORDER" CFL="$CFL" FLUX="$FLUX" ./run_steady.sh "$GRID" "$ITERS_MAX"

  RESIDUAL_SRC="data_steady/steady_${GRID}_residual.bin"
  STATE_SRC="data_steady/steady_${GRID}_results.bin"
  RESIDUAL_DST="${OUTPUT_DIR}/conv_gri${GRID}_ord${ORDER}_${FLUX}_cfl${CFL}.bin"
  STATE_DST="${OUTPUT_DIR}/state_gri${GRID}_ord${ORDER}_${FLUX}_cfl${CFL}.bin"

  if [ -f "$RESIDUAL_SRC" ]; then
    cp "$RESIDUAL_SRC" "$RESIDUAL_DST"
    echo "Saved $RESIDUAL_DST"
  else
    echo "Warning: missing $RESIDUAL_SRC"
  fi

  if [ -f "$STATE_SRC" ]; then
    cp "$STATE_SRC" "$STATE_DST"
    echo "Saved $STATE_DST"
  else
    echo "Warning: missing $STATE_SRC"
  fi
done
