#!/usr/bin/env bash
set -euo pipefail

ORDER="${ORDER:-2}"
CFL="${CFL:-1}"
FLUX="${FLUX:-roe}"
OUTPUT_DIR="${OUTPUT_DIR:-output_final}"

mkdir -p "$OUTPUT_DIR"

if [ ! -x "./euler_solver" ]; then
  echo "euler_solver not found; building first..."
  ./build.sh
fi

# Define the sequence of meshes and iteration counts
declare -a GRIDS=("2k" "8k" "32k" "128k")
declare -a ITERS=("1000000" "1000000" "1000000" "1000000")

PREV_GRID=""

for i in "${!GRIDS[@]}"; do
  GRID="${GRIDS[$i]}"
  ITER="${ITERS[$i]}"

  echo "======================================================================"
  if [ -z "$PREV_GRID" ]; then
    echo "Running steady solve: grid=${GRID}, order=${ORDER}, cfl=${CFL}, flux=${FLUX}, iters=${ITER}"
    echo "======================================================================"
    ORDER="$ORDER" CFL="$CFL" FLUX="$FLUX" MPLBACKEND=Agg ./run_steady.sh "$GRID" "$ITER"
  else
    echo "Running steady solve: grid=${GRID}, order=${ORDER}, cfl=${CFL}, flux=${FLUX}, iters=${ITER} (Mapped from ${PREV_GRID})"
    echo "======================================================================"
    ORDER="$ORDER" CFL="$CFL" FLUX="$FLUX" MPLBACKEND=Agg ./run_steady.sh "$GRID" "$ITER" --map-ic "grids/${PREV_GRID}.gri" "data_steady/steady_${PREV_GRID}_results.bin"
  fi

  RESIDUAL_SRC="data_steady/steady_${GRID}_residual.bin"
  STATE_SRC="data_steady/steady_${GRID}_results.bin"
  CELLRES_SRC="data_steady/steady_${GRID}_cell_res.bin"
  
  RESIDUAL_DST="${OUTPUT_DIR}/conv_gri${GRID}_ord${ORDER}_${FLUX}_cfl${CFL}.bin"
  STATE_DST="${OUTPUT_DIR}/state_gri${GRID}_ord${ORDER}_${FLUX}_cfl${CFL}.bin"
  CELLRES_DST="${OUTPUT_DIR}/cellres_gri${GRID}_ord${ORDER}_${FLUX}_cfl${CFL}.bin"

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

  if [ -f "$CELLRES_SRC" ]; then
    cp "$CELLRES_SRC" "$CELLRES_DST"
    echo "Saved $CELLRES_DST"
  fi
  
  # Set up for the next loop iteration mapping
  PREV_GRID="$GRID"
done

echo "All meshes ran and saved successfully to $OUTPUT_DIR/"
