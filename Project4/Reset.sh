#!/bin/bash
# ============================================================================
# Reset.sh — rebuild, run adjoint adaptation, animate, export latest mesh,
#            rerun the exported mesh, and plot the steady solution summary.
# ============================================================================
# Usage:
#   bash Reset.sh [mesh] [p_order] [--preview]
#
# Examples:
#   bash Reset.sh
#   bash Reset.sh test_q2
#   bash Reset.sh grids/test_q2.gri 0 --preview
# ============================================================================
set -euo pipefail

# Keep plotting/font caches inside writable locations when running under the
# desktop sandbox or other restricted environments.
CACHE_ROOT="${TMPDIR:-/tmp}/ae623_reset_cache"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg-cache"
export MPLBACKEND="Agg"
mkdir -p "$MPLCONFIGDIR" "${XDG_CACHE_HOME}/fontconfig"

MESH_INPUT="${1:-test_q2}"
P_ORDER="${2:-0}"
PREVIEW=false

for arg in "$@"; do
  case "$arg" in
    --preview) PREVIEW=true ;;
  esac
done

if [[ "$MESH_INPUT" == --preview ]]; then
  MESH_INPUT="test_q2"
fi
if [[ "$P_ORDER" == --preview ]]; then
  P_ORDER=0
fi

if [[ "$MESH_INPUT" == *.gri ]]; then
  GRIDFILE="$MESH_INPUT"
  MESH_NAME="$(basename "${MESH_INPUT%.gri}")"
else
  GRIDFILE="grids/${MESH_INPUT}.gri"
  MESH_NAME="$MESH_INPUT"
fi

if [[ ! -f "$GRIDFILE" ]]; then
  echo "Error: mesh file not found: $GRIDFILE"
  exit 1
fi

OUTDIR="data_steady"
ADAPT_GIF="mesh_adaptation.gif"
LATEST_GRI="grids/latest_adapted.gri"
LATEST_NAME="latest_adapted"
FLUX="roe"
ADAPT_CFL="1.0"
ADAPT_ITERS="2000000"
ADAPT_TOL="1e-9"
ADAPT_CYCLES="200"
ADAPT_FRACTION="0.01"
LATEST_CFL="0.1"
LATEST_ITERS="100000"

echo "=== Cleaning previous steady data ==="
rm -rf "$OUTDIR"

echo "=== Building solver ==="
./build.sh

echo "=== Running adjoint adaptation on $GRIDFILE (p=$P_ORDER) ==="
./euler_solver "$GRIDFILE" "$P_ORDER" "$ADAPT_CFL" "$FLUX" "$ADAPT_ITERS" steady \
  --adjoint-adapt "$ADAPT_TOL" "$ADAPT_CYCLES" "$ADAPT_FRACTION"

echo "=== Animating adaptation history ==="
# uv run postproc/animate_mesh_adaptation.py "$OUTDIR" "$ADAPT_GIF" --show-blade --show-next-refine

if $PREVIEW; then
  echo "=== Previewing animation ==="
  qlmanage -p "$ADAPT_GIF"
fi

echo "=== Exporting newest adapted mesh snapshot for ${MESH_NAME} ==="
LATEST_ACCEPTED="${OUTDIR}/steady_${MESH_NAME}_p${P_ORDER}_adjoint_mesh_latest_accepted.bin"
if [[ -f "$LATEST_ACCEPTED" ]]; then
  LATEST_MESH="$LATEST_ACCEPTED"
else
  LATEST_MESH="$(ls -t "${OUTDIR}/steady_${MESH_NAME}_p${P_ORDER}_adjoint_mesh_cycle"[0-9]*.bin | head -n 1)"
fi
python3 postproc/export_adapted_mesh_gri.py "$LATEST_MESH" "$LATEST_GRI"

echo "=== Running exported adapted mesh ==="
./euler_solver "$LATEST_GRI" "$P_ORDER" "$LATEST_CFL" "$FLUX" "$LATEST_ITERS" steady

echo "=== Plotting latest adapted steady results ==="
python3 postproc/plot_results.py \
  "grids/${LATEST_NAME}.gri" \
  "${OUTDIR}/steady_${LATEST_NAME}_p${P_ORDER}_results.bin" \
  "${OUTDIR}/steady_${LATEST_NAME}_p${P_ORDER}_residual.bin" \
  "${OUTDIR}/steady_${LATEST_NAME}_p${P_ORDER}_cell_res.bin" \
  --no-show
