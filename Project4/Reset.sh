#!/bin/bash
# ============================================================================
# Reset.sh — rebuild, run adjoint adaptation, animate, export latest mesh,
#            rerun the exported mesh, and plot the steady solution summary.
# ============================================================================
# Usage:
#   bash Reset.sh [mesh] [p_order] [--preview] [--no-build] [--show-adjoint] [--show-adjoint-wall-only] [--no-selection-overlay]
#
# Examples:
#   bash Reset.sh
#   bash Reset.sh test_q2
#   bash Reset.sh grids/test_q2.gri 0 --preview
#   bash Reset.sh test 0 --no-build
#   bash Reset.sh test 0 --show-adjoint --preview
#   bash Reset.sh test 0 --show-adjoint-wall-only --preview
#   bash Reset.sh test 0 --show-adjoint --no-selection-overlay
# ============================================================================
set -euo pipefail

# Keep plotting/font caches inside writable locations when running under the
# desktop sandbox or other restricted environments.
CACHE_ROOT="${TMPDIR:-/tmp}/ae623_reset_cache"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg-cache"
export MPLBACKEND="Agg"
mkdir -p "$MPLCONFIGDIR" "${XDG_CACHE_HOME}/fontconfig"

MESH_INPUT="${1:-test}"
P_ORDER="${2:-0}"
PREVIEW=false
NO_BUILD=false
SHOW_ADJOINT=false
SHOW_ADJOINT_WALL_ONLY=false
SHOW_SELECTION_OVERLAY=true

for arg in "$@"; do
  case "$arg" in
    --preview) PREVIEW=true ;;
    --no-build) NO_BUILD=true ;;
    --show-adjoint) SHOW_ADJOINT=true ;;
    --show-adjoint-wall-only)
      SHOW_ADJOINT=true
      SHOW_ADJOINT_WALL_ONLY=true
      ;;
    --no-selection-overlay) SHOW_SELECTION_OVERLAY=false ;;
  esac
done

if [[ "$MESH_INPUT" == --preview ]]; then
  MESH_INPUT="test_q2"
fi
if [[ "$MESH_INPUT" == --no-build ]]; then
  MESH_INPUT="test_q2"
fi
if [[ "$MESH_INPUT" == --show-adjoint ]]; then
  MESH_INPUT="test_q2"
fi
if [[ "$MESH_INPUT" == --show-adjoint-wall-only ]]; then
  MESH_INPUT="test_q2"
fi
if [[ "$MESH_INPUT" == --no-selection-overlay ]]; then
  MESH_INPUT="test_q2"
fi
if [[ "$P_ORDER" == --preview ]]; then
  P_ORDER=0
fi
if [[ "$P_ORDER" == --no-build ]]; then
  P_ORDER=0
fi
if [[ "$P_ORDER" == --show-adjoint ]]; then
  P_ORDER=0
fi
if [[ "$P_ORDER" == --show-adjoint-wall-only ]]; then
  P_ORDER=0
fi
if [[ "$P_ORDER" == --no-selection-overlay ]]; then
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
FAILED_GRI="grids/latest_failed_adapted.gri"
LATEST_NAME="latest_adapted"
LATEST_MESH_PNG_DIR="postproc_out/reset_mesh"
LATEST_MESH_PNG="${LATEST_MESH_PNG_DIR}/mesh_latest_adapted_full.png"
FLUX="roe"
ADAPT_CFL="1.0"
ADAPT_ITERS="2000000"
ADAPT_TOL="1e-9"
ADAPT_CYCLES="50"
ADAPT_FRACTION="0.05"
LATEST_CFL="0.1"
LATEST_ITERS="100000"

echo "=== Cleaning previous steady data ==="
rm -rf "$OUTDIR"

if $NO_BUILD; then
  echo "=== Skipping solver build (--no-build) ==="
else
  echo "=== Building solver ==="
  ./build.sh
fi

echo "=== Running adjoint adaptation on $GRIDFILE (p=$P_ORDER) ==="
./euler_solver "$GRIDFILE" "$P_ORDER" "$ADAPT_CFL" "$FLUX" "$ADAPT_ITERS" steady \
  --adjoint-adapt "$ADAPT_TOL" "$ADAPT_CYCLES" "$ADAPT_FRACTION"

echo "=== Animating adaptation history ==="
ANIM_ARGS=(
  "$OUTDIR"
  "$ADAPT_GIF"
  --show-blade
)
if $SHOW_SELECTION_OVERLAY; then
  ANIM_ARGS+=(--show-next-refine)
fi
if $SHOW_ADJOINT; then
  ANIM_ARGS+=(--show-indicators)
fi
if $SHOW_ADJOINT_WALL_ONLY; then
  ANIM_ARGS+=(--show-indicators-wall-only)
fi
uv run python postproc/animate_mesh_adaptation.py "${ANIM_ARGS[@]}"

if $PREVIEW; then
  if [[ -f "$ADAPT_GIF" ]]; then
    echo "=== Previewing animation ==="
    qlmanage -p "$ADAPT_GIF"
  else
    echo "Warning: animation preview requested, but ${ADAPT_GIF} was not created."
  fi
fi

echo "=== Exporting newest adapted mesh snapshot for ${MESH_NAME} ==="
LATEST_ACCEPTED="${OUTDIR}/steady_${MESH_NAME}_p${P_ORDER}_adjoint_mesh_latest_accepted.bin"
if [[ -f "$LATEST_ACCEPTED" ]]; then
  LATEST_MESH="$LATEST_ACCEPTED"
else
  LATEST_MESH="$(ls -t "${OUTDIR}/steady_${MESH_NAME}_p${P_ORDER}_adjoint_mesh_cycle"[0-9]*.bin | head -n 1)"
fi
python3 postproc/export_adapted_mesh_gri.py "$LATEST_MESH" "$LATEST_GRI"

echo "=== Exporting newest failed adapted mesh snapshot for ${MESH_NAME} (if any) ==="
shopt -s nullglob
FAILED_MESH_CANDIDATES=("${OUTDIR}/steady_${MESH_NAME}_p${P_ORDER}_adjoint_mesh_cycle"*_failed.bin)
if (( ${#FAILED_MESH_CANDIDATES[@]} > 0 )); then
  FAILED_MESH="$(ls -t "${FAILED_MESH_CANDIDATES[@]}" | head -n 1)"
  python3 postproc/export_adapted_mesh_gri.py "$FAILED_MESH" "$FAILED_GRI"
  echo "Wrote ${FAILED_GRI}"
else
  echo "No failed refined mesh snapshot found."
fi
shopt -u nullglob

echo "=== Plotting exported adapted mesh with blade overlay ==="
mkdir -p "$LATEST_MESH_PNG_DIR"
python3 postproc/plot_mesh_zoom_compare.py \
  --meshes "$LATEST_GRI" \
  --labels adapted \
  --views full \
  --show-blade \
  --outdir "$LATEST_MESH_PNG_DIR" \
  --no-show
if [[ -f "${LATEST_MESH_PNG_DIR}/mesh_latest_adapted_full.png" ]]; then
  echo "Saved ${LATEST_MESH_PNG_DIR}/mesh_latest_adapted_full.png"
fi

echo "=== Running exported adapted mesh ==="
./euler_solver "$LATEST_GRI" "$P_ORDER" "$LATEST_CFL" "$FLUX" "$LATEST_ITERS" steady

echo "=== Plotting latest adapted steady results ==="
python3 postproc/plot_results.py \
  "grids/${LATEST_NAME}.gri" \
  "${OUTDIR}/steady_${LATEST_NAME}_p${P_ORDER}_results.bin" \
  "${OUTDIR}/steady_${LATEST_NAME}_p${P_ORDER}_residual.bin" \
  "${OUTDIR}/steady_${LATEST_NAME}_p${P_ORDER}_cell_res.bin" \
  --no-show
