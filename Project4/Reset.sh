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
#   bash Reset.sh 2k 0
#   bash Reset.sh 8k 0
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

MESH_INPUT="${1:-2k}"
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
  MESH_INPUT="2k"
fi
if [[ "$MESH_INPUT" == --no-build ]]; then
  MESH_INPUT="2k"
fi
if [[ "$MESH_INPUT" == --show-adjoint ]]; then
  MESH_INPUT="2k"
fi
if [[ "$MESH_INPUT" == --show-adjoint-wall-only ]]; then
  MESH_INPUT="2k"
fi
if [[ "$MESH_INPUT" == --no-selection-overlay ]]; then
  MESH_INPUT="2k"
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
ADAPT_CFL="${ADAPT_CFL:-1.0}"
ADAPT_ITERS="${ADAPT_ITERS:-2000000}"
ADAPT_TOL="${ADAPT_TOL:-1e-9}"
ADAPT_CYCLES="${ADAPT_CYCLES:-15}"
ADAPT_FRACTION="${ADAPT_FRACTION:-0.025}"
FINAL_AR_CLEANUP="${FINAL_AR_CLEANUP:-8.0}"
SMOOTH_ITERS="${SMOOTH_ITERS:-120}"
WALL_GEOM_TOL="${WALL_GEOM_TOL:-0.15}"
LATEST_CFL="${LATEST_CFL:-0.1}"
LATEST_ITERS="${LATEST_ITERS:-100000}"

# Conservative q1 mesh-quality controls. Override any of these in the shell if
# you want to loosen/tighten a run without rebuilding.
export AMR_Q1_SPLIT_QUALITY_GUARD="${AMR_Q1_SPLIT_QUALITY_GUARD:-1}"
export AMR_Q1_SPLIT_MIN_ANGLE="${AMR_Q1_SPLIT_MIN_ANGLE:-20.0}"
export AMR_Q1_SPLIT_MIN_QUALITY="${AMR_Q1_SPLIT_MIN_QUALITY:-0.10}"
export AMR_Q1_CLOSURE_SPLIT_MIN_ANGLE="${AMR_Q1_CLOSURE_SPLIT_MIN_ANGLE:-4.5}"
export AMR_Q1_CLOSURE_SPLIT_MIN_QUALITY="${AMR_Q1_CLOSURE_SPLIT_MIN_QUALITY:-0.10}"
export AMR_Q1_LOCAL_GRADING_AREA_RATIO="${AMR_Q1_LOCAL_GRADING_AREA_RATIO:-1.5}"
export AMR_Q1_LOCAL_GRADING_MAX_SUPPORTS="${AMR_Q1_LOCAL_GRADING_MAX_SUPPORTS:-6}"
export AMR_Q1_GLOBAL_CLEANUP="${AMR_Q1_GLOBAL_CLEANUP:-1}"
export AMR_Q1_GLOBAL_CLEANUP_MIN_ANGLE="${AMR_Q1_GLOBAL_CLEANUP_MIN_ANGLE:-5.0}"
export AMR_Q1_GLOBAL_CLEANUP_MIN_QUALITY="${AMR_Q1_GLOBAL_CLEANUP_MIN_QUALITY:-0.10}"
export AMR_Q1_GLOBAL_CLEANUP_MAX_SEEDS="${AMR_Q1_GLOBAL_CLEANUP_MAX_SEEDS:-12}"
export AMR_Q1_GLOBAL_CLEANUP_PASSES="${AMR_Q1_GLOBAL_CLEANUP_PASSES:-2}"
export AMR_Q1_PATCH_REPAIR_MIN_ANGLE="${AMR_Q1_PATCH_REPAIR_MIN_ANGLE:-6.0}"
export AMR_Q1_PATCH_REPAIR_MIN_QUALITY="${AMR_Q1_PATCH_REPAIR_MIN_QUALITY:-0.14}"
export AMR_Q1_PATCH_REPAIR_MAX_SWAPS="${AMR_Q1_PATCH_REPAIR_MAX_SWAPS:-24}"
export AMR_Q1_PATCH_REPAIR_RINGS="${AMR_Q1_PATCH_REPAIR_RINGS:-2}"
export AMR_Q1_PROACTIVE_REPAIR="${AMR_Q1_PROACTIVE_REPAIR:-1}"
export AMR_Q1_PROACTIVE_REPAIR_AREA_RATIO="${AMR_Q1_PROACTIVE_REPAIR_AREA_RATIO:-2.0}"
export AMR_Q1_PROACTIVE_REPAIR_MAX_SEEDS="${AMR_Q1_PROACTIVE_REPAIR_MAX_SEEDS:-12}"
export AMR_Q1_TRANSITION_REPAIR_MAX_SEEDS="${AMR_Q1_TRANSITION_REPAIR_MAX_SEEDS:-12}"

echo "=== Cleaning previous steady data ==="
rm -rf "$OUTDIR"

if $NO_BUILD; then
  echo "=== Skipping solver build (--no-build) ==="
else
  echo "=== Building solver ==="
  ./build.sh
fi

echo "=== Running adjoint adaptation on $GRIDFILE (p=$P_ORDER) ==="
echo "    cycles=${ADAPT_CYCLES}, fraction=${ADAPT_FRACTION}, final_ar_cleanup=${FINAL_AR_CLEANUP}"
echo "    q1 split guard: min_angle=${AMR_Q1_SPLIT_MIN_ANGLE}, min_quality=${AMR_Q1_SPLIT_MIN_QUALITY}"
./euler_solver "$GRIDFILE" "$P_ORDER" "$ADAPT_CFL" "$FLUX" "$ADAPT_ITERS" steady \
  --adjoint-adapt "$ADAPT_TOL" "$ADAPT_CYCLES" "$ADAPT_FRACTION" \
  --final-ar-cleanup "$FINAL_AR_CLEANUP" \
  --smooth-iters "$SMOOTH_ITERS" \
  --wall-geom-tol "$WALL_GEOM_TOL"

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

echo "=== Mesh quality summary for exported adapted mesh ==="
python3 - "$LATEST_GRI" <<'PY'
import math
import sys
from pathlib import Path

sys.path.insert(0, "postproc")
from dg_utils import read_gri_mesh

meshfile = Path(sys.argv[1])
mesh = read_gri_mesh(str(meshfile))
nodes = mesh["nodes"]

def metrics(elem):
    corners = elem["corners"]
    pts = [nodes[i] for i in corners]
    lens = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        dx = float(pts[a][0] - pts[b][0])
        dy = float(pts[a][1] - pts[b][1])
        lens.append(math.hypot(dx, dy))
    twice_area = abs(
        (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1])
        - (pts[1][1] - pts[0][1]) * (pts[2][0] - pts[0][0])
    )
    area = 0.5 * twice_area
    angles = []
    for i, j, k in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        ux = float(pts[j][0] - pts[i][0])
        uy = float(pts[j][1] - pts[i][1])
        vx = float(pts[k][0] - pts[i][0])
        vy = float(pts[k][1] - pts[i][1])
        nu = math.hypot(ux, uy)
        nv = math.hypot(vx, vy)
        if nu * nv <= 0.0:
            angles.append(0.0)
        else:
            c = max(-1.0, min(1.0, (ux * vx + uy * vy) / (nu * nv)))
            angles.append(math.degrees(math.acos(c)))
    quality = 2.0 * math.sqrt(3.0) * twice_area / max(sum(l * l for l in lens), 1e-300)
    aspect_ratio = max(lens) / max(twice_area / max(lens), 1e-300)
    return area, min(angles), quality, aspect_ratio

vals = [metrics(elem) for elem in mesh["elements"]]
areas = [v[0] for v in vals]
angles = [v[1] for v in vals]
qualities = [v[2] for v in vals]
aspect_ratios = [v[3] for v in vals]
print(f"  mesh: {meshfile}")
print(f"  elements: {len(vals)}")
print(f"  min angle: {min(angles):.4g} deg")
print(f"  min quality: {min(qualities):.4g}")
print(f"  max aspect ratio: {max(aspect_ratios):.4g}")
print(f"  global area ratio: {max(areas) / max(min(areas), 1e-300):.4g}")
PY

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
