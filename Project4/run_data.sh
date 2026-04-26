#!/bin/bash
# ============================================================================
# run_data.sh - Generate adapted-grid data for Project 4 postprocessing
# ============================================================================
# Default no-argument run:
#   2k/q1/p0, 2k/q2/p0, 8k/q1/p0, 8k/q2/p0
#
# Main outputs:
#   data_steady/*adjoint_*.bin
#   final_grids/<ncells>/<curvature>/<solver_order>/iter<N>.gri
#   final_grids/<ncells>/<curvature>/<solver_order>/latest_accepted.gri
#
# Examples:
#   bash run_data.sh
#   bash run_data.sh --skip-existing
#   bash run_data.sh 2k q2 p0
#   bash run_data.sh --cycles 5 --fraction 0.025 8k q1 p0
#   bash run_data.sh --export-only 2k q1 p0
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -x "./euler_solver" ]]; then
  SOLVER="./euler_solver"
elif [[ -x "./euler_solver.exe" ]]; then
  SOLVER="./euler_solver.exe"
else
  echo "Error: solver binary not found. Expected ./euler_solver or ./euler_solver.exe"
  echo "Build it first with: bash build.sh"
  exit 1
fi

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: python executable not found: $PYTHON"
  exit 1
fi

OUTDIR="data_steady"
FLUX="roe"
CFL="1.0"
ITERCAP="1000000"
ADAPT_CYCLES=10
ADAPT_FRACTION="0.025"
ADAPT_TOL="0.0"
AR_CLEANUP="${AR_CLEANUP:-10.0}"
SMOOTH_ITERS="${SMOOTH_ITERS:-120}"
WALL_GEOM_TOL="${WALL_GEOM_TOL:-0.15}"
SKIP_EXISTING=false
EXPORT_ONLY=false
POSITIONAL=()

usage() {
  sed -n '1,17p' "$0"
  cat <<'EOF'

Options:
  --skip-existing       Skip solver runs whose final iter<N>.gri already exists
  --export-only         Only export existing data_steady mesh snapshots to final_grids
  --cycles N            Adaptation cycle count (default: 10)
  --fraction X          Marking fraction (default: 0.025)
  --tol X               Adjoint tolerance; 0.0 forces all cycles (default: 0.0)
  --itercap N           Primal iteration cap during adaptation (default: 1000000)
  --cfl X               Solver CFL during adaptation (default: 1.0)
  --flux roe|hlle       Numerical flux (default: roe)
  --final-ar-cleanup X  Also mark cells above aspect-ratio threshold (default: 10.0)
  --smooth-iters N      Post-refinement smoothing iterations (default: 120)
  --wall-geom-tol X     Wall geometry tolerance passed to solver (default: 0.15)
  -h, --help            Show this help

Positional forms:
  none                  Run all four report cases
  all                   Run all four report cases
  <ncells> [q] [p]      Run one case, e.g. 2k q2 p0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-existing)
      SKIP_EXISTING=true
      shift
      ;;
    --export-only)
      EXPORT_ONLY=true
      shift
      ;;
    --cycles)
      ADAPT_CYCLES="$2"
      shift 2
      ;;
    --fraction)
      ADAPT_FRACTION="$2"
      shift 2
      ;;
    --tol)
      ADAPT_TOL="$2"
      shift 2
      ;;
    --itercap)
      ITERCAP="$2"
      shift 2
      ;;
    --cfl)
      CFL="$2"
      shift 2
      ;;
    --flux)
      FLUX="$2"
      shift 2
      ;;
    --final-ar-cleanup)
      AR_CLEANUP="$2"
      shift 2
      ;;
    --smooth-iters)
      SMOOTH_ITERS="$2"
      shift 2
      ;;
    --wall-geom-tol)
      WALL_GEOM_TOL="$2"
      shift 2
      ;;
    --resume)
      echo "Warning: --resume is accepted for compatibility but adjoint adaptation does not resume mid-cycle."
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "Error: unknown option: $1"
      usage
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -gt 3 ]]; then
  echo "Error: expected at most 3 positional arguments: [ncells] [curvature] [solver_order]"
  exit 1
fi

normalize_curvature() {
  local value="$1"
  if [[ "$value" =~ ^[0-9]+$ ]]; then
    value="q${value}"
  fi
  if [[ ! "$value" =~ ^q[0-9]+$ ]]; then
    echo "Error: curvature must look like q1, q2, q3, or a number such as 1" >&2
    exit 1
  fi
  echo "$value"
}

normalize_solver_order() {
  local value="$1"
  if [[ "$value" =~ ^[0-9]+$ ]]; then
    value="p${value}"
  fi
  if [[ ! "$value" =~ ^p[0-9]+$ ]]; then
    echo "Error: solver_order must look like p0, p1, p2, or a number such as 0" >&2
    exit 1
  fi
  echo "$value"
}

grid_tag_for_case() {
  local ncells="$1"
  local curvature="$2"
  if [[ "$curvature" == "q1" ]]; then
    echo "$ncells"
  else
    echo "${ncells}_${curvature}"
  fi
}

prefix_for_case() {
  local ncells="$1"
  local curvature="$2"
  local order_tag="$3"
  local grid_tag
  grid_tag="$(grid_tag_for_case "$ncells" "$curvature")"
  echo "steady_${grid_tag}_${order_tag}_"
}

export_final_grids() {
  local ncells="$1"
  local curvature="$2"
  local order_tag="$3"
  local prefix="$4"
  local final_dir="final_grids/${ncells}/${curvature}/${order_tag}"
  local count=0

  mkdir -p "$final_dir"
  if [[ "${PRESERVE_FINAL_GRIDS:-0}" != "1" ]]; then
    rm -f "${final_dir}"/iter*.gri "${final_dir}/latest_accepted.gri"
  fi
  shopt -s nullglob
  local mesh_files=("${OUTDIR}/${prefix}"adjoint_mesh_cycle*.bin)
  shopt -u nullglob

  if [[ ${#mesh_files[@]} -eq 0 ]]; then
    echo "  [WARN] no mesh snapshots found for prefix ${prefix}"
    return 0
  fi

  while IFS= read -r mesh_file; do
    local base cycle out_file
    base="$(basename "$mesh_file")"
    if [[ ! "$base" =~ cycle([0-9]+)\.bin$ ]]; then
      continue
    fi
    cycle="${BASH_REMATCH[1]}"
    out_file="${final_dir}/iter${cycle}.gri"
    echo "  exporting ${mesh_file} -> ${out_file}"
    "$PYTHON" postproc/export_adapted_mesh_gri.py "$mesh_file" "$out_file"
    count=$((count + 1))
  done < <("$PYTHON" - "${mesh_files[@]}" <<'PY'
import re
import sys

def key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)]

for item in sorted(sys.argv[1:], key=key):
    print(item)
PY
)

  local latest_bin="${OUTDIR}/${prefix}adjoint_mesh_latest_accepted.bin"
  if [[ -f "$latest_bin" ]]; then
    echo "  exporting ${latest_bin} -> ${final_dir}/latest_accepted.gri"
    "$PYTHON" postproc/export_adapted_mesh_gri.py "$latest_bin" "${final_dir}/latest_accepted.gri"
  fi

  echo "  exported ${count} iter mesh(es) for ${ncells}/${curvature}/${order_tag}"
}

run_case() {
  local ncells="$1"
  local curvature="$2"
  local order_tag="$3"
  curvature="$(normalize_curvature "$curvature")"
  order_tag="$(normalize_solver_order "$order_tag")"
  local order="${order_tag#p}"
  local grid_tag grid_file prefix final_dir marker log_file

  if [[ "$order_tag" != "p0" ]]; then
    echo "Warning: this pipeline is intended for p0 adaptation; running ${order_tag} because it was requested."
  fi

  grid_tag="$(grid_tag_for_case "$ncells" "$curvature")"
  grid_file="grids/${grid_tag}.gri"
  prefix="$(prefix_for_case "$ncells" "$curvature" "$order_tag")"
  final_dir="final_grids/${ncells}/${curvature}/${order_tag}"
  marker="${final_dir}/iter${ADAPT_CYCLES}.gri"
  log_file="${OUTDIR}/adapt_${grid_tag}_${order_tag}_run.log"

  if [[ ! -f "$grid_file" ]]; then
    echo "Error: grid file not found: $grid_file"
    return 1
  fi

  echo ""
  echo "================================================================"
  echo "  Adjoint adaptation: ${ncells}/${curvature}/${order_tag}"
  echo "  grid=${grid_file}, cycles=${ADAPT_CYCLES}, fraction=${ADAPT_FRACTION}"
  echo "================================================================"

  mkdir -p "$OUTDIR"

  if $EXPORT_ONLY; then
    echo "  export-only mode: skipping solver"
  elif $SKIP_EXISTING && [[ -f "$marker" ]]; then
    echo "  [SKIP] ${marker} already exists"
  else
    "$SOLVER" "$grid_file" "$order" "$CFL" "$FLUX" "$ITERCAP" steady \
      --adjoint-adapt "$ADAPT_TOL" "$ADAPT_CYCLES" "$ADAPT_FRACTION" \
      --final-ar-cleanup "$AR_CLEANUP" \
      --smooth-iters "$SMOOTH_ITERS" \
      --wall-geom-tol "$WALL_GEOM_TOL" \
      2>&1 | tee "$log_file"
  fi

  export_final_grids "$ncells" "$curvature" "$order_tag" "$prefix"
}

CASES=()
if [[ ${#POSITIONAL[@]} -eq 0 || "${POSITIONAL[0]:-}" == "all" ]]; then
  CASES=("2k q1 p0" "8k q1 p0")
else
  NCELLS="${POSITIONAL[0]}"
  CURVATURE="$(normalize_curvature "${POSITIONAL[1]:-q1}")"
  ORDER_TAG="$(normalize_solver_order "${POSITIONAL[2]:-p0}")"
  CASES=("${NCELLS} ${CURVATURE} ${ORDER_TAG}")
fi

for case_spec in "${CASES[@]}"; do
  read -r ncells curvature order_tag <<< "$case_spec"
  run_case "$ncells" "$curvature" "$order_tag"
done

echo ""
echo "================================================================"
echo "  Data generation complete."
echo "  Adaptation data: ${OUTDIR}/"
echo "  Final grids    : final_grids/"
echo "================================================================"
