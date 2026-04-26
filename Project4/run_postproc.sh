#!/bin/bash
# ============================================================================
# run_postproc.sh - Generate P4final-style final postprocessing outputs
# ============================================================================
# Inputs:
#   final_grids/<ncells>/<curvature>/<solver_order>/iter<N>.gri
#   final_solutions/<ncells>/<curvature>/<solver_order>/iter<N>_*.bin
#   data_steady/*adjoint_indicators_cycle<N>.bin, when available
#
# Outputs:
#   final_cases/cl_values.csv
#   final_cases/cl_convergence.png
#   final_cases/cl_vs_cells.png
#   final_cases/cl_error.png
#   final_cases/indicators/*adjoint_indicators_cycle<N>.bin
#   final_figures/cp_*.png
#   final_figures/mach_*.png
#   final_figures/aspect_ratio_*.png
#   final_figures/adjoint_indicator_*.png
#   final_figures/cl_convergence.png              copied from final_cases/cl_vs_cells.png
#   final_figures/cl_convergence_by_iteration.png copied from final_cases/cl_convergence.png
#   final_figures/cl_error.png                    copied from final_cases/cl_error.png
#   final_figures/cl_error_by_iteration.png       copied from final_cases/cl_error_by_iteration.png
#
# Examples:
#   bash run_postproc.sh
#   bash run_postproc.sh --case 8k/q2/p0
#   bash run_postproc.sh 2k q1 p0
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: python executable not found: $PYTHON"
  exit 1
fi

export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPATH="${ROOT_DIR}/postproc${PYTHONPATH:+:${PYTHONPATH}}"

FINAL_GRIDS="${FINAL_GRIDS:-final_grids}"
FINAL_SOLUTIONS="${FINAL_SOLUTIONS:-final_solutions}"
FINAL_CASES="${FINAL_CASES:-final_cases}"
FINAL_FIGURES="${FINAL_FIGURES:-final_figures}"
REFERENCE_MESH="${REFERENCE_MESH:-/Users/eheilshorn/Courses/AE623/Code/ae623/Project3/grids/128k_q3.gri}"
REFERENCE_RESULTS="${REFERENCE_RESULTS:-/Users/eheilshorn/Courses/AE623/Code/ae623/Project3/data_steady/steady_128k_q3_p0_results.bin}"
REFERENCE_LABEL="${REFERENCE_LABEL:-Project3/128k/q3/p0}"
CP_REFERENCE_LABEL="${CP_REFERENCE_LABEL:-reference}"
UNIFORM_CURVATURE="${UNIFORM_CURVATURE:-q3}"
UNIFORM_ORDER="${UNIFORM_ORDER:-p0}"
UNIFORM_GRIDS="${UNIFORM_GRIDS:-2k,8k,32k,128k}"

CASE_FILTER=""
CURVATURE_FILTER=""
SKIP_EXISTING=false
NO_UNIFORM=false
DO_CL=true
DO_CP=true
DO_CONTOURS=true
DO_INDICATORS=true
INDICATOR_ALL_ITERATIONS=false
RECOMPUTE_INDICATORS=false
POSITIONAL=()

usage() {
  sed -n '1,23p' "$0"
  cat <<'EOF'

Options:
  --case A/B/C              Process one case, e.g. --case 8k/q2/p0
  --curvature qN            Process only one geometry curvature, e.g. q1
  --skip-existing           Skip existing fixed-grid indicator files
  --no-uniform              Do not add uniform-grid comparison rows to CL plots
  --no-cl                   Skip CL tables/convergence plots
  --no-cp                   Skip Cp distribution plots
  --no-contours             Skip Mach/aspect-ratio/indicator contour plots
  --no-indicators           Skip fixed-grid indicator computation and indicator contours
  --indicator-all-iterations Compute fixed-grid indicators for all solved iterations
  --recompute-indicators    Recompute fixed-grid indicators even if reusable ones exist
  --reference-mesh PATH     Reference mesh for CL/Cp overlays
  --reference-results PATH  Reference solution for CL/Cp overlays
  --reference-label TEXT    Reference label for plot legends
  -h, --help                Show this help

Positional form:
  <ncells> [q] [p]          Same as --case <ncells>/<q>/<p>
EOF
}

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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case)
      CASE_FILTER="${2#/}"
      CASE_FILTER="${CASE_FILTER%/}"
      shift 2
      ;;
    --curvature)
      CURVATURE_FILTER="$(normalize_curvature "$2")"
      shift 2
      ;;
    --skip-existing)
      SKIP_EXISTING=true
      shift
      ;;
    --no-uniform)
      NO_UNIFORM=true
      shift
      ;;
    --no-cl)
      DO_CL=false
      shift
      ;;
    --no-cp)
      DO_CP=false
      shift
      ;;
    --no-contours)
      DO_CONTOURS=false
      shift
      ;;
    --no-indicators)
      DO_INDICATORS=false
      shift
      ;;
    --indicator-all-iterations)
      INDICATOR_ALL_ITERATIONS=true
      shift
      ;;
    --recompute-indicators)
      RECOMPUTE_INDICATORS=true
      shift
      ;;
    --reference-mesh)
      REFERENCE_MESH="$2"
      shift 2
      ;;
    --reference-results)
      REFERENCE_RESULTS="$2"
      shift 2
      ;;
    --reference-label)
      REFERENCE_LABEL="$2"
      shift 2
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

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  NCELLS="${POSITIONAL[0]}"
  CURVATURE="$(normalize_curvature "${POSITIONAL[1]:-q1}")"
  ORDER_TAG="$(normalize_solver_order "${POSITIONAL[2]:-p0}")"
  CASE_FILTER="${NCELLS}/${CURVATURE}/${ORDER_TAG}"
fi

if [[ -n "$CASE_FILTER" && -n "$CURVATURE_FILTER" ]]; then
  IFS=/ read -r _case_ncells case_curvature _case_order <<< "$CASE_FILTER"
  if [[ "$case_curvature" != "$CURVATURE_FILTER" ]]; then
    echo "Error: --case ${CASE_FILTER} conflicts with --curvature ${CURVATURE_FILTER}" >&2
    exit 1
  fi
fi

mkdir -p "$FINAL_CASES" "$FINAL_FIGURES"

if [[ ! -d "$FINAL_GRIDS" ]]; then
  echo "Error: missing ${FINAL_GRIDS}/. Run run_data.sh first."
  exit 1
fi

if [[ ! -d "$FINAL_SOLUTIONS" ]]; then
  echo "Error: missing ${FINAL_SOLUTIONS}/. Run solve_final_grids.sh first."
  exit 1
fi

if $DO_CL; then
  echo ""
  echo "================================================================"
  echo "  CL tables and convergence plots"
  echo "================================================================"

  cl_args=(
    --final-grids "$FINAL_GRIDS"
    --final-solutions "$FINAL_SOLUTIONS"
    --outdir "$FINAL_CASES"
    --reference-mesh "$REFERENCE_MESH"
    --reference-results "$REFERENCE_RESULTS"
    --reference-label "$REFERENCE_LABEL"
    --uniform-curvature "$UNIFORM_CURVATURE"
    --uniform-order "$UNIFORM_ORDER"
    --uniform-grids "$UNIFORM_GRIDS"
  )
  if [[ -n "$CASE_FILTER" ]]; then
    cl_args+=(--case "$CASE_FILTER")
  fi
  if [[ -n "$CURVATURE_FILTER" ]]; then
    cl_args+=(--curvature "$CURVATURE_FILTER")
  fi
  if $NO_UNIFORM; then
    cl_args+=(--no-uniform)
  fi

  "$PYTHON" postproc/process_final_cases.py "${cl_args[@]}"

  "$PYTHON" - "$FINAL_CASES" "$FINAL_FIGURES" "$CASE_FILTER" "$CURVATURE_FILTER" <<'PY'
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

final_cases = Path(sys.argv[1])
final_figures = Path(sys.argv[2])
case_filter = sys.argv[3] or None
curvature_filter = sys.argv[4] or None
final_figures.mkdir(parents=True, exist_ok=True)

copies = [
    ("cl_vs_cells.png", "cl_convergence.png"),
    ("cl_convergence.png", "cl_convergence_by_iteration.png"),
    ("cl_error.png", "cl_error.png"),
    ("cl_error_by_iteration.png", "cl_error_by_iteration.png"),
    ("cl_vs_logh.png", "cl_vs_log_cells.png"),
]
for src_name, dst_name in copies:
    src = final_cases / src_name
    if src.exists():
        dst = final_figures / dst_name
        shutil.copy2(src, dst)
        print(f"CL figure: {dst}  (from {src})")

csv_path = final_cases / "cl_values.csv"
if not csv_path.exists():
    raise SystemExit

with csv_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

by_case = defaultdict(list)
for row in rows:
    if case_filter and row["case"] != case_filter:
        continue
    if curvature_filter and row["curvature"] != curvature_filter:
        continue
    by_case[row["case"]].append(row)

if not by_case:
    print("Warning: no CL rows matched the requested case.")
else:
    for case, case_rows in sorted(by_case.items()):
        iters = sorted({int(row["iteration"]) for row in case_rows if row["iteration"] != ""})
        elems = [int(row["n_elements"]) for row in case_rows if row["n_elements"] != ""]
        if len(iters) < 2:
            print(
                f"Warning: {case} has only {len(iters)} solved adaptation iteration(s): {iters}. "
                "The P4final-style CL convergence plot will be a single adapted point plus uniform data. "
                "Run solve_final_grids.sh on more iter*.gri snapshots after adaptation produces them."
            )
        else:
            print(
                f"CL data: {case} has {len(iters)} solved iterations "
                f"({min(elems)} to {max(elems)} elements)."
            )
PY
fi

if $DO_INDICATORS; then
  echo ""
  echo "================================================================"
  echo "  Fixed-grid adjoint-weighted error indicators"
  echo "================================================================"

  indicator_args=(
    --solver ./euler_solver
    --final-grids "$FINAL_GRIDS"
    --final-solutions "$FINAL_SOLUTIONS"
    --outdir "${FINAL_CASES}/indicators"
  )
  if [[ -n "$CASE_FILTER" ]]; then
    indicator_args+=(--case "$CASE_FILTER")
  fi
  if [[ -n "$CURVATURE_FILTER" ]]; then
    indicator_args+=(--curvature "$CURVATURE_FILTER")
  fi
  if ! $RECOMPUTE_INDICATORS || $SKIP_EXISTING; then
    indicator_args+=(--skip-existing)
  fi
  if ! $RECOMPUTE_INDICATORS && ! $INDICATOR_ALL_ITERATIONS; then
    indicator_args+=(--skip-if-any-indicator-root data_steady)
  fi
  if $INDICATOR_ALL_ITERATIONS; then
    indicator_args+=(--all-iterations)
  fi

  "$PYTHON" postproc/compute_final_indicators.py "${indicator_args[@]}"
fi

if $DO_CONTOURS; then
  echo ""
  echo "================================================================"
  echo "  Mach contours"
  echo "================================================================"
  contour_args=(
    --final-grids "$FINAL_GRIDS"
    --final-solutions "$FINAL_SOLUTIONS"
    --outdir "$FINAL_FIGURES"
    --field mach
  )
  if [[ -n "$CASE_FILTER" ]]; then
    contour_args+=(--case "$CASE_FILTER")
  fi
  if [[ -n "$CURVATURE_FILTER" ]]; then
    contour_args+=(--curvature "$CURVATURE_FILTER")
  fi
  "$PYTHON" postproc/plot_final_contours.py "${contour_args[@]}"

  echo ""
  echo "================================================================"
  echo "  Aspect-ratio contours"
  echo "================================================================"
  contour_args=(
    --final-grids "$FINAL_GRIDS"
    --final-solutions "$FINAL_SOLUTIONS"
    --outdir "$FINAL_FIGURES"
    --field aspect_ratio
    --vmin 1
    --vmax 10
    --cmap viridis
  )
  if [[ -n "$CASE_FILTER" ]]; then
    contour_args+=(--case "$CASE_FILTER")
  fi
  if [[ -n "$CURVATURE_FILTER" ]]; then
    contour_args+=(--curvature "$CURVATURE_FILTER")
  fi
  "$PYTHON" postproc/plot_final_contours.py "${contour_args[@]}"

  if $DO_INDICATORS; then
    echo ""
    echo "================================================================"
    echo "  Adjoint-weighted error indicator contours"
    echo "================================================================"
    contour_args=(
      --final-grids "$FINAL_GRIDS"
      --final-solutions "$FINAL_SOLUTIONS"
      --indicator-root "${FINAL_CASES}/indicators"
      --indicator-root data_steady
      --outdir "$FINAL_FIGURES"
      --field adjoint_indicator
    )
    if [[ -n "$CASE_FILTER" ]]; then
      contour_args+=(--case "$CASE_FILTER")
    fi
    if [[ -n "$CURVATURE_FILTER" ]]; then
      contour_args+=(--curvature "$CURVATURE_FILTER")
    fi
    "$PYTHON" postproc/plot_final_contours.py "${contour_args[@]}"
  fi
fi

if $DO_CP; then
  echo ""
  echo "================================================================"
  echo "  Cp distributions"
  echo "================================================================"

  cp_args=(
    --final-grids "$FINAL_GRIDS"
    --final-solutions "$FINAL_SOLUTIONS"
    --outdir "$FINAL_FIGURES"
    --reference-mesh "$REFERENCE_MESH"
    --reference-results "$REFERENCE_RESULTS"
    --reference-label "$CP_REFERENCE_LABEL"
  )
  if [[ -n "$CASE_FILTER" ]]; then
    cp_args+=(--case "$CASE_FILTER")
  fi
  if [[ -n "$CURVATURE_FILTER" ]]; then
    cp_args+=(--curvature "$CURVATURE_FILTER")
  fi

  "$PYTHON" postproc/plot_final_cp.py "${cp_args[@]}"
fi

echo ""
echo "================================================================"
echo "  P4final-style postprocessing complete."
echo "  Cases  : ${FINAL_CASES}/"
echo "  Figures: ${FINAL_FIGURES}/"
echo "================================================================"
