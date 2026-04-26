#!/bin/bash
# ============================================================================
# solve_final_grids.sh - Solve exported final_grids without touching data_steady
# ============================================================================
# Inputs:
#   final_grids/<ncells>/<curvature>/<solver_order>/iter<N>.gri
#
# Outputs:
#   final_solutions/<ncells>/<curvature>/<solver_order>/iter<N>_soln.bin
#   final_solutions/<ncells>/<curvature>/<solver_order>/iter<N>_soln_dg.bin
#   final_solutions/<ncells>/<curvature>/<solver_order>/iter<N>_residual.bin
#   final_solutions/<ncells>/<curvature>/<solver_order>/iter<N>_cell_res.bin
#   final_solutions/<ncells>/<curvature>/<solver_order>/iter<N>.log
#
# Examples:
#   bash solve_final_grids.sh
#   bash solve_final_grids.sh --skip-existing
#   bash solve_final_grids.sh --map-previous --skip-existing 2k q2 p0
#   bash solve_final_grids.sh --iter 10 8k q2 p0
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -x "${ROOT_DIR}/euler_solver" ]]; then
  SOLVER="${ROOT_DIR}/euler_solver"
elif [[ -x "${ROOT_DIR}/euler_solver.exe" ]]; then
  SOLVER="${ROOT_DIR}/euler_solver.exe"
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

SKIP_EXISTING=false
MAP_PREVIOUS=false
ITER_FILTER=""
ITERCAP=200000
CFL=1.0
FLUX="roe"
POSITIONAL=()

usage() {
  sed -n '1,18p' "$0"
  cat <<'EOF'

Options:
  --skip-existing   Skip iter files whose final solution already exists
  --map-previous    Seed iter<N> from iter<N-1> using the solver's mapped IC
  --iter N          Solve only iter<N>.gri
  --itercap N       Solver iteration cap (default: 200000)
  --cfl X           Solver CFL (default: 1.0)
  --flux roe|hlle   Numerical flux (default: roe)
  -h, --help        Show this help

Positional forms:
  none              Solve every case under final_grids/*/*/*
  <ncells> [q] [p]  Solve one case, e.g. 8k q2 p0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-existing)
      SKIP_EXISTING=true
      shift
      ;;
    --map-previous)
      MAP_PREVIOUS=true
      shift
      ;;
    --iter)
      ITER_FILTER="$2"
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

sort_paths() {
  "$PYTHON" - "$@" <<'PY'
import re
import sys

def key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)]

for item in sorted(sys.argv[1:], key=key):
    print(item)
PY
}

solve_case() {
  local ncells="$1"
  local curvature="$2"
  local order_tag="$3"
  local order="${order_tag#p}"
  local grid_dir="${ROOT_DIR}/final_grids/${ncells}/${curvature}/${order_tag}"
  local out_dir="${ROOT_DIR}/final_solutions/${ncells}/${curvature}/${order_tag}"

  if [[ ! -d "$grid_dir" ]]; then
    echo "Error: final grid directory not found: $grid_dir"
    return 1
  fi

  mkdir -p "$out_dir"

  local grid_files=()
  if [[ -n "$ITER_FILTER" ]]; then
    grid_files=("${grid_dir}/iter${ITER_FILTER}.gri")
  else
    shopt -s nullglob
    grid_files=("${grid_dir}"/iter*.gri)
    shopt -u nullglob
  fi

  if [[ ${#grid_files[@]} -eq 0 ]]; then
    echo "Error: no iter*.gri files found in $grid_dir"
    return 1
  fi

  echo ""
  echo "================================================================"
  echo "  Solving final grids: ${ncells}/${curvature}/${order_tag}"
  echo "  CFL=${CFL}, flux=${FLUX}, itercap=${ITERCAP}"
  if $MAP_PREVIOUS; then
    echo "  mapped IC from previous solved iteration: enabled"
  fi
  echo "================================================================"

  while IFS= read -r grid_file; do
    if [[ ! -f "$grid_file" ]]; then
      echo "Error: grid file not found: $grid_file"
      return 1
    fi

    local stem iter_num soln_file dg_file residual_file cell_res_file log_file
    stem="$(basename "$grid_file" .gri)"
    iter_num="${stem#iter}"
    soln_file="${out_dir}/${stem}_soln.bin"
    dg_file="${out_dir}/${stem}_soln_dg.bin"
    residual_file="${out_dir}/${stem}_residual.bin"
    cell_res_file="${out_dir}/${stem}_cell_res.bin"
    log_file="${out_dir}/${stem}.log"

    if $SKIP_EXISTING && [[ -f "$soln_file" && -f "$dg_file" ]]; then
      echo "[SKIP] ${soln_file} already exists"
      continue
    fi

    local scratch
    scratch="$(mktemp -d "${TMPDIR:-/tmp}/final_grid_solve.XXXXXX")"
    cleanup() {
      rm -rf "$scratch"
    }
    trap cleanup EXIT

    local solver_args=("$grid_file" "$order" "$CFL" "$FLUX" "$ITERCAP" steady)
    if $MAP_PREVIOUS && [[ "$iter_num" =~ ^[0-9]+$ ]] && (( iter_num > 0 )); then
      local prev_iter prev_grid prev_soln
      prev_iter=$((iter_num - 1))
      prev_grid="${grid_dir}/iter${prev_iter}.gri"
      prev_soln="${out_dir}/iter${prev_iter}_soln.bin"
      if [[ -f "$prev_grid" && -f "$prev_soln" ]]; then
        solver_args+=(--map-ic "$prev_grid" "$prev_soln")
        echo "--- Solving ${grid_file} with mapped IC from iter${prev_iter} ---"
      else
        echo "--- Solving ${grid_file} ---"
        echo "  mapped IC unavailable: missing ${prev_grid} or ${prev_soln}"
      fi
    else
      echo "--- Solving ${grid_file} ---"
    fi

    (
      cd "$scratch"
      "$SOLVER" "${solver_args[@]}"
    ) 2>&1 | tee "$log_file"

    local solver_prefix scratch_data expected_results
    solver_prefix="steady_${stem}_p${order}"
    scratch_data="${scratch}/data_steady"
    expected_results="${scratch_data}/${solver_prefix}_results.bin"

    if [[ ! -f "$expected_results" ]]; then
      echo "Error: expected solver output was not created: $expected_results"
      echo "       See log: $log_file"
      return 1
    fi

    cp "$expected_results" "$soln_file"
    cp "${scratch_data}/${solver_prefix}_results_dg.bin" "$dg_file"
    cp "${scratch_data}/${solver_prefix}_residual.bin" "$residual_file"
    cp "${scratch_data}/${solver_prefix}_cell_res.bin" "$cell_res_file"

    cleanup
    trap - EXIT

    echo "  wrote ${soln_file}"
  done < <(sort_paths "${grid_files[@]}")
}

CASE_SPECS=()
if [[ ${#POSITIONAL[@]} -eq 0 ]]; then
  shopt -s nullglob
  case_dirs=("${ROOT_DIR}"/final_grids/*/*/*)
  shopt -u nullglob
  if [[ ${#case_dirs[@]} -eq 0 ]]; then
    echo "Error: no final grid case directories found under final_grids/"
    exit 1
  fi
  while IFS= read -r case_dir; do
    rel="${case_dir#${ROOT_DIR}/final_grids/}"
    IFS=/ read -r ncells curvature order_tag <<< "$rel"
    CASE_SPECS+=("${ncells} ${curvature} ${order_tag}")
  done < <(sort_paths "${case_dirs[@]}")
else
  NCELLS="${POSITIONAL[0]}"
  CURVATURE="$(normalize_curvature "${POSITIONAL[1]:-q1}")"
  ORDER_TAG="$(normalize_solver_order "${POSITIONAL[2]:-p0}")"
  CASE_SPECS=("${NCELLS} ${CURVATURE} ${ORDER_TAG}")
fi

for case_spec in "${CASE_SPECS[@]}"; do
  read -r ncells curvature order_tag <<< "$case_spec"
  solve_case "$ncells" "$curvature" "$order_tag"
done

echo ""
echo "All requested final grids solved."
