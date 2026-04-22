#!/bin/bash
# ============================================================================
# run_data.sh — Generate all solver data needed for the Project 4 report
# ============================================================================
# Usage:  bash run_data.sh [--skip-existing] [--resume]
#
#   --skip-existing : Skip cases whose final results.bin already exists
#   --resume        : Resume interrupted cases from _temp_iter_latest_dg.bin
#                     (implies --skip-existing)
#
# This script runs:
#   1. Uniform refinement: p=0 steady on 2k, 8k, 32k, 128k grids
#   2. Higher-order uniform: p=1, p=2 steady on 2k, 8k grids (CFL=10 for p=2)
#   3. Adjoint-adapted refinement: p=0 on base.gri with 6 cycles
#
# Outputs land in data_steady/ and logs go to data_steady/*.log
# ============================================================================
set -e

if [[ -x "./euler_solver" ]]; then
  SOLVER="./euler_solver"
elif [[ -x "./euler_solver.exe" ]]; then
  SOLVER="./euler_solver.exe"
else
  echo "Error: solver binary not found. Expected ./euler_solver or ./euler_solver.exe"
  echo "Build it first with: bash build.sh"
  exit 1
fi
OUTDIR="data_steady"
FLUX="roe"

SKIP_EXISTING=false
RESUME=false
for arg in "$@"; do
  case "$arg" in
    --skip-existing) SKIP_EXISTING=true ;;
    --resume)        RESUME=true; SKIP_EXISTING=true ;;
  esac
done

mkdir -p "$OUTDIR"

# Helper: skip a run if output already exists (when --skip-existing)
should_skip() {
  local marker_file="$1"
  if $SKIP_EXISTING && [[ -f "$marker_file" ]]; then
    echo "  [SKIP] $marker_file already exists"
    return 0
  fi
  return 1
}

# Helper: find a resume IC file for a case, return path in RESUME_IC
find_resume_ic() {
  RESUME_IC=""
  if ! $RESUME; then return; fi
  local dg_temp="$1"
  if [[ -f "$dg_temp" ]]; then
    echo "  [RESUME] Found checkpoint: $dg_temp"
    RESUME_IC="$dg_temp"
  fi
}

# # ──────────────────────────────────────────────────────────────────────────────
# # 1. UNIFORM REFINEMENT — p=0 on q=1 meshes (2k, 8k, 32k, 128k)
# # ──────────────────────────────────────────────────────────────────────────────
# echo "================================================================"
# echo "  SECTION 1: Uniform refinement — p=0 on all mesh sizes"
# echo "================================================================"
# 
# for grid in 2k 8k 32k; do #128k
#   GRIDFILE="grids/${grid}.gri"
#   MARKER="${OUTDIR}/steady_${grid}_p0_results.bin"
# 
#   if [[ ! -f "$GRIDFILE" ]]; then
#     echo "  [WARN] Grid $GRIDFILE not found, skipping"
#     continue
#   fi
# 
#   if should_skip "$MARKER"; then continue; fi
# 
#   find_resume_ic "${OUTDIR}/steady_${grid}_p0_temp_iter_latest_dg.bin"
# 
#   echo "--- Running p=0 on ${grid} mesh ---"
#   $SOLVER "$GRIDFILE" 0 1.0 $FLUX 200000 steady $RESUME_IC \
#     2>&1 | tee "${OUTDIR}/${grid}_p0_run.log"
#   echo ""
# done
# 
# # Collect uniform Cl values into a single log for plot_cl_convergence.py
# echo "--- Collecting uniform Cl values ---"
# UNIFORM_LOG="${OUTDIR}/uniform_run.log"
# > "$UNIFORM_LOG"
# for grid in 2k 8k 32k; do #128k
#   LOG="${OUTDIR}/${grid}_p0_run.log"
#   if [[ -f "$LOG" ]]; then
#     cat "$LOG" >> "$UNIFORM_LOG"
#   fi
# done
# echo "  Uniform log: $UNIFORM_LOG"
# 
# # ──────────────────────────────────────────────────────────────────────────────
# # 2. HIGHER-ORDER UNIFORM — p=1 and p=2 on 2k, 8k meshes
# # ──────────────────────────────────────────────────────────────────────────────
# echo ""
# echo "================================================================"
# echo "  SECTION 2: Higher-order uniform — p=1, p=2 on 2k and 8k"
# echo "================================================================"
# 
# # p=1 runs (CFL=1 is fine, auto-chains from p=0)
# for grid in 2k 8k; do
#   GRIDFILE="grids/${grid}.gri"
#   MARKER="${OUTDIR}/steady_${grid}_p1_results.bin"
# 
#   if should_skip "$MARKER"; then continue; fi
# 
#   find_resume_ic "${OUTDIR}/steady_${grid}_p1_temp_iter_latest_dg.bin"
# 
#   echo "--- Running p=1 on ${grid} mesh ---"
#   $SOLVER "$GRIDFILE" 1 1.0 $FLUX 200000 steady $RESUME_IC \
#     2>&1 | tee "${OUTDIR}/${grid}_p1_run.log"
#   echo ""
# done
# 
# # p=2 runs (CFL=10 is stable for p=2, gives 10x speedup)
# for grid in 2k 8k; do
#   GRIDFILE="grids/${grid}.gri"
#   MARKER="${OUTDIR}/steady_${grid}_p2_results.bin"
# 
#   if should_skip "$MARKER"; then continue; fi
# 
#   find_resume_ic "${OUTDIR}/steady_${grid}_p2_temp_iter_latest_dg.bin"
# 
#   echo "--- Running p=2 CFL=10 on ${grid} mesh ---"
#   $SOLVER "$GRIDFILE" 2 10.0 $FLUX 200000 steady $RESUME_IC \
#     2>&1 | tee "${OUTDIR}/${grid}_p2_run.log"
#   echo ""
# done
# 
# # ──────────────────────────────────────────────────────────────────────────────
# # 3. ADJOINT-ADAPTED REFINEMENT — p=0 on 2k_q3.gri (q=3 curved mesh)
# # ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SECTION 3: Adjoint-adapted refinement — p=0 on 500_q3.gri"
echo "================================================================"

ADAPT_MARKER="${OUTDIR}/steady_500_q3_adjoint_indicators_cycle0.bin"
if should_skip "$ADAPT_MARKER"; then
  echo "  Adjoint adapt data already exists"
else
  echo "--- Running adjoint-adapt on 500_q3.gri (5 cycles, 25% fraction) ---"
  $SOLVER grids/500_q3.gri 0 1.0 $FLUX 200000 steady \
    --adjoint-adapt 1e-4 5 0.25 \
    2>&1 | tee "${OUTDIR}/adapt_run.log"
  echo ""
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  All data generation complete."
echo "  Results in: $OUTDIR/"
echo "================================================================"
