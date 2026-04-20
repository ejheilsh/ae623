#!/bin/bash
# ============================================================================
# run_postproc.sh — Generate all figures and tables for the Project 4 report
# ============================================================================
# Prerequisites: run run_data.sh first to populate data_steady/
#
# Usage:  bash run_postproc.sh
#
# Outputs go to report_figures/<section>/
# ============================================================================
set -e

OUTDIR="report_figures"
DATADIR="data_steady"
POSTPROC="postproc"

# Use Windows Python (has matplotlib/numpy); MSYS2 Python does not
PYTHON="/c/Users/William Zhang/AppData/Local/Programs/Python/Python313/python.exe"
if [[ ! -x "$PYTHON" ]]; then
  echo "Warning: Windows Python not found at $PYTHON, falling back to 'python'"
  PYTHON=python
fi

# Allow postproc scripts to import dg_utils, plot_cp, etc. without cd
export PYTHONPATH="$POSTPROC${PYTHONPATH:+:$PYTHONPATH}"

# Section subdirectories
DIR_SUMMARY="$OUTDIR/01_solution_summaries"
DIR_PROCESSED="$OUTDIR/02_processed_cases"
DIR_MACH="$OUTDIR/03_mach_contours"
DIR_ENTROPY="$OUTDIR/04_entropy_contours"
DIR_CP="$OUTDIR/05_cp_overlays"
DIR_CONVERGENCE="$OUTDIR/06_convergence"
DIR_CL="$OUTDIR/07_cl_convergence"
DIR_ADJOINT="$OUTDIR/08_adjoint"

mkdir -p "$DIR_SUMMARY" "$DIR_PROCESSED" "$DIR_MACH" "$DIR_ENTROPY" \
         "$DIR_CP" "$DIR_CONVERGENCE" "$DIR_CL" "$DIR_ADJOINT"

# ──────────────────────────────────────────────────────────────────────────────
# 1. SOLUTION SUMMARIES — 4-panel plots (Mach, entropy, cell res, convergence)
# ──────────────────────────────────────────────────────────────────────────────
#echo "================================================================"
#echo "  SECTION 1: Solution summary plots → $DIR_SUMMARY"
#echo "================================================================"
#
#for grid in 2k 8k 32k 128k; do
#  for p in 0 1 2; do
#    RES="${DATADIR}/steady_${grid}_p${p}_results.bin"
#    RESID="${DATADIR}/steady_${grid}_p${p}_residual.bin"
#    CELLRES="${DATADIR}/steady_${grid}_p${p}_cell_res.bin"
#    GRIDFILE="grids/${grid}.gri"
#
#    if [[ ! -f "$RES" ]] || [[ ! -f "$RESID" ]]; then continue; fi
#
#    OUTFILE="${DIR_SUMMARY}/summary_${grid}_p${p}.png"
#    echo "--- Solution summary: ${grid} p=${p} → $OUTFILE ---"
#    "$PYTHON" -c "
#from plot_results import plot_results
#plot_results('$GRIDFILE', '$RES', '$RESID', '$CELLRES',
#             show_plot=False, output_file='$OUTFILE')
#" 2>/dev/null || true
#  done
#done
#
## ──────────────────────────────────────────────────────────────────────────────
## 2. PROCESS STEADY CASES — JSON summaries + Cp + solution plots
## ──────────────────────────────────────────────────────────────────────────────
#echo ""
#echo "================================================================"
#echo "  SECTION 2: Process steady cases → $DIR_PROCESSED"
#echo "================================================================"
#
#for grid in 2k 8k 32k 128k; do
#  for p in 0 1 2; do
#    RES="${DATADIR}/steady_${grid}_p${p}_results.bin"
#    RESID="${DATADIR}/steady_${grid}_p${p}_residual.bin"
#    CELLRES="${DATADIR}/steady_${grid}_p${p}_cell_res.bin"
#    GRIDFILE="grids/${grid}.gri"
#
#    if [[ ! -f "$RES" ]] || [[ ! -f "$RESID" ]]; then continue; fi
#
#    CASEDIR="${DIR_PROCESSED}/steady_${grid}_p${p}"
#    echo "--- Processing: ${grid} p=${p} → $CASEDIR ---"
#    "$PYTHON" "$POSTPROC/process_steady_case.py" \
#      "$GRIDFILE" "$RES" "$RESID" "$CELLRES" \
#      --outdir "$CASEDIR" 2>/dev/null || true
#  done
#done
#
## ──────────────────────────────────────────────────────────────────────────────
## 3. MACH CONTOUR COMPARISON
## ──────────────────────────────────────────────────────────────────────────────
#echo ""
#echo "================================================================"
#echo "  SECTION 3: Mach contour plots → $DIR_MACH"
#echo "================================================================"
#
#for grid in 2k 8k; do
#  for p in 0 1 2; do
#    RES="${DATADIR}/steady_${grid}_p${p}_results.bin"
#    GRIDFILE="grids/${grid}.gri"
#
#    if [[ ! -f "$RES" ]]; then continue; fi
#
#    OUTFILE="${DIR_MACH}/mach_${grid}_p${p}.png"
#    echo "--- Mach contour: ${grid} p=${p} → $OUTFILE ---"
#    "$PYTHON" "$POSTPROC/plot_mach_contour.py" \
#      "$GRIDFILE" "$RES" \
#      --output "$OUTFILE" \
#      --title "${grid} mesh, p=${p}" \
#      --no-show 2>/dev/null || true
#  done
#done
#
## ──────────────────────────────────────────────────────────────────────────────
## 4. ENTROPY CONTOUR COMPARISON
## ──────────────────────────────────────────────────────────────────────────────
#echo ""
#echo "================================================================"
#echo "  SECTION 4: Entropy contour plots → $DIR_ENTROPY"
#echo "================================================================"
#
#for grid in 2k 8k; do
#  for p in 0 1 2; do
#    RES="${DATADIR}/steady_${grid}_p${p}_results.bin"
#    GRIDFILE="grids/${grid}.gri"
#
#    if [[ ! -f "$RES" ]]; then continue; fi
#
#    OUTFILE="${DIR_ENTROPY}/entropy_${grid}_p${p}.png"
#    echo "--- Entropy contour: ${grid} p=${p} → $OUTFILE ---"
#    "$PYTHON" "$POSTPROC/plot_entropy_contour.py" \
#      "$GRIDFILE" "$RES" \
#      --output "$OUTFILE" \
#      --title "${grid} mesh, p=${p}" \
#      --no-show 2>/dev/null || true
#  done
#done
#
## ──────────────────────────────────────────────────────────────────────────────
## 5. Cp OVERLAY — compare p=0, p=1, p=2 on same mesh
## ──────────────────────────────────────────────────────────────────────────────
#echo ""
#echo "================================================================"
#echo "  SECTION 5: Cp overlay plots → $DIR_CP"
#echo "================================================================"
#
#for grid in 2k 8k; do
#  GRIDFILE="grids/${grid}.gri"
#  ARGS=""
#  HAVE_ANY=false
#
#  for p in 0 1 2; do
#    RES="${DATADIR}/steady_${grid}_p${p}_results.bin"
#    if [[ -f "$RES" ]]; then
#      ARGS="$ARGS $RES p=${p}"
#      HAVE_ANY=true
#    fi
#  done
#
#  if $HAVE_ANY; then
#    OUTFILE="${DIR_CP}/cp_overlay_${grid}.png"
#    echo "--- Cp overlay: ${grid} → $OUTFILE ---"
#    "$PYTHON" "$POSTPROC/plot_cp_overlay.py" \
#      "$GRIDFILE" "$OUTFILE" \
#      $ARGS 2>/dev/null || true
#  fi
#done
#
## ──────────────────────────────────────────────────────────────────────────────
## 6. CONVERGENCE OVERLAY — residual histories for all orders on each mesh
## ──────────────────────────────────────────────────────────────────────────────
#echo ""
#echo "================================================================"
#echo "  SECTION 6: Convergence history overlay → $DIR_CONVERGENCE"
#echo "================================================================"
#
#for grid in 2k 8k; do
#  INPUTS=""
#  for p in 0 1 2; do
#    RESID="${DATADIR}/steady_${grid}_p${p}_residual.bin"
#    if [[ -f "$RESID" ]]; then
#      INPUTS="$INPUTS $RESID"
#    fi
#  done
#
#  if [[ -n "$INPUTS" ]]; then
#    OUTFILE="${DIR_CONVERGENCE}/convergence_${grid}.png"
#    echo "--- Convergence overlay: ${grid} → $OUTFILE ---"
#    "$PYTHON" "$POSTPROC/plot_convergence_overlay.py" \
#      $INPUTS \
#      -o "$OUTFILE" \
#      --title "${grid} mesh convergence" \
#      --trim-shared-prefix 2>/dev/null || true
#  fi
#done
#
## ──────────────────────────────────────────────────────────────────────────────
## 7. Cl CONVERGENCE — adjoint-adapted vs uniform refinement
## ──────────────────────────────────────────────────────────────────────────────
#echo ""
#echo "================================================================"
#echo "  SECTION 7: Cl convergence — adjoint vs uniform → $DIR_CL"
#echo "================================================================"
#
#echo "--- Cl convergence plot ---"
#"$PYTHON" "$POSTPROC/plot_cl_convergence.py" \
#  --adapt  "${DATADIR}/adapt_run.log" \
#  --uniform "${DATADIR}/uniform_run.log" \
#  --out     "${DIR_CL}/cl_convergence.png" \
#  --effectivity-out "${DIR_CL}/cl_effectivity.png" \
#  2>/dev/null || true
#
## ──────────────────────────────────────────────────────────────────────────────
## 8. ADJOINT VISUALIZATION — psi field + error indicators per cycle
## ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SECTION 8: Adjoint field + error indicator plots → $DIR_ADJOINT"
echo "================================================================"

for cycle in 0 1 2 3 4 5 6 7 8 9; do
  IND="${DATADIR}/steady_base_p0_adjoint_indicators_cycle${cycle}.bin"
  if [[ ! -f "$IND" ]]; then continue; fi

  echo "--- Adjoint cycle ${cycle} ---"
  # plot_adjoint.py saves to <output_dir>/adjoint_cycle<N>.png
  # Run it pointing at DATADIR, then copy result to report subfolder
  "$PYTHON" "$POSTPROC/plot_adjoint.py" "$DATADIR" "$cycle" 2>/dev/null || true
  if [[ -f "${DATADIR}/adjoint_cycle${cycle}.png" ]]; then
    cp "${DATADIR}/adjoint_cycle${cycle}.png" "${DIR_ADJOINT}/"
  fi
done

# ──────────────────────────────────────────────────────────────────────────────
# 9. PRINT Cl SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SECTION 9: Cl summary table (from solver logs)"
echo "================================================================"

echo ""
echo "UNIFORM REFINEMENT Cl VALUES:"
echo "Grid    | p=0         | p=1         | p=2"
echo "--------+-------------+-------------+-------------"
for grid in 2k 8k 32k 128k; do
  LINE="${grid}"
  for p in 0 1 2; do
    LOG="${DATADIR}/${grid}_p${p}_run.log"
    if [[ -f "$LOG" ]]; then
      CL=$(grep -oP 'Cl = \K[0-9.e+-]+' "$LOG" | tail -1)
      LINE="${LINE}    | ${CL:-n/a}"
    else
      LINE="${LINE}    | n/a"
    fi
  done
  echo "$LINE"
done

echo ""
echo "ADJOINT-ADAPTED Cl VALUES:"
if [[ -f "${DATADIR}/adapt_run.log" ]]; then
  grep -E 'Cl = ' "${DATADIR}/adapt_run.log" || true
fi

echo ""
echo "================================================================"
echo "  All postprocessing complete."
echo "  Figures saved to: $OUTDIR/"
echo "================================================================"
echo ""
echo "Directory structure:"
find "$OUTDIR" -type f -name '*.png' -o -name '*.json' -o -name '*.csv' | sort
