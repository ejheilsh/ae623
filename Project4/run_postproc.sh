#!/bin/bash
# ============================================================================
# run_postproc.sh — Generate all figures and tables for the Project 4 report
# ============================================================================
# Prerequisites: run run_data.sh first to populate data_steady/
#
# Usage:  bash run_postproc.sh
#
# Outputs go to report_figures/
# ============================================================================
set -e

OUTDIR="report_figures"
DATADIR="data_steady"
POSTPROC="postproc"

mkdir -p "$OUTDIR"

# ──────────────────────────────────────────────────────────────────────────────
# 1. SOLUTION SUMMARIES — 4-panel plots (Mach, entropy, cell res, convergence)
# ──────────────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  SECTION 1: Solution summary plots"
echo "================================================================"

for grid in 2k 8k 32k 128k; do
  for p in 0 1 2; do
    RES="${DATADIR}/steady_${grid}_p${p}_results.bin"
    RESID="${DATADIR}/steady_${grid}_p${p}_residual.bin"
    CELLRES="${DATADIR}/steady_${grid}_p${p}_cell_res.bin"
    GRIDFILE="grids/${grid}.gri"

    if [[ ! -f "$RES" ]] || [[ ! -f "$RESID" ]]; then continue; fi

    echo "--- Solution summary: ${grid} p=${p} ---"
    cd "$POSTPROC"
    python plot_results.py \
      "../$GRIDFILE" "../$RES" "../$RESID" "../$CELLRES" \
      --no-show 2>/dev/null || true
    cd ..
    # move output if created in postproc dir
  done
done

# ──────────────────────────────────────────────────────────────────────────────
# 2. PROCESS STEADY CASES — JSON summaries + Cp + solution plots
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SECTION 2: Process steady cases (summaries, Cp distributions)"
echo "================================================================"

for grid in 2k 8k 32k 128k; do
  for p in 0 1 2; do
    RES="${DATADIR}/steady_${grid}_p${p}_results.bin"
    RESID="${DATADIR}/steady_${grid}_p${p}_residual.bin"
    CELLRES="${DATADIR}/steady_${grid}_p${p}_cell_res.bin"
    GRIDFILE="grids/${grid}.gri"

    if [[ ! -f "$RES" ]] || [[ ! -f "$RESID" ]]; then continue; fi

    CASEDIR="${OUTDIR}/steady_${grid}_p${p}"
    echo "--- Processing: ${grid} p=${p} ---"
    cd "$POSTPROC"
    python process_steady_case.py \
      "../$GRIDFILE" "../$RES" "../$RESID" "../$CELLRES" \
      --outdir "../$CASEDIR" 2>/dev/null || true
    cd ..
  done
done

# ──────────────────────────────────────────────────────────────────────────────
# 3. MACH CONTOUR COMPARISON — side-by-side for different p on same mesh
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SECTION 3: Mach contour plots"
echo "================================================================"

for grid in 2k 8k; do
  for p in 0 1 2; do
    RES="${DATADIR}/steady_${grid}_p${p}_results.bin"
    GRIDFILE="grids/${grid}.gri"

    if [[ ! -f "$RES" ]]; then continue; fi

    echo "--- Mach contour: ${grid} p=${p} ---"
    cd "$POSTPROC"
    python plot_mach_contour.py \
      "../$GRIDFILE" "../$RES" \
      --output "../${OUTDIR}/mach_${grid}_p${p}.png" \
      --title "${grid} mesh, p=${p}" \
      --no-show 2>/dev/null || true
    cd ..
  done
done

# ──────────────────────────────────────────────────────────────────────────────
# 4. ENTROPY CONTOUR COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SECTION 4: Entropy contour plots"
echo "================================================================"

for grid in 2k 8k; do
  for p in 0 1 2; do
    RES="${DATADIR}/steady_${grid}_p${p}_results.bin"
    GRIDFILE="grids/${grid}.gri"

    if [[ ! -f "$RES" ]]; then continue; fi

    echo "--- Entropy contour: ${grid} p=${p} ---"
    cd "$POSTPROC"
    python plot_entropy_contour.py \
      "../$GRIDFILE" "../$RES" \
      --output "../${OUTDIR}/entropy_${grid}_p${p}.png" \
      --title "${grid} mesh, p=${p}" \
      --no-show 2>/dev/null || true
    cd ..
  done
done

# ──────────────────────────────────────────────────────────────────────────────
# 5. Cp OVERLAY — compare p=0, p=1, p=2 on same mesh
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SECTION 5: Cp overlay plots (p=0 vs p=1 vs p=2 per mesh)"
echo "================================================================"

for grid in 2k 8k; do
  GRIDFILE="grids/${grid}.gri"
  ARGS=""
  HAVE_ANY=false

  for p in 0 1 2; do
    RES="${DATADIR}/steady_${grid}_p${p}_results.bin"
    if [[ -f "$RES" ]]; then
      ARGS="$ARGS $RES p=${p}"
      HAVE_ANY=true
    fi
  done

  if $HAVE_ANY; then
    echo "--- Cp overlay: ${grid} ---"
    cd "$POSTPROC"
    python plot_cp_overlay.py \
      "../$GRIDFILE" "../${OUTDIR}/cp_overlay_${grid}.png" \
      $ARGS 2>/dev/null || true
    cd ..
  fi
done

# ──────────────────────────────────────────────────────────────────────────────
# 6. CONVERGENCE OVERLAY — residual histories for all orders on each mesh
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SECTION 6: Convergence history overlay"
echo "================================================================"

for grid in 2k 8k; do
  INPUTS=""
  for p in 0 1 2; do
    RESID="${DATADIR}/steady_${grid}_p${p}_residual.bin"
    if [[ -f "$RESID" ]]; then
      INPUTS="$INPUTS $RESID"
    fi
  done

  if [[ -n "$INPUTS" ]]; then
    echo "--- Convergence overlay: ${grid} ---"
    cd "$POSTPROC"
    python plot_convergence_overlay.py \
      $INPUTS \
      -o "../${OUTDIR}/convergence_${grid}.png" \
      --title "${grid} mesh convergence" \
      --trim-shared-prefix 2>/dev/null || true
    cd ..
  fi
done

# ──────────────────────────────────────────────────────────────────────────────
# 7. Cl CONVERGENCE — adjoint-adapted vs uniform refinement
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SECTION 7: Cl convergence — adjoint vs uniform"
echo "================================================================"

echo "--- Cl convergence plot ---"
cd "$POSTPROC"
python plot_cl_convergence.py \
  --adapt  "../${DATADIR}/adapt_run.log" \
  --uniform "../${DATADIR}/uniform_run.log" \
  --out     "../${OUTDIR}/cl_convergence.png" \
  --effectivity-out "../${OUTDIR}/cl_effectivity.png" \
  2>/dev/null || true
cd ..

# ──────────────────────────────────────────────────────────────────────────────
# 8. ADJOINT VISUALIZATION — psi field + error indicators per cycle
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SECTION 8: Adjoint field + error indicator plots"
echo "================================================================"

for cycle in 0 1 2 3 4 5; do
  IND="${DATADIR}/steady_base_p0_adjoint_indicators_cycle${cycle}.bin"
  if [[ ! -f "$IND" ]]; then continue; fi

  echo "--- Adjoint cycle ${cycle} ---"
  cd "$POSTPROC"
  python plot_adjoint.py "../$DATADIR" "$cycle" 2>/dev/null || true
  cd ..
  # Move generated png to report dir
  if [[ -f "${DATADIR}/adjoint_cycle${cycle}.png" ]]; then
    cp "${DATADIR}/adjoint_cycle${cycle}.png" "${OUTDIR}/"
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
