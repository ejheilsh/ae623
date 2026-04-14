#!/usr/bin/env bash
# run_paper.sh - Generate all figures needed for the Project 4 report.
#
# Usage (from repo root, via MSYS2 or any bash):
#   bash run_paper.sh
#
# What it does:
#   1. Builds the solver
#   2. Runs adjoint-based adaptation on base.gri (p=0, 6 cycles) → adapt_run.log
#   3. Runs uniform steady on 2k / 8k / 32k grids → uniform_run.log
#   4. Runs steady on 128k grid as reference → ref_run.log
#   5. Calls plot_adjoint.py for each adaptation cycle
#   6. Calls plot_cl_convergence.py to produce Cl vs DoFs and effectivity plots
#
# Output directory: output_final/
set -euo pipefail

SOLVER="./euler_solver.exe"     # Windows; change to ./euler_solver on Linux/Mac
if [ ! -f "$SOLVER" ]; then
    SOLVER="./euler_solver"
fi

LOG_DIR="data_steady"
OUT_DIR="output_final"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# ── 1. Build ──────────────────────────────────────────────────────────────────
echo "=== Building solver ==="
bash build.sh
echo ""

# ── 2. Adjoint-adapted run (base.gri, 6 cycles, p=0) ─────────────────────────
echo "=== Adjoint-adaptation run (base.gri, 6 cycles) ==="
"$SOLVER" grids/base.gri 0 1.0 roe 200000 steady \
    --adjoint-adapt 1e-3 6 0.25 \
    2>&1 | tee "$LOG_DIR/adapt_run.log"
echo ""

# ── 3. Uniform refinement: 2k / 8k / 32k ────────────────────────────────────
echo "=== Uniform refinement runs ==="
for GRID in 2k 8k 32k; do
    echo "--- grids/${GRID}.gri ---"
    "$SOLVER" grids/${GRID}.gri 0 1.0 roe 200000 steady \
        2>&1 | tee -a "$LOG_DIR/uniform_run.log"
done
echo ""

# ── 4. High-resolution reference (128k) ──────────────────────────────────────
echo "=== Reference run (128k.gri) ==="
"$SOLVER" grids/128k.gri 0 1.0 roe 200000 steady \
    2>&1 | tee "$LOG_DIR/ref_run.log"
echo ""

# ── 5. Adjoint field plots for each completed cycle ───────────────────────────
echo "=== Generating adjoint field plots ==="
N_CYCLES=0
for BIN in "$LOG_DIR"/steady_base_p0_adjoint_psi_cycle*_dg.bin; do
    [ -f "$BIN" ] || continue
    CYCLE=$(echo "$BIN" | grep -oP '(?<=cycle)\d+')
    echo "  Plotting cycle $CYCLE..."
    python postproc/plot_adjoint.py "$LOG_DIR" "$CYCLE"
    cp "$LOG_DIR/adjoint_cycle${CYCLE}.png" "$OUT_DIR/adjoint_cycle${CYCLE}.png"
    N_CYCLES=$((N_CYCLES + 1))
done
echo "  Generated $N_CYCLES adjoint field plots."
echo ""

# ── 6. Cl vs DoFs convergence + effectivity plots ───────────────────────────
echo "=== Generating convergence plots ==="
python postproc/plot_cl_convergence.py \
    --adapt   "$LOG_DIR/adapt_run.log" \
    --uniform "$LOG_DIR/uniform_run.log" \
    --ref-log "$LOG_DIR/ref_run.log" \
    --out     "$OUT_DIR/cl_convergence.png" \
    --effectivity-out "$OUT_DIR/cl_effectivity.png"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo "=== Done. Files written to $OUT_DIR/ ==="
ls -lh "$OUT_DIR"/
