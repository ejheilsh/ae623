#!/usr/bin/env bash
set -euo pipefail
SOLVER="./euler_solver.exe"
LOG_DIR="data_steady"
OUT_DIR="output_final"
mkdir -p "$OUT_DIR"

echo "=== Uniform runs (2k, 8k) ==="
"$SOLVER" grids/2k.gri 0 1.0 roe 200000 steady 2>&1 | tee -a "$LOG_DIR/uniform_run.log"
"$SOLVER" grids/8k.gri 0 1.0 roe 200000 steady 2>&1 | tee -a "$LOG_DIR/uniform_run.log"
echo "=== Uniform runs done ==="

echo "=== Generating adjoint field plots ==="
for BIN in "$LOG_DIR"/steady_base_p0_adjoint_psi_cycle*_dg.bin; do
    [ -f "$BIN" ] || continue
    CYCLE=$(echo "$BIN" | grep -oP '(?<=cycle)\d+')
    echo "  Plotting cycle $CYCLE..."
    python postproc/plot_adjoint.py "$LOG_DIR" "$CYCLE"
    cp "$LOG_DIR/adjoint_cycle${CYCLE}.png" "$OUT_DIR/adjoint_cycle${CYCLE}.png"
done
echo "  Done adjoint plots."

echo "=== Generating convergence plots ==="
python postproc/plot_cl_convergence.py \
    --adapt   "$LOG_DIR/adapt_run.log" \
    --uniform "$LOG_DIR/uniform_run.log" \
    --out     "$OUT_DIR/cl_convergence.png" \
    --effectivity-out "$OUT_DIR/cl_effectivity.png"
echo "=== All done. Files in $OUT_DIR/ ==="
ls -lh "$OUT_DIR"/
