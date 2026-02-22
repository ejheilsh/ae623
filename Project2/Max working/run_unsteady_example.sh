#!/bin/bash
# Runs unsteady simulations for all mesh/order combinations and saves snapshot
# bins into named per-case directories for later postprocessing.

echo "======================================================================"
echo "Unsteady Simulation Suite"
echo "======================================================================"

# ── Global settings ──────────────────────────────────────────────────────────
MESH_LIST=("grids/2k.gri" "grids/8k.gri" "grids/32k.gri")
ORDER_LIST=(1 2)
CFL=0.1
FLUX="hlle"
MAXITER=100000000   # effectively unlimited — t_end controls termination
T_END=300
MODE="unsteady"

# rm -rf unsteady_data_*
rm -f euler_solver
./build.sh

# ── Loop over every mesh × order combination ─────────────────────────────────
for MESH in "${MESH_LIST[@]}"; do
    GRID_NAME=$(basename $MESH .gri)

    for ORDER in "${ORDER_LIST[@]}"; do

        # Directory where snapshot bins and force data will be saved
        CASE_DIR="unsteady_data_${ORDER}_${GRID_NAME}"
        IC_FILE="data_steady/steady_${GRID_NAME}_results.bin"

        echo ""
        echo "======================================================================"
        echo "Case: order=${ORDER}  mesh=${GRID_NAME}  →  ${CASE_DIR}"
        echo "======================================================================"

        # Step 0: prepare output dir and clean the shared data/ staging area
        mkdir -p "$CASE_DIR"
        ./clean_data.sh > /dev/null

        # Step 1: run steady solver if IC is missing
        if [ ! -f "$IC_FILE" ]; then
            echo "  Running steady solver to generate IC..."
            ./euler_solver $MESH 1 1.0 $FLUX 50000 steady
            echo ""
        fi

        # Step 2: run unsteady solver (snapshots go to data/)
        echo "  Running: ./euler_solver $MESH $ORDER $CFL $FLUX $MAXITER $MODE $IC_FILE $T_END"
        ./euler_solver $MESH $ORDER $CFL $FLUX $MAXITER $MODE $IC_FILE $T_END

        if [ $? -ne 0 ]; then
            echo "  ERROR: solver failed for $CASE_DIR — aborting."
            exit 1
        fi

        # Step 3: move snapshot bins into the case directory
        BIN_COUNT=$(ls data/results_*.bin 2>/dev/null | wc -l | tr -d ' ')
        if [ "$BIN_COUNT" -eq 0 ]; then
            echo "  WARNING: no snapshots found in data/ — skipping move."
        else
            mv data/results_*.bin "$CASE_DIR/"
            echo "  Moved $BIN_COUNT snapshots → $CASE_DIR/"
        fi

    done
done

echo ""
echo "======================================================================"
echo "All cases complete. Generating overlay force coefficient plot..."
echo "======================================================================"
python3 postproc/plot_force_overlay.py

echo ""
echo "Done. Case directories:"
for MESH in "${MESH_LIST[@]}"; do
    GRID_NAME=$(basename $MESH .gri)
    for ORDER in "${ORDER_LIST[@]}"; do
        echo "  unsteady_data_${ORDER}_${GRID_NAME}/"
    done
done
echo "  force_coefficient_overlay.png"
