#!/bin/bash
# Example workflow for running unsteady simulations and generating plots

echo "======================================================================"
echo "Example Unsteady Simulation Workflow"
echo "======================================================================"
echo ""
echo "This script demonstrates how to:"
echo "  1. Run an unsteady simulation"
echo "  2. Generate entropy field plots at various timesteps"
echo "  3. Compute force coefficient histories"
echo ""

# Configuration
MESH="grids/8k.gri"
ORDER=2
CFL=0.5
FLUX="hlle"
MAXITER=1000000
MODE="unsteady"
T_END=1000000

# Extract grid name for IC file
GRID_NAME=$(basename $MESH .gri)
IC_FILE="data_steady/steady_${GRID_NAME}_results.bin"

echo "Configuration:"
echo "  Mesh: $MESH"
echo "  Order: $ORDER"
echo "  CFL: $CFL"
echo "  Flux: $FLUX"
echo "  Max Iterations: $MAXITER"
echo "  Mode: $MODE"
echo "  t_end: $T_END"
echo "  Initial Condition: $IC_FILE"
echo ""

# Step 0: Clean the files
./clean_data.sh

# Check if IC file exists, if not run steady first
if [ ! -f "$IC_FILE" ]; then
    echo "Initial condition file not found. Running steady solver first..."
    ./euler_solver $MESH $ORDER 0.1 $FLUX 100000 steady
    echo ""
fi

# Step 1: Run the simulation
echo "Step 1: Running unsteady simulation..."
echo "Command: ./euler_solver $MESH $ORDER $CFL $FLUX $MAXITER $MODE $IC_FILE $T_END"
echo ""

./euler_solver $MESH $ORDER $CFL $FLUX $MAXITER $MODE $IC_FILE $T_END

if [ $? -ne 0 ]; then
    echo "Error: Simulation failed!"
    exit 1
fi

echo ""
echo "Simulation complete!"
echo ""

# Check for snapshot files in data/
SNAPSHOT_COUNT=$(ls data/results_*.bin 2>/dev/null | wc -l)
echo "Found $SNAPSHOT_COUNT snapshot files in data/"
echo ""

if [ $SNAPSHOT_COUNT -eq 0 ]; then
    echo "Warning: No snapshot files found in data/!"
    echo "The simulation may not have run long enough or snapshots weren't saved."
    exit 1
fi

# Step 2: Generate plots
echo "Step 2: Generating entropy field plots and force history..."
echo "Command: python3 postproc/plot_unsteady.py $MESH data/ unsteady_plots"
echo ""

python3 postproc/plot_unsteady.py $MESH data/ unsteady_plots

if [ $? -ne 0 ]; then
    echo "Error: Plotting failed!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Workflow complete!"
echo "======================================================================"
echo ""
echo "Generated files:"
echo "  - unsteady_plots/entropy_field_*.png - Entropy fields at various times"
echo "  - unsteady_plots/force_history.png - Force coefficient time history"
echo ""
