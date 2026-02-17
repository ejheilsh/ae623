# Unsteady Euler Solver - Quick Guide

```bash
./euler_solver <mesh> <order> <CFL> <flux> <maxiter> unsteady [ic_file]
```

```bash
# Steady solver automatically saves to data_steady/steady_<gridname>_results.bin
./euler_solver grids/2k.gri 1 0.5 hlle 10000 steady

# Unsteady solver can then use this as IC
./euler_solver grids/2k.gri 1 0.1 hlle 5000 unsteady data_steady/steady_2k_results.bin
```

**Output:**
- **Steady mode**: Results saved to `data_steady/steady_<gridname>_*.bin`
- **Unsteady mode**: 
  - Snapshots saved to `data/results_<time>_<snapshot>.bin` (every 100 iterations)
  - Final files: `results.bin`, `residual.bin`, `cell_res.bin` (in current directory)

**Note:** Using a converged steady-state solution as the initial condition eliminates the initial transient and allows the unsteady wake effects to be observed immediately.

## Postprocessing

**Single command to generate all plots:**
```bash
python3 postproc/plot_unsteady.py <meshfile> data/ [output_dir]
```

**Example:**
```bash
python3 postproc/plot_unsteady.py grids/coarse.gri data/
# or specify custom output directory:
python3 postproc/plot_unsteady.py grids/coarse.gri data/ my_plots/
```

**Output:**
- `unsteady_plots/entropy_field_XXXX_tY.YYYYYY.png` - Entropy + Mach at each timestep
- `unsteady_plots/force_history.png` - Force coefficients (Cf_x, Cf_y) vs time
- Terminal output with force coefficient statistics

## Complete Workflow

```bash
# 1. Run unsteady simulation
./euler_solver grids/coarse.gri 1 0.1 hlle 10000 unsteady

# 2. Generate all plots
python3 postproc/plot_unsteady.py grids/coarse.gri data/

# 3. View results
open entropy_field_*.png force_history.png
```

## For Project Requirements

Run on multiple configurations:

```bash
# Coarse 1st order
./euler_solver grids/coarse.gri 1 0.1 hlle 15000 unsteady
mkdir -p results/coarse_1st && mv data/*.bin results/coarse_1st/
python3 postproc/plot_unsteady.py grids/coarse.gri results/coarse_1st/

# Coarse 2nd order  
./euler_solver grids/coarse.gri 2 0.1 hlle 15000 unsteady
mkdir -p results/coarse_2nd && mv data/*.bin results/coarse_2nd/
python3 postproc/plot_unsteady.py grids/coarse.gri results/coarse_2nd/

# Fine 1st order
./euler_solver grids/fine.gri 1 0.1 hlle 15000 unsteady
mkdir -p results/fine_1st && mv data/*.bin results/fine_1st/
python3 postproc/plot_unsteady.py grids/fine.gri results/fine_1st/

# Fine 2nd order
./euler_solver grids/fine.gri 2 0.1 hlle 15000 unsteady
mkdir -p results/fine_2nd && mv data/*.bin results/fine_2nd/
python3 postproc/plot_unsteady.py grids/fine.gri results/fine_2nd/
```

Then compare entropy plots and force histories across configurations.

## Notes

- Default snapshot interval: 100 iterations (adjust in `Solver.cpp` line ~396)
- Wake passing period ≈ 0.018 time units
- CFL = 0.1-0.3 recommended for stability
- Run to t ≈ 0.1-0.15 to see multiple wake cycles
