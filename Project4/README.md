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

## Adjoint-Based Mesh Adaptation (Project 4)

### Usage
```bash
./euler_solver <mesh> <order> <CFL> <flux> <maxiter> steady --adjoint-adapt [tol] [max_cycles] [fraction]
```

**Example:**
```bash
./euler_solver grids/base.gri 0 1.0 roe 50000 steady --adjoint-adapt 1e-3 5 0.25
```

- `tol`: stop when estimated |delta_Cl| < tol (default 1e-3)
- `max_cycles`: maximum adaptation cycles (default 5)
- `fraction`: fraction of elements to refine each cycle (default 0.25)

### Tests
```bash
# Build and run all 3 tests (flux Jacobian, assembled Jacobian, adjoint sensitivity)
g++ -O2 -std=c++17 test_results/test_jacobian.cpp src/State.cpp src/Mesh.cpp src/Fluxes.cpp src/Solver.cpp src/Adjoint.cpp src/MeshRefinement.cpp -o test_jacobian.exe -lstdc++fs
./test_jacobian.exe grids/base.gri
```

### Implementation Status
- **Working:** Jacobian assembly, adjoint solve, dCl/dU, sensitivity (psi^T * dR/dalpha), error indicators, prolongation (p→p+1), mesh bisection, solution interpolation
- **Known issue:** Adaptation cycles after the first may diverge on curved meshes (q=2) because bisection creates straight (q=1) child elements. Use straight meshes (e.g. `coarse.gri`) or implement curved midpoint insertion in `bisectMarkedElements`.

### Key Files
- `src/Adjoint.cpp` — adjoint solve, sensitivity, error indicators
- `src/MeshRefinement.cpp` — element marking, longest-edge bisection, solution interpolation
- `src/Solver.cpp` — `prolongP1toP2` (nodal interpolation for p→p+1)
- `src/main.cpp` — adaptation loop (search for `--adjoint-adapt`)


## Quick Start (Windows)

Run the adjoint-based adaptive solver on the baseline curved mesh:

```powershell
# Arguments: <mesh> <p_order> <CFL> <flux> <max_iter> steady --adjoint-adapt <tol> <max_cycles> <refine_fraction>
#   p_order=0          → piecewise-constant (DG p=0)
#   CFL=1.0            → time step scaling
#   flux=roe           → Roe upwind flux
#   max_iter=200000    → iteration budget per cycle
#   tol=1e-3           → stop when estimated |ΔCl| < tol
#   max_cycles=5       → run up to 5 adaptation cycles
#   refine_fraction=0.25 → refine the top 25% of elements by error indicator
.\euler_solver.exe grids/base.gri 0 1.0 roe 200000 steady --adjoint-adapt 1e-3 5 0.25
```

Results are saved per-cycle to `data_steady/`. Plot the adjoint field and error indicators for any cycle:

```powershell
# Cycle 0 (89 elements)
python postproc/plot_adjoint.py data_steady 0

# Cycle 1 (178 elements), etc.
python postproc/plot_adjoint.py data_steady 1
```