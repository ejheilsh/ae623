# AE623 Project 4 — Adjoint-Based Mesh Adaptation

DG Euler solver with adjoint-driven error estimation and mesh refinement for NACA 0012 at M∞=0.5, α=2°.

## Quick Start

```bash
# 1. Build (MSYS2 UCRT64)
bash build.sh

# 2. Generate all solver data
bash run_data.sh              # full run
bash run_data.sh --skip-existing  # resume / skip completed cases

# 3. Generate all report figures
bash run_postproc.sh
```

On Windows, launch via MSYS2:
```powershell
C:\msys64\msys2_shell.cmd -ucrt64 -defterm -no-start -here -c "bash run_data.sh"
```

## What `run_data.sh` Runs

| Section | Cases | Details |
|---------|-------|---------|
| 1. Uniform refinement | p=0 on 2k, 8k, 32k, 128k grids | CFL=1, Roe flux, 200k iter cap |
| 2. Higher-order | p=1, p=2 on 2k, 8k grids | CFL=10 for p=2 (stable, 10× speedup) |
| 3. Adjoint-adapted | p=0 on `base.gri` (89 elements) | 6 cycles, 25% refinement fraction |

Solver data and logs are saved to `data_steady/`.

## What `run_postproc.sh` Generates

All figures are saved to `report_figures/`.

| Section | Output | Script used |
|---------|--------|-------------|
| Solution summaries | 4-panel plots per case | `plot_results.py` |
| Processed cases | JSON/CSV summaries + Cp + solution PNGs | `process_steady_case.py` |
| Mach contours | Per mesh/order | `plot_mach_contour.py` |
| Entropy contours | Per mesh/order | `plot_entropy_contour.py` |
| Cp overlays | p=0 vs p=1 vs p=2 per mesh | `plot_cp_overlay.py` |
| Convergence overlays | Residual histories per mesh | `plot_convergence_overlay.py` |
| Cl convergence | Adjoint-adapted vs uniform | `plot_cl_convergence.py` |
| Adjoint fields | ψ + error indicators per cycle | `plot_adjoint.py` |
| Cl summary table | Printed to terminal | grep from logs |

## Solver CLI Reference

```bash
./euler_solver <mesh> <order> <CFL> <flux> <maxiter> <mode> [options]
```

- **Modes:** `steady`, `unsteady`, `freestream`
- **Fluxes:** `roe`, `hlle`
- For p>0 steady without an IC file, the solver auto-chains a p=0 solve first.

### Adjoint adaptation
```bash
./euler_solver grids/base.gri 0 1.0 roe 200000 steady --adjoint-adapt [tol] [max_cycles] [fraction]
```
- `tol` — stop when |ΔCl| < tol (default 1e-4)
- `max_cycles` — max adaptation cycles (default 10)
- `fraction` — fraction of elements to refine (default 0.25)

### Output files
- **Steady:** `data_steady/steady_<grid>_p<order>_{results,residual,cell_res}.bin`
- **Adjoint:** `data_steady/steady_<grid>_p<order>_adjoint_{psi_cycle<N>_dg,mesh_cycle<N>,indicators_cycle<N>}.bin`

## Key Source Files

| File | Purpose |
|------|---------|
| `src/main.cpp` | CLI entry point, auto-chaining, adaptation loop |
| `src/Solver.cpp` | DG solver, time stepping, Cl integration |
| `src/Adjoint.cpp` | Adjoint solve, sensitivity, error indicators |
| `src/MeshRefinement.cpp` | Element marking, longest-edge bisection, solution interpolation |
| `src/Fluxes.cpp` | Roe and HLLE numerical fluxes |

## Tests

```bash
g++ -O2 -std=c++17 test_results/test_jacobian.cpp src/State.cpp src/Mesh.cpp \
    src/Fluxes.cpp src/Solver.cpp src/Adjoint.cpp src/MeshRefinement.cpp \
    -o test_jacobian.exe -lstdc++fs
./test_jacobian.exe grids/base.gri
```