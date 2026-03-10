# DG Implementation Guide - WHERE TO INSERT CODE

This guide shows EXACTLY where to modify your existing FV code to support DG (p=0,1,2,3).

## STRATEGY: Incremental Approach

**Phase 1**: Keep your mesh unchanged (q=1, linear). Implement DG for p=0,1 first.
**Phase 2**: Later, add mesh curving (q=2,3) for p>0.

---

## FILE 1: Solver.hpp - DATA STRUCTURE CHANGES

### LOCATION 1A: Add polynomial order and DOF tracking

```cpp
class FiniteVolumeSolver {
public:
  Mesh mesh;
  // OLD (FV): std::vector<Vec4> U;  // [nelem][4] - one state per element
  
  // NEW (DG): Multi-DOF representation
  int p_order = 0;                              // ← INSERT: polynomial order (0,1,2,3)
  int ndof_per_elem;                            // ← INSERT: (p+1)*(p+2)/2
  std::vector<std::vector<Vec4>> U_dg;          // ← INSERT: [nelem][ndof][4]
  std::vector<std::vector<Vec4>> U0_dg;         // ← INSERT: for baseline
  
  // Keep old U for compatibility (p=0 case can use it)
  std::vector<Vec4> U;     
  std::vector<Vec4> U0;
```

### LOCATION 1B: Add DG-specific helper data

```cpp
private:
  // ← INSERT: Mass matrix storage (pre-computed per element or global)
  std::vector<std::vector<double>> MassMatrixInv;  // [ndof][ndof] - inverted mass matrix
  
  // ← INSERT: Basis function helper (you'll implement this)
  void computeMassMatrix();                         // compute M^{-1} once
  std::vector<double> evaluateBasis(double xi, double eta, int p);  // φ_i(ξ,η)
  std::vector<double> evaluateBasisGrad(double xi, double eta, int p); // ∂φ_i/∂ξ, ∂φ_i/∂η
```

### LOCATION 1C: Change residual signature

```cpp
  // OLD:
  // ResidualResult calcResidual(const std::vector<Vec4> &Un, ...);
  
  // NEW: 0Need to handle multi-DOF
  ResidualResult calcResidualDG(const std::vector<std::vector<Vec4>> &Un_dg, 
                                 double time = 0.0, 
                                 bool use_unsteady_wake = false);
```

---

## FILE 2: Solver.cpp - INITIALIZATION CHANGES

### LOCATION 2A: Constructor - set DOF count

In `FiniteVolumeSolver::FiniteVolumeSolver(const std::string &meshfile)`:

```cpp
FiniteVolumeSolver::FiniteVolumeSolver(const std::string &meshfile) {
  if (!mesh.readGRI(meshfile)) {
    throw std::runtime_error("Failed to read mesh file");
  }
  
  // ← INSERT: Initialize DG parameters
  // p_order should be set BEFORE calling this constructor (or pass as parameter)
  // Assume p_order is set externally or add parameter to constructor
  ndof_per_elem = (p_order + 1) * (p_order + 2) / 2;
  std::cout << "DG solver with p=" << p_order << ", ndof=" << ndof_per_elem << std::endl;
  
  p0 = (rho0 * a0 * a0) / gamma;
  pout = 0.7 * p0;
  alpha = alpha * M_PI / 180.0;
  
  // ← INSERT: Compute mass matrix inverse once
  computeMassMatrix();
  
  setInitialCondition();
}
```

### LOCATION 2B: setInitialCondition - initialize multi-DOF

```cpp
void FiniteVolumeSolver::setInitialCondition() {
  int Ne = mesh.E.size();
  
  // ← INSERT: Allocate DG data structure
  U_dg.resize(Ne);
  for (int e = 0; e < Ne; ++e) {
    U_dg[e].resize(ndof_per_elem);
  }
  
  double rhou = rho0 * Minf * a0 * std::cos(alpha);
  double rhov = rho0 * Minf * a0 * std::sin(alpha);
  double rhoE = rho0 * (a0 * a0 / ((gamma - 1.0) * gamma) + 0.5 * Minf * Minf);

  for (int e = 0; e < Ne; ++e) {
    for (int j = 0; j < ndof_per_elem; ++j) {
      // ← PSEUDOCODE: For now, set all DOFs to the same value
      // Later (p>0): you might want to project the initial condition onto basis
      U_dg[e][j] = {rho0, rhou, rhov, rhoE};
    }
  }
  U0_dg = U_dg;
  
  // Also set old U for p=0 compatibility
  if (p_order == 0) {
    U.resize(Ne);
    for (int e = 0; e < Ne; ++e) {
      U[e] = U_dg[e][0];  // p=0: only 1 DOF
    }
    U0 = U;
  }
}
```

---

## FILE 3: Solver.cpp - CORE DG RESIDUAL CALCULATION

### LOCATION 3: Replace calcResidual with DG version

**THIS IS THE BIG ONE - where the DG magic happens**

```cpp
FiniteVolumeSolver::ResidualResult
FiniteVolumeSolver::calcResidualDG(const std::vector<std::vector<Vec4>> &Un_dg,
                                    double time, bool use_unsteady_wake) {
  int Ne = mesh.E.size();
  ResidualResult res;
  
  // ← INSERT: Residual now has [nelem][ndof] structure
  res.R.resize(Ne * ndof_per_elem);  // flatten for now, or use 2D vector
  res.sdl.assign(Ne, 0.0);
  
  // Initialize residual to zero
  for (int e = 0; e < Ne; ++e) {
    for (int j = 0; j < ndof_per_elem; ++j) {
      res.R[e * ndof_per_elem + j] = {0, 0, 0, 0};
    }
  }
  
  // ═══════════════════════════════════════════════════════════════
  // INTERIOR EDGES
  // ═══════════════════════════════════════════════════════════════
  for (int i = 0; i < (int)mesh.IE.size(); ++i) {
    int eL = mesh.IE[i].elemL;
    int eR = mesh.IE[i].elemR;
    Vec2 normal = mesh.inormals[i];
    double len = mesh.ilengths[i];
    
    // ← PSEUDOCODE FOR DG:
    // 1. Get quadrature points for this edge (from quad1d.c)
    int nq = getQuadratureOrder(p_order);  // e.g., p+1 or p+2 points
    double *xq = ...; // quadrature points on edge
    double *wq = ...; // quadrature weights
    
    // 2. Loop over quadrature points
    for (int q = 0; q < nq; ++q) {
      // 2a. Map quadrature point to reference coordinates
      double xi_q = ..., eta_q = ...;  // depends on which edge
      
      // 2b. Evaluate basis functions at this quadrature point
      std::vector<double> phi = evaluateBasis(xi_q, eta_q, p_order);
      
      // 2c. Reconstruct solution at quadrature point
      Vec4 u_L = {0,0,0,0};
      Vec4 u_R = {0,0,0,0};
      for (int j = 0; j < ndof_per_elem; ++j) {
        u_L += Un_dg[eL][j] * phi[j];  // u = Σ u_j φ_j
        u_R += Un_dg[eR][j] * phi[j];
      }
      
      // 2d. Compute Riemann flux at this point
      FluxResult fr;
      if (fluxname == "hlle")
        fr = fluxHLLE(u_L, u_R, normal, gamma);
      else
        fr = fluxRoe(u_L, u_R, normal, gamma);
      
      // 2e. Integrate: add to residual for each DOF
      for (int j = 0; j < ndof_per_elem; ++j) {
        // R_j += weight * F* * φ_j * edge_length
        res.R[eL * ndof_per_elem + j] += fr.F * (wq[q] * phi[j] * len);
        res.R[eR * ndof_per_elem + j] -= fr.F * (wq[q] * phi[j] * len);
      }
      
      // Spectral radius for time step (same as FV)
      res.sdl[eL] += fr.smax * len;
      res.sdl[eR] += fr.smax * len;
    } // end quadrature loop
  } // end interior edge loop
  
  // ═══════════════════════════════════════════════════════════════
  // BOUNDARY EDGES - same concept
  // ═══════════════════════════════════════════════════════════════
  for (int i = 0; i < (int)mesh.BE.size(); ++i) {
    int eL = mesh.BE[i].elemL;
    int bIdx = mesh.BE[i].bIndex;
    std::string bName = mesh.Bname[bIdx];
    Vec2 n = mesh.bnormals[i];
    double len = mesh.blengths[i];
    
    // ← PSEUDOCODE: Similar quadrature loop
    int nq = getQuadratureOrder(p_order);
    for (int q = 0; q < nq; ++q) {
      double xi_q = ..., eta_q = ...;
      std::vector<double> phi = evaluateBasis(xi_q, eta_q, p_order);
      
      // Reconstruct interior state
      Vec4 u_int = {0,0,0,0};
      for (int j = 0; j < ndof_per_elem; ++j) {
        u_int += Un_dg[eL][j] * phi[j];
      }
      
      // Apply boundary condition to get exterior state
      Vec4 u_ext;
      if (bName == "inflow") {
        Vec2 edge_pos = ...; // position of quadrature point
        u_ext = subsonicInflow(u_int, n, rho0, a0, alpha, gamma, 
                               edge_pos.y, time, use_unsteady_wake);
      } else if (bName == "outflow") {
        u_ext = subsonicOutflow(u_int, n, pout, gamma);
      } else if (bName == "wall") {
        // For slip wall, flux is just pressure term
        // (can still use your inviscidWallFlux function)
      }
      
      // Compute flux
      FluxResult fr;
      if (bName == "wall") {
        fr = inviscidWallFlux(u_int, n, gamma);
      } else {
        if (fluxname == "hlle")
          fr = fluxHLLE(u_int, u_ext, n, gamma);
        else
          fr = fluxRoe(u_int, u_ext, n, gamma);
      }
      
      // Integrate
      for (int j = 0; j < ndof_per_elem; ++j) {
        res.R[eL * ndof_per_elem + j] += fr.F * (wq[q] * phi[j] * len);
      }
      res.sdl[eL] += fr.smax * len;
    }
  }
  
  return res;
}
```

---

## FILE 4: Solver.cpp - TIME STEPPING (RK method)

### LOCATION 4: Modify sspRK2 to apply mass matrix inverse

```cpp
std::vector<std::vector<Vec4>>
FiniteVolumeSolver::sspRK2_DG(const std::vector<std::vector<Vec4>> &Un_dg,
                               bool secondOrder, bool limited, 
                               double time, bool use_unsteady_wake) {
  int Ne = mesh.E.size();
  
  // Compute residual
  ResidualResult res1 = calcResidualDG(Un_dg, time, use_unsteady_wake);
  
  // ← PSEUDOCODE: Apply M^{-1} to residual and compute time step
  std::vector<double> dt = calcDt(res1.sdl);  // same as FV
  
  // First RK stage: U1 = Un - dt * M^{-1} * R
  std::vector<std::vector<Vec4>> U1(Ne);
  for (int e = 0; e < Ne; ++e) {
    U1[e].resize(ndof_per_elem);
    
    for (int i = 0; i < ndof_per_elem; ++i) {
      U1[e][i] = Un_dg[e][i];  // start with Un
      
      // ← KEY STEP: Apply M^{-1} to residual
      for (int j = 0; j < ndof_per_elem; ++j) {
        U1[e][i] -= res1.R[e * ndof_per_elem + j] * 
                    (dt[e] / mesh.areas[e]) * MassMatrixInv[i][j];
      }
    }
  }
  
  // Second RK stage (similar)
  //ResidualResult res2 = calcResidualDG(U1, time, use_unsteady_wake);
  // ... apply M^{-1} again ...
  
  return U1;  // or (Un + U1)/2 for SSP-RK2
}
```

---

## FILE 5: NEW FILE - BasisFunctions.cpp

### Create helper functions for basis evaluation

```cpp
// ← INSERT: New file to handle shape functions
#include "Solver.hpp"
#include <cmath>

std::vector<double> 
FiniteVolumeSolver::evaluateBasis(double xi, double eta, int p) {
  int ndof = (p+1)*(p+2)/2;
  std::vector<double> phi(ndof);
  
  // ← COPY from shape.c (translate C to C++)
  switch (p) {
    case 0:
      phi[0] = 1.0;
      break;
    case 1:
      phi[0] = 1 - xi - eta;
      phi[1] = xi;
      phi[2] = eta;
      break;
    case 2:
      // ... copy from shape.c ...
      break;
    case 3:
      // ... copy from shape.c ...
      break;
  }
  return phi;
}

void FiniteVolumeSolver::computeMassMatrix() {
  int ndof = ndof_per_elem;
  MassMatrixInv.resize(ndof, std::vector<double>(ndof, 0.0));
  
  // ← PSEUDOCODE: Integrate M_ij = ∫∫ φ_i φ_j dA using 2D quadrature
  // Use quad2d.c to get quadrature points on reference triangle
  
  // For simplified version (p=0,1), you can compute analytically:
  if (p_order == 0) {
    MassMatrixInv[0][0] = 1.0;  // Mass matrix = element area (will divide by area)
  } else if (p_order == 1) {
    // For p=1 on reference triangle, M is known analytically
    // M_inv = ... (can look up or compute numerically)
  }
  
  // For p>1: Use numerical quadrature (quad2d.c)
}
```

---

## MAIN.CPP MODIFICATIONS

### LOCATION 6: Pass polynomial order from command line

```cpp
int main(int argc, char **argv) {
  // Existing args: meshfile, order (this was secondOrder bool)
  // → REPURPOSE "order" to be p_order (0,1,2,3)
  
  int p_order = 0;  // ← INSERT: polynomial order
  if (argc >= 3) {
    p_order = std::stoi(argv[2]);  // was: secondOrder = (argv[2] == "2")
  }
  
  // ← INSERT: Create DG solver with polynomial order
  FiniteVolumeSolver solver(meshfile);
  solver.p_order = p_order;  // or pass to constructor
  solver.CFL = cfl;
  solver.fluxname = fluxname;
  
  // ... rest of code ...
}
```

---

## SUMMARY OF MODIFICATIONS

**For p=0 (your current FV code):**
- ndof_per_elem = 1
- U_dg[e][0] = your current U[e]
- No change in behavior!

**For p=1:**
- ndof_per_elem = 3 (3 DOFs per triangle)
- Need to implement basis functions (linear on reference triangle)
- Need mass matrix (3x3, can compute analytically)
- Quadrature: 2-3 points per edge

**For p=2:**
- ndof_per_elem = 6
- Quadratic basis functions
- Mass matrix (6x6)
- Quadrature: 3-4 points per edge

**For p=3:**
- ndof_per_elem = 10
- Cubic basis functions
- Mass matrix (10x10)
- Quadrature: 4-5 points per edge

**MESH:** Can stay q=1 (linear) for initial testing. Curve later.

---

## WHAT TO CODE FIRST (order of implementation):

1. ✅ Modify Solver.hpp: add p_order, ndof_per_elem, U_dg
2. ✅ Create evaluateBasis() for p=0,1 (copy from shape.c)
3. ✅ Modify setInitialCondition() to initialize U_dg
4. ✅ Implement calcResidualDG() for p=0 only (should match your FV)
5. ✅ Test p=0 - should give identical results to your current code
6. ✅ Implement calcResidualDG() for p=1 with proper quadrature
7. ✅ Implement computeMassMatrix() for p=1
8. ✅ Test p=1 - run freestream preservation test
9. ✅ Extend to p=2,3
10. ✅ Add mesh curving (separate task)
