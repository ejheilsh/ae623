# MINIMAL WORKING EXAMPLE - Start Here!

This is a SIMPLIFIED version to get you started with p=0 and p=1 ONLY.
Once this works, extend to p=2,3.

## STEP 1: Modify Solver.hpp

Add these lines to the class definition:

```cpp
class FiniteVolumeSolver {
public:
  Mesh mesh;
  
  // ===== ADD THESE NEW MEMBERS =====
  int p_order = 0;                              // polynomial order
  int ndof_per_elem = 1;                        // (p+1)*(p+2)/2
  std::vector<std::vector<Vec4>> U_dg;          // [nelem][ndof][4]
  std::vector<std::vector<Vec4>> U0_dg;
  std::vector<std::vector<double>> MassMatrixInv;  // [ndof][ndof]
  // ===== END NEW MEMBERS =====
  
  std::vector<Vec4> U;  // keep for backward compatibility
  std::vector<Vec4> U0;
  std::vector<double> res_history;
  std::vector<double> cell_residuals;
  std::string fluxname = "roe";
  // ... rest unchanged ...
  
private:
  // ===== ADD THESE NEW METHODS =====
  void initializeDG();
  std::vector<double> evaluateBasis(double xi, double eta);
  void computeMassMatrix();
  // ===== END NEW METHODS =====
  
  // ... rest unchanged ...
};
```

## STEP 2: Add initialization in Solver.cpp

At the end of the constructor, add:

```cpp
FiniteVolumeSolver::FiniteVolumeSolver(const std::string &meshfile) {
  if (!mesh.readGRI(meshfile)) {
    throw std::runtime_error("Failed to read mesh file");
  }
  p0 = (rho0 * a0 * a0) / gamma;
  pout = 0.7 * p0;
  alpha = alpha * M_PI / 180.0;
  
  // ===== ADD THIS =====
  initializeDG();
  // ===== END ADD =====
  
  setInitialCondition();
}

// ===== ADD THIS NEW FUNCTION =====
void FiniteVolumeSolver::initializeDG() {
  // p_order should be set BEFORE calling constructor
  // For now, you can hardcode or read from a config
  // p_order = 0;  // set this before calling constructor
  
  ndof_per_elem = (p_order + 1) * (p_order + 2) / 2;
  std::cout << "DG initialization: p=" << p_order 
            << ", ndof=" << ndof_per_elem << std::endl;
  
  computeMassMatrix();
}
```

## STEP 3: Implement basis functions (minimal p=0,1)

```cpp
std::vector<double> 
FiniteVolumeSolver::evaluateBasis(double xi, double eta) {
  std::vector<double> phi(ndof_per_elem);
  
  if (p_order == 0) {
    phi[0] = 1.0;
  } 
  else if (p_order == 1) {
    phi[0] = 1.0 - xi - eta;
    phi[1] = xi;
    phi[2] = eta;
  }
  else {
    std::cerr << "p > 1 not implemented yet!" << std::endl;
    exit(1);
  }
  
  return phi;
}
```

## STEP 4: Compute mass matrix (simplified)

```cpp
void FiniteVolumeSolver::computeMassMatrix() {
  int ndof = ndof_per_elem;
  MassMatrixInv.resize(ndof, std::vector<double>(ndof, 0.0));
  
  if (p_order == 0) {
    // For p=0, mass matrix M = ∫ 1*1 dA = area
    // So M^{-1} = 1.0 (we'll divide by area separately)
    MassMatrixInv[0][0] = 1.0;
  }
  else if (p_order == 1) {
    // For p=1 on reference triangle with area 0.5:
    // M_ij = ∫∫ φᵢ φⱼ dA
    // This is known analytically:
    // M = (1/24) * [[2, 1, 1],
    //               [1, 2, 1],
    //               [1, 1, 2]]
    // M^{-1} = 6 * [[3, -1, -1],
    //               [-1, 3, -1],
    //               [-1, -1, 3]]
    double factor = 6.0;  // inverse of mass matrix on ref triangle
    MassMatrixInv[0][0] =  3.0 * factor;
    MassMatrixInv[0][1] = -1.0 * factor;
    MassMatrixInv[0][2] = -1.0 * factor;
    MassMatrixInv[1][0] = -1.0 * factor;
    MassMatrixInv[1][1] =  3.0 * factor;
    MassMatrixInv[1][2] = -1.0 * factor;
    MassMatrixInv[2][0] = -1.0 * factor;
    MassMatrixInv[2][1] = -1.0 * factor;
    MassMatrixInv[2][2] =  3.0 * factor;
  }
  else {
    std::cerr << "Mass matrix for p > 1 not implemented!" << std::endl;
    exit(1);
  }
}
```

## STEP 5: Modify setInitialCondition

```cpp
void FiniteVolumeSolver::setInitialCondition() {
  int Ne = mesh.E.size();
  
  // ===== ADD DG INITIALIZATION =====
  U_dg.resize(Ne);
  for (int e = 0; e < Ne; ++e) {
    U_dg[e].resize(ndof_per_elem);
  }
  // ===== END ADD =====
  
  double rhou = rho0 * Minf * a0 * std::cos(alpha);
  double rhov = rho0 * Minf * a0 * std::sin(alpha);
  double rhoE = rho0 * (a0 * a0 / ((gamma - 1.0) * gamma) + 0.5 * Minf * Minf);

  for (int e = 0; e < Ne; ++e) {
    // ===== ADD DG IC =====
    for (int j = 0; j < ndof_per_elem; ++j) {
      U_dg[e][j] = {rho0, rhou, rhov, rhoE};
    }
    // ===== END ADD =====
    
    // Keep old U for compatibility
    U.push_back(U_dg[e][0]);  // p=0 case: first DOF is the cell average
  }
  U0_dg = U_dg;
  U0 = U;
}
```

## STEP 6: Create a SIMPLE DG residual (p=0 first!)

For p=0, you should get EXACTLY the same result as your FV code.
This is a good test!

```cpp
// Add this as a NEW method (don't replace calcResidual yet)
FiniteVolumeSolver::ResidualResult
FiniteVolumeSolver::calcResidualDG_Simple(const std::vector<std::vector<Vec4>> &Un_dg,
                                           double time, bool use_unsteady_wake) {
  int Ne = mesh.E.size();
  ResidualResult res;
  
  // Create residual storage: [nelem][ndof]
  // We'll flatten it for simplicity: R[e*ndof + j]
  res.R.resize(Ne * ndof_per_elem);
  res.sdl.assign(Ne, 0.0);
  
  for (int e = 0; e < Ne; ++e) {
    for (int j = 0; j < ndof_per_elem; ++j) {
      res.R[e * ndof_per_elem + j] = {0, 0, 0, 0};
    }
  }
  
  // ===== INTERIOR EDGES =====
  for (int i = 0; i < (int)mesh.IE.size(); ++i) {
    int eL = mesh.IE[i].elemL;
    int eR = mesh.IE[i].elemR;
    Vec2 normal = mesh.inormals[i];
    double len = mesh.ilengths[i];
    
    // For p=0: only 1 quadrature point (midpoint is fine)
    // For p=1: need 2 quadrature points
    int nq = (p_order == 0) ? 1 : 2;
    
    // Simple approach for p=0: use element average (DOF 0)
    if (p_order == 0) {
      Vec4 uL = Un_dg[eL][0];
      Vec4 uR = Un_dg[eR][0];
      
      FluxResult fr;
      if (fluxname == "hlle")
        fr = fluxHLLE(uL, uR, normal, gamma);
      else
        fr = fluxRoe(uL, uR, normal, gamma);
      
      res.R[eL * ndof_per_elem + 0] += fr.F * len;
      res.R[eR * ndof_per_elem + 0] -= fr.F * len;
      res.sdl[eL] += fr.smax * len;
      res.sdl[eR] += fr.smax * len;
    }
    else if (p_order == 1) {
      // For p=1: use 2-point Gauss quadrature on edge
      // Gauss points on [-1,1]: ξ = ±1/√3
      double gauss_pts[2] = {-0.5773502691896257, 0.5773502691896257};
      double gauss_wts[2] = {1.0, 1.0};
      
      // Determine which edge this is for element eL
      // (You need to map edge to local edge index 0,1,2)
      // For simplicity, assume edge 0 here (you'll need proper mapping)
      
      for (int q = 0; q < 2; ++q) {
        // Map Gauss point to reference coordinates
        // This depends on which edge (0,1,2) of the triangle
        // For edge 0 (v0->v1): ξ varies, η=0
        // For edge 1 (v1->v2): ξ+η=1
        // For edge 2 (v2->v0): ξ=0, η varies
        
        // SIMPLIFIED: Use midpoint for now (you'll refine this)
        double xi = 0.5, eta = 0.0;  // edge 0 midpoint
        
        std::vector<double> phiL = evaluateBasis(xi, eta);
        std::vector<double> phiR = evaluateBasis(xi, eta);  // need proper mapping
        
        // Reconstruct solution at quad point
        Vec4 uL = {0,0,0,0};
        Vec4 uR = {0,0,0,0};
        for (int j = 0; j < ndof_per_elem; ++j) {
          uL += Un_dg[eL][j] * phiL[j];
          uR += Un_dg[eR][j] * phiR[j];
        }
        
        FluxResult fr;
        if (fluxname == "hlle")
          fr = fluxHLLE(uL, uR, normal, gamma);
        else
          fr = fluxRoe(uL, uR, normal, gamma);
        
        // Integrate to residual
        for (int j = 0; j < ndof_per_elem; ++j) {
          double weight = gauss_wts[q] * 0.5;  // 0.5 for mapping [-1,1] to edge
          res.R[eL * ndof_per_elem + j] += fr.F * (weight * phiL[j] * len);
          res.R[eR * ndof_per_elem + j] -= fr.F * (weight * phiR[j] * len);
        }
        
        res.sdl[eL] += fr.smax * len;
        res.sdl[eR] += fr.smax * len;
      }
    }
  }
  
  // ===== BOUNDARY EDGES ===== (similar structure)
  for (int i = 0; i < (int)mesh.BE.size(); ++i) {
    int eL = mesh.BE[i].elemL;
    int bIdx = mesh.BE[i].bIndex;
    std::string bName = mesh.Bname[bIdx];
    Vec2 n = mesh.bnormals[i];
    double len = mesh.blengths[i];
    
    if (p_order == 0) {
      Vec4 u_int = Un_dg[eL][0];
      
      FluxResult fr;
      if (bName == "inflow") {
        Vec2 edge_midpoint = (mesh.V[mesh.BE[i].v[0]] + mesh.V[mesh.BE[i].v[1]]) * 0.5;
        Vec4 Ub = subsonicInflow(u_int, n, rho0, a0, alpha, gamma, 
                                 edge_midpoint.y, time, use_unsteady_wake);
        if (fluxname == "hlle")
          fr = fluxHLLE(u_int, Ub, n, gamma);
        else
          fr = fluxRoe(u_int, Ub, n, gamma);
      } else if (bName == "outflow") {
        Vec4 Ub = subsonicOutflow(u_int, n, pout, gamma);
        if (fluxname == "hlle")
          fr = fluxHLLE(u_int, Ub, n, gamma);
        else
          fr = fluxRoe(u_int, Ub, n, gamma);
      } else if (bName == "wall") {
        fr = inviscidWallFlux(u_int, n, gamma);
      }
      
      res.R[eL * ndof_per_elem + 0] += fr.F * len;
      res.sdl[eL] += fr.smax * len;
    }
    // ... p=1 similar to interior edges ...
  }
  
  return res;
}
```

## STEP 7: TEST p=0 first!

Modify `main.cpp` to set p_order:

```cpp
int main(int argc, char **argv) {
  // ... existing code ...
  
  int p_order = 0;  // ADD THIS
  if (argc >= 3) {
    p_order = std::stoi(argv[2]);
  }
  
  FiniteVolumeSolver solver(meshfile);
  solver.p_order = p_order;  // ADD THIS (must be BEFORE initializeDG!)
  solver.CFL = cfl;
  solver.fluxname = fluxname;
  
  // ... rest unchanged ...
}
```

## COMPILATION & TESTING

1. Compile your code
2. Run with `p_order = 0`:
   ```bash
   ./euler_solver grids/coarse.gri 0 0.5 hlle 1000 steady
   ```
3. Compare results to your old FV code - should be IDENTICAL!
4. Once p=0 works, implement p=1 properly using the guide

## KEY SIMPLIFICATIONS IN THIS EXAMPLE:

- ❌ Edge-to-reference-coordinate mapping not fully implemented
- ❌ Quadrature not complete for p=1
- ❌ Only works for straight edges (q=1 mesh)
- ✅ Data structure is correct
- ✅ p=0 should work perfectly
- ✅ Shows the overall flow

## NEXT STEPS AFTER THIS WORKS:

1. Implement proper edge quadrature mapping
2. Use quad1d.c for quadrature points/weights
3. Extend to p=2,3
4. Add mesh curving (q=2,3)
5. Implement visualization with sub-element refinement

The key insight: **Start with p=0 to validate your structure, then add complexity!**
