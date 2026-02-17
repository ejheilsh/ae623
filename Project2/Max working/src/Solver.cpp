#include "Solver.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>

FiniteVolumeSolver::FiniteVolumeSolver(const std::string &meshfile) {
  if (!mesh.readGRI(meshfile)) {
    throw std::runtime_error("Failed to read mesh file");
  }
  p0 = (rho0 * a0 * a0) / gamma;
  pout = 0.7 * p0;
  alpha = alpha * M_PI / 180.0;
  setInitialCondition();
}

void FiniteVolumeSolver::setInitialCondition() {
  int Ne = mesh.E.size();
  U.resize(Ne);
  double rhou = rho0 * Minf * a0 * std::cos(alpha);
  double rhov = rho0 * Minf * a0 * std::sin(alpha);
  double rhoE = rho0 * (a0 * a0 / ((gamma - 1.0) * gamma) + 0.5 * Minf * Minf);

  for (int i = 0; i < Ne; ++i) {
    U[i] = {rho0, rhou, rhov, rhoE};
  }
  U0 = U;
}

void FiniteVolumeSolver::loadInitialCondition(const std::string &filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) {
    std::cerr << "Warning: Could not open initial condition file " << filename 
              << ", using default IC instead" << std::endl;
    setInitialCondition();
    return;
  }
  
  int Ne_file;
  in.read((char*)&Ne_file, sizeof(int));
  
  int Ne = mesh.E.size();
  if (Ne_file != Ne) {
    std::cerr << "Warning: Initial condition file has " << Ne_file 
              << " elements but mesh has " << Ne 
              << " elements. Using default IC instead." << std::endl;
    in.close();
    setInitialCondition();
    return;
  }
  
  U.resize(Ne);
  for (int i = 0; i < Ne; ++i) {
    in.read((char*)U[i].v, sizeof(double) * 4);
  }
  in.close();
  U0 = U;
  
  std::cout << "Successfully loaded initial condition from " << filename << std::endl;
}

FiniteVolumeSolver::ResidualResult
FiniteVolumeSolver::calcResidual(const std::vector<Vec4> &Un, double time,
                                 bool use_unsteady_wake) {
  int Ne = mesh.E.size();
  ResidualResult res;
  res.R.assign(Ne, {0, 0, 0, 0});
  res.sdl.assign(Ne, 0.0);

  for (int i = 0; i < (int)mesh.IE.size(); ++i) {
    int eL = mesh.IE[i].elemL;
    int eR = mesh.IE[i].elemR;
    FluxResult fr;
    if (fluxname == "hlle")
      fr = fluxHLLE(Un[eL], Un[eR], mesh.inormals[i], gamma);
    else
      fr = fluxRoe(Un[eL], Un[eR], mesh.inormals[i], gamma);

    double len = mesh.ilengths[i];
    res.R[eL] += fr.F * len;
    res.R[eR] -= fr.F * len;
    res.sdl[eL] += fr.smax * len;
    res.sdl[eR] += fr.smax * len;
  }

  for (int i = 0; i < (int)mesh.BE.size(); ++i) {
    int eL = mesh.BE[i].elemL;
    int bIdx = mesh.BE[i].bIndex;
    std::string bName = mesh.Bname[bIdx];
    Vec2 n = mesh.bnormals[i];
    double len = mesh.blengths[i];

    FluxResult fr;
    if (bName == "inflow") {
      // Calculate y-position at boundary edge (midpoint)
      Vec2 edge_midpoint = (mesh.V[mesh.BE[i].v[0]] + mesh.V[mesh.BE[i].v[1]]) * 0.5;
      double y_pos = edge_midpoint.y;
      
      Vec4 Ub = subsonicInflow(Un[eL], n, rho0, a0, alpha, gamma, y_pos, time, use_unsteady_wake);
      if (fluxname == "hlle")
        fr = fluxHLLE(Un[eL], Ub, n, gamma);
      else
        fr = fluxRoe(Un[eL], Ub, n, gamma);
    } else if (bName == "outflow") {
      Vec4 Ub = subsonicOutflow(Un[eL], n, pout, gamma);
      if (fluxname == "hlle")
        fr = fluxHLLE(Un[eL], Ub, n, gamma);
      else
        fr = fluxRoe(Un[eL], Ub, n, gamma);
    } else if (bName == "wall") {
      fr = inviscidWallFlux(Un[eL], n, gamma);
    }

    res.R[eL] += fr.F * len;
    res.sdl[eL] += fr.smax * len;
  }
  return res;
}

void FiniteVolumeSolver::solveSteady(int itercap, bool secondOrder,
                                     bool limited) {
  int Ne = mesh.E.size();
  std::cout << "Beginning solver loop for " << itercap << " iterations..."
            << std::endl;
  for (int niter = 0; niter < itercap; ++niter) {
    // Calculate residual just for the norm and sdl (for dt)
    ResidualResult res =
        secondOrder ? calcResidualSecondOrder(U, limited, 0.0, false) : calcResidual(U, 0.0, false);

    double Rnorm = 0;
    for (const auto &r : res.R) {
      Rnorm +=
          std::abs(r[0]) + std::abs(r[1]) + std::abs(r[2]) + std::abs(r[3]);
    }
    res_history.push_back(Rnorm);

    if (niter % 10 == 0 || Rnorm < rtol) {
      double minRho = 1e10, minP = 1e10;
      cell_residuals.resize(Ne);
      for (int i = 0; i < Ne; ++i) {
        const auto &u = U[i];
        const auto &r = res.R[i];
        cell_residuals[i] =
            std::abs(r[0]) + std::abs(r[1]) + std::abs(r[2]) + std::abs(r[3]);

        State s(u, gamma);
        minRho = std::min(minRho, s.rho());
        minP = std::min(minP, s.p());
      }
      std::cout << "Iter: " << std::setw(6) << niter
                << " | Residual: " << std::scientific << std::setprecision(6)
                << Rnorm << " | Min Rho: " << minRho << " | Min P: " << minP
                << std::endl;
    }

    if (Rnorm < rtol) {
      std::cout << "Converged in " << niter << " iterations." << std::endl;
      break;
    }

    // Use SSP-RK2 for time integration
    U = sspRK2(U, secondOrder, limited, 0.0, false);

    if (!isPhysical(U)) {
      std::cerr << "Non-physical state detected at iteration " << niter
                << std::endl;
      break;
    }
  }
}

std::vector<Vec4> FiniteVolumeSolver::sspRK2(const std::vector<Vec4> &Un,
                                             bool secondOrder, bool limited, double time,
                                             bool use_unsteady_wake) {
  int Ne = Un.size();
  ResidualResult res1 =
      secondOrder ? calcResidualSecondOrder(Un, limited, time, use_unsteady_wake) : calcResidual(Un, time, use_unsteady_wake);
  std::vector<Vec4> U1(Ne);
  double cfl_eff = CFL;
  if (secondOrder && limited)
    // cfl_eff *= 0.2;
    cfl_eff *= 1;

  for (int i = 0; i < Ne; ++i) {
    double sdl = std::max(res1.sdl[i], 1e-12);
    double dt = cfl_eff * 2.0 * mesh.areas[i] / sdl;
    U1[i] = Un[i] - (res1.R[i] / mesh.areas[i]) * dt;
  }

  ResidualResult res2 =
      secondOrder ? calcResidualSecondOrder(U1, limited, time, use_unsteady_wake) : calcResidual(U1, time, use_unsteady_wake);
  std::vector<Vec4> Unp1(Ne);
  for (int i = 0; i < Ne; ++i) {
    double sdl = std::max(res1.sdl[i], 1e-12);
    double dt = cfl_eff * 2.0 * mesh.areas[i] / sdl;
    Unp1[i] = Un[i] * 0.5 + (U1[i] - (res2.R[i] / mesh.areas[i]) * dt) * 0.5;
  }
  return Unp1;
}

std::vector<Vec4> FiniteVolumeSolver::applyLimiter(const std::vector<Vec4> &Un,
                                                   std::vector<Vec4> &gradX,
                                                   std::vector<Vec4> &gradY) {
  int Ne = mesh.E.size();
  std::vector<std::vector<int>> neighbors(Ne);
  for (const auto &ie : mesh.IE) {
    neighbors[ie.elemL].push_back(ie.elemR);
    neighbors[ie.elemR].push_back(ie.elemL);
  }
  // (Boundary neighbors are technically included if we want, but let's stick to
  // Python logic)

  for (int i = 0; i < Ne; ++i) {
    Vec4 Umin = Un[i], Umax = Un[i];
    for (int nbor : neighbors[i]) {
      for (int k = 0; k < 4; ++k) {
        Umin[k] = std::min(Umin[k], Un[nbor][k]);
        Umax[k] = std::max(Umax[k], Un[nbor][k]);
      }
    }

    Vec4 phi = {1, 1, 1, 1};
    for (int v = 0; v < 3; ++v) {
      Vec2 r = mesh.V[mesh.E[i].v[v]] - mesh.centroids[i];
      Vec4 dU = gradX[i] * r.x + gradY[i] * r.y;
      for (int k = 0; k < 4; ++k) {
        if (dU[k] > 1e-12)
          phi[k] = std::min(phi[k], (Umax[k] - Un[i][k]) / dU[k]);
        else if (dU[k] < -1e-12)
          phi[k] = std::min(phi[k], (Umin[k] - Un[i][k]) / dU[k]);
      }
    }
    for (int k = 0; k < 4; ++k) {
      phi[k] = std::max(0.0, std::min(1.0, phi[k]));
      gradX[i][k] *= phi[k];
      gradY[i][k] *= phi[k];
    }
  }
  return Un; // Gradient is modified in-place
}

bool FiniteVolumeSolver::isPhysical(const std::vector<Vec4> &Un) {
  for (const auto &u : Un) {
    State s(u, gamma);
    if (s.rho() <= 0 || s.p() <= 0)
      return false;
  }
  return true;
}

FiniteVolumeSolver::ResidualResult
FiniteVolumeSolver::calcResidualSecondOrder(const std::vector<Vec4> &Un,
                                            bool limited, double time,
                                            bool use_unsteady_wake) {
  // Basic implementation for now, mirroring the calcResidual structure but with
  // gradients This part requires more complex logic for Green-Gauss gradients
  // and limiting. Given the task's scope, I'll provide a simplified version or
  // the first order for now if time is tight, but the user expects second
  // order. Let's implement the gradient logic.
  int Ne = mesh.E.size();
  std::vector<Vec4> gradX(Ne, {0, 0, 0, 0}), gradY(Ne, {0, 0, 0, 0});

  // Green-Gauss Gradient
  for (int i = 0; i < (int)mesh.IE.size(); ++i) {
    int eL = mesh.IE[i].elemL;
    int eR = mesh.IE[i].elemR;
    Vec4 u_hat = (Un[eL] + Un[eR]) * 0.5;
    Vec2 n = mesh.inormals[i];
    double len = mesh.ilengths[i];
    gradX[eL] += u_hat * (n.x * len);
    gradY[eL] += u_hat * (n.y * len);
    gradX[eR] -= u_hat * (n.x * len);
    gradY[eR] -= u_hat * (n.y * len);
  }
  for (int i = 0; i < (int)mesh.BE.size(); ++i) {
    int eL = mesh.BE[i].elemL;
    Vec4 Ub;
    std::string bName = mesh.Bname[mesh.BE[i].bIndex];
    if (bName == "inflow") {
      Vec2 edge_midpoint = (mesh.V[mesh.BE[i].v[0]] + mesh.V[mesh.BE[i].v[1]]) * 0.5;
      double y_pos = edge_midpoint.y;
      Ub = subsonicInflow(Un[eL], mesh.bnormals[i], rho0, a0, alpha, gamma, y_pos, time);
    } else if (bName == "outflow")
      Ub = subsonicOutflow(Un[eL], mesh.bnormals[i], pout, gamma);
    else if (bName == "wall")
      Ub = inviscidWallState(Un[eL], mesh.bnormals[i], gamma);

    Vec4 u_hat = (Un[eL] + Ub) * 0.5;
    Vec2 n = mesh.bnormals[i];
    double len = mesh.blengths[i];
    gradX[eL] += u_hat * (n.x * len);
    gradY[eL] += u_hat * (n.y * len);
  }

  for (int i = 0; i < Ne; ++i) {
    gradX[i] = gradX[i] / mesh.areas[i];
    gradY[i] = gradY[i] / mesh.areas[i];
  }

  if (limited) {
    applyLimiter(Un, gradX, gradY);
  }

  ResidualResult res;
  res.R.assign(Ne, {0, 0, 0, 0});
  res.sdl.assign(Ne, 0.0);

  for (int i = 0; i < (int)mesh.IE.size(); ++i) {
    int eL = mesh.IE[i].elemL;
    int eR = mesh.IE[i].elemR;
    Vec2 xf = (mesh.V[mesh.IE[i].v[0]] + mesh.V[mesh.IE[i].v[1]]) * 0.5;

    Vec2 dL = xf - mesh.centroids[eL];
    Vec2 dR = xf - mesh.centroids[eR];

    // Periodic shift correction (standard 18.0 as per user mesh fixes)
    if (std::abs(dL.y) > 9.0) {
      if (dL.y > 0)
        dL.y -= 18.0;
      else
        dL.y += 18.0;
    }
    if (std::abs(dR.y) > 9.0) {
      if (dR.y > 0)
        dR.y -= 18.0;
      else
        dR.y += 18.0;
    }

    Vec4 ULf = Un[eL] + gradX[eL] * dL.x + gradY[eL] * dL.y;
    Vec4 URf = Un[eR] + gradX[eR] * dR.x + gradY[eR] * dR.y;

    FluxResult fr;
    if (fluxname == "hlle")
      fr = fluxHLLE(ULf, URf, mesh.inormals[i], gamma);
    else
      fr = fluxRoe(ULf, URf, mesh.inormals[i], gamma);

    double len = mesh.ilengths[i];
    res.R[eL] += fr.F * len;
    res.R[eR] -= fr.F * len;
    res.sdl[eL] += fr.smax * len;
    res.sdl[eR] += fr.smax * len;
  }

  for (int i = 0; i < (int)mesh.BE.size(); ++i) {
    int eL = mesh.BE[i].elemL;
    Vec2 xf = (mesh.V[mesh.BE[i].v[0]] + mesh.V[mesh.BE[i].v[1]]) * 0.5;
    Vec2 dL = xf - mesh.centroids[eL];
    Vec4 ULf = Un[eL] + gradX[eL] * dL.x + gradY[eL] * dL.y;

    std::string bName = mesh.Bname[mesh.BE[i].bIndex];
    Vec2 n = mesh.bnormals[i];
    double len = mesh.blengths[i];
    FluxResult fr;
    if (bName == "inflow") {
      double y_pos = xf.y;
      Vec4 Ub = subsonicInflow(ULf, n, rho0, a0, alpha, gamma, y_pos, time, use_unsteady_wake);
      if (fluxname == "hlle")
        fr = fluxHLLE(ULf, Ub, n, gamma);
      else
        fr = fluxRoe(ULf, Ub, n, gamma);
    } else if (bName == "outflow") {
      Vec4 Ub = subsonicOutflow(ULf, n, pout, gamma);
      if (fluxname == "hlle")
        fr = fluxHLLE(ULf, Ub, n, gamma);
      else
        fr = fluxRoe(ULf, Ub, n, gamma);
    } else if (bName == "wall") {
      fr = inviscidWallFlux(ULf, n, gamma);
    }
    res.R[eL] += fr.F * len;
    res.sdl[eL] += fr.smax * len;
  }
  return res;
}

void FiniteVolumeSolver::solveUnsteady(int itercap, bool secondOrder,
                                       bool limited) {
  int Ne = mesh.E.size();
  std::cout << "Beginning unsteady solver loop for " << itercap << " iterations..."
            << std::endl;
  
  // For unsteady, use a global time step (smallest over all cells)
  current_time = 0.0;
  std::vector<Vec4> U_prev = U;  // Store previous solution for monitoring change
  
  // Snapshot saving parameters
  int snapshot_interval = 100;  // Save every N iterations
  int snapshot_count = 0;
  
  // Create data directory if it doesn't exist
  system("mkdir -p data");
  
  for (int niter = 0; niter < itercap; ++niter) {
    // Calculate residual (spatial flux divergence = time derivative)
    ResidualResult res =
        secondOrder ? calcResidualSecondOrder(U, limited, current_time, true) : calcResidual(U, current_time, true);

    // Calculate global time step (smallest over all cells for time-accurate unsteady)
    double dt_global = 1e10;
    for (int i = 0; i < Ne; ++i) {
      double sdl = std::max(res.sdl[i], 1e-12);
      double dt_local = CFL * 2.0 * mesh.areas[i] / sdl;
      dt_global = std::min(dt_global, dt_local);
    }
    
    // Explicit Euler update with global time step
    for (int i = 0; i < Ne; ++i) {
      U[i] = U[i] - (res.R[i] / mesh.areas[i]) * dt_global;
    }
    
    // Advance physical time
    current_time += dt_global;
    
    // Calculate residual norm (time derivative magnitude)
    double Rnorm = 0;
    for (const auto &r : res.R) {
      Rnorm += std::abs(r[0]) + std::abs(r[1]) + std::abs(r[2]) + std::abs(r[3]);
    }
    res_history.push_back(Rnorm);
    
    // Calculate solution change from previous time step
    double dU_norm = 0;
    for (int i = 0; i < Ne; ++i) {
      Vec4 dU = U[i] - U_prev[i];
      dU_norm += std::abs(dU[0]) + std::abs(dU[1]) + std::abs(dU[2]) + std::abs(dU[3]);
    }

    // Save periodic snapshots
    if (niter % snapshot_interval == 0) {
      char filename[256];
      snprintf(filename, sizeof(filename), "data/results_%.6f_%04d.bin", 
               current_time, snapshot_count);
      saveSnapshot(filename);
      std::cout << "Saved snapshot: " << filename << std::endl;
      snapshot_count++;
    }

    if (niter % 1 == 0) {
      double minRho = 1e10, minP = 1e10;
      cell_residuals.resize(Ne);
      for (int i = 0; i < Ne; ++i) {
        const auto &u = U[i];
        const auto &r = res.R[i];
        cell_residuals[i] =
            std::abs(r[0]) + std::abs(r[1]) + std::abs(r[2]) + std::abs(r[3]);

        State s(u, gamma);
        minRho = std::min(minRho, s.rho());
        minP = std::min(minP, s.p());
      }
      std::cout << "Iter: " << std::setw(6) << niter
                << " | ||dU/dt||: " << std::scientific << std::setprecision(6)
                << Rnorm << " | ||ΔU||: " << dU_norm 
                << " | dt: " << dt_global
                << " | time: " << current_time
                << " | Min Rho: " << minRho << " | Min P: " << minP
                << std::endl;
    }

    // Check for convergence (solution stopped changing)
    if (dU_norm < rtol) {
      std::cout << "Reached steady state (solution stopped changing) at iteration " 
                << niter << std::endl;
      break;
    }
    
    // Check for NaN/Inf
    if (!isPhysical(U)) {
      std::cerr << "Non-physical state detected at iteration " << niter
                << std::endl;
      break;
    }
    
    // Store current solution for next iteration comparison
    U_prev = U;
  }
}

void FiniteVolumeSolver::saveSnapshot(const std::string &filename) {
  std::ofstream out(filename, std::ios::binary);
  int Ne = U.size();
  out.write((char *)&Ne, sizeof(int));
  for (const auto &u : U) {
    out.write((char *)u.v, sizeof(double) * 4);
  }
  out.close();
}
