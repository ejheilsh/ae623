// =====================================================================
// PROJECT 4 — Unit tests for residual Jacobian and adjoint sensitivity
// =====================================================================

#include "../src/Fluxes.hpp"
#include "../src/Solver.hpp"
#include "../src/Adjoint.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// ─── Test 1b: Flux Jacobian FD Ping (unit-level) ─────────────────────
bool test_flux_jacobian_fd_ping() {
  std::cout << "=== Test 1b: Flux Jacobian FD Ping ===" << std::endl;

  double gamma = 1.4;
  Vec4 UL = {1.0, 0.5, 0.1, 2.5};
  Vec4 UR = {0.8, 0.3, -0.05, 2.0};
  Vec2 n = {0.6, 0.8};

  FluxJacobianResult Janalytic = fluxRoeJacobian(UL, UR, n, gamma);

  // Since our fluxRoeJacobian IS FD-based, this test mainly confirms
  // self-consistency. It becomes meaningful once we switch to analytic.
  double eps = 1e-7;
  FluxResult F0 = fluxRoe(UL, UR, n, gamma);
  double max_err = 0.0;

  for (int j = 0; j < 4; ++j) {
    Vec4 ULp = UL; ULp[j] += eps;
    FluxResult Fp = fluxRoe(ULp, UR, n, gamma);
    for (int i = 0; i < 4; ++i) {
      double fd = (Fp.F[i] - F0.F[i]) / eps;
      double ref = std::max(std::abs(fd), 1e-12);
      double err = std::abs(Janalytic.dFdUL[i][j] - fd) / ref;
      max_err = std::max(max_err, err);
    }
  }
  for (int j = 0; j < 4; ++j) {
    Vec4 URp = UR; URp[j] += eps;
    FluxResult Fp = fluxRoe(UL, URp, n, gamma);
    for (int i = 0; i < 4; ++i) {
      double fd = (Fp.F[i] - F0.F[i]) / eps;
      double ref = std::max(std::abs(fd), 1e-12);
      double err = std::abs(Janalytic.dFdUR[i][j] - fd) / ref;
      max_err = std::max(max_err, err);
    }
  }

  bool pass = (max_err < 1e-4);
  std::cout << "  Max relative error: " << std::scientific << max_err
            << (pass ? "  PASS" : "  FAIL") << std::endl;
  return pass;
}

// ─── Test 1: Full Assembled Jacobian FD Ping ─────────────────────────
bool test_jacobian_fd_ping(const std::string &meshfile) {
  std::cout << "=== Test 1: Jacobian FD Ping ===" << std::endl;

  FiniteVolumeSolver solver(meshfile);
  solver.p_order = 0;
  solver.initializeDG();
  solver.CFL = 1.0;
  solver.fluxname = "roe";
  std::cout << "  Converging p=0 solution..." << std::endl;
  solver.solveSteady(50000);
  if (!solver.last_steady_converged) {
    std::cout << "  WARNING: solver did not converge, testing anyway" << std::endl;
  }

  // Get analytic Jacobian
  auto blocks = solver.calcJacobian();
  int Ne = solver.mesh.E.size();
  int N = 4 * Ne;

  // Assemble into dense matrix
  std::vector<std::vector<double>> J_an(N, std::vector<double>(N, 0.0));
  for (auto &blk : blocks)
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        J_an[blk.row * 4 + i][blk.col * 4 + j] += blk.B[i][j];

  // FD Jacobian column by column using DG residual (matches calcJacobian)
  double eps = 1e-7;
  auto R0 = solver.calcResidualDG(solver.U_dg);

  double max_rel_err = 0.0;
  int worst_row = 0, worst_col = 0;

  for (int col = 0; col < N; ++col) {
    int elem = col / 4;
    int comp = col % 4;
    auto U_pert = solver.U_dg;
    U_pert[elem][0][comp] += eps;  // p=0: single DOF per element
    auto R_pert = solver.calcResidualDG(U_pert);

    for (int row = 0; row < N; ++row) {
      int re = row / 4, rc = row % 4;
      double fd = (R_pert.R[re][rc] - R0.R[re][rc]) / eps;
      double ref = std::max({std::abs(fd), std::abs(J_an[row][col]), 1e-8});
      double err = std::abs(J_an[row][col] - fd) / ref;
      if (err > max_rel_err) {
        max_rel_err = err;
        worst_row = row;
        worst_col = col;
      }
    }
  }

  bool pass = (max_rel_err < 1e-3);  // 1e-3 tolerance for FD-based Jacobians
  std::cout << "  Max relative error: " << std::scientific << max_rel_err
            << " at (" << worst_row << "," << worst_col << ")"
            << (pass ? "  PASS" : "  FAIL") << std::endl;
  return pass;
}

// ─── Test 2: Adjoint Sensitivity Validation ──────────────────────────
bool test_adjoint_sensitivity(const std::string &meshfile) {
  std::cout << "=== Test 2: Adjoint Sensitivity ===" << std::endl;

  // 1. Converge baseline solution
  FiniteVolumeSolver solver(meshfile);
  solver.p_order = 0;
  solver.initializeDG();
  solver.CFL = 1.0;
  solver.fluxname = "roe";
  std::cout << "  Converging baseline p=0 solution..." << std::endl;
  solver.solveSteady(50000);
  // Second pass: warm-start converges much tighter (1e-5 of new baseline ≈ 1e-10 absolute)
  solver.solveSteady(50000);

  // 2. Solve adjoint
  std::cout << "  Solving adjoint..." << std::endl;
  AdjointSolver adj(solver);
  adj.solve();

  // 3. Adjoint-based sensitivity
  double dalpha = 1e-6;  // radians
  double sens_adj = adj.sensitivityAlpha(dalpha);
  std::cout << "  Adjoint sensitivity: " << std::scientific << sens_adj << std::endl;

  // 4. FD sensitivity: re-converge at perturbed alpha
  std::cout << "  Computing FD sensitivity (re-converging at perturbed alpha)..." << std::endl;
  double alpha_base = solver.alpha;

  // Compute Cl at baseline (use dCl_dU to get Cl — or compute directly)
  // Actually we need to compute Cl. Let's use the forces output.
  // Simpler: use dCl_dU dot U to get partial, or just compute Cl from pressure.
  // Most robust: compute Cl from the state directly.
  auto computeCl = [&](FiniteVolumeSolver &s) {
    double gamma = s.gamma;
    double Fy = 0.0;
    for (int i = 0; i < (int)s.mesh.BE.size(); ++i) {
      int bIdx = s.mesh.BE[i].bIndex;
      if (s.mesh.Bname[bIdx] != "wall") continue;
      int eL = s.mesh.BE[i].elemL;
      Vec4 Ui = s.cellAverage(s.U_dg[eL]);
      State st(Ui, gamma);
      double p = st.p();

      // For curved elements, use quadrature with curved normals
      if (s.mesh.E[eL].q_order > 1) {
        int va = s.mesh.BE[i].v[0], vb = s.mesh.BE[i].v[1];
        auto bqr = s.getQuadratureRule(s.mesh.E[eL].q_order);
        for (int q = 0; q < bqr.n; ++q) {
          auto eg = s.mesh.evaluateEdgeGeometry(eL, va, vb, bqr.points[q]);
          Fy += p * eg.normal.y * bqr.weights[q] * eg.ds_dt;
        }
      } else {
        Vec2 n = s.mesh.bnormals[i];
        double len = s.mesh.blengths[i];
        Fy += p * n.y * len;
      }
    }
    double qinf = s.Minf * s.a0;
    double norm = 0.5 * s.rho0 * qinf * qinf * 1.0;
    return Fy / norm;
  };

  double Cl_base = computeCl(solver);

  // Perturbed
  FiniteVolumeSolver solver2(meshfile);
  solver2.p_order = 0;
  solver2.alpha = alpha_base + dalpha;
  solver2.initializeDG();
  // Warm start from base solution
  solver2.U_dg = solver.U_dg;
  solver2.U = solver.U;
  std::cout << "  Re-converging at alpha + dalpha..." << std::endl;
  solver2.solveSteady(50000);

  double Cl_pert = computeCl(solver2);
  double sens_fd = (Cl_pert - Cl_base) / dalpha;

  std::cout << "  FD sensitivity:      " << std::scientific << sens_fd << std::endl;
  std::cout << "  Cl_base=" << Cl_base << "  Cl_pert=" << Cl_pert << std::endl;

  double rel_err = std::abs(sens_adj - sens_fd) / std::max(std::abs(sens_fd), 1e-12);
  bool pass = (rel_err < 0.05);  // 5% tolerance for adjoint vs FD
  std::cout << "  Relative error: " << rel_err
            << (pass ? "  PASS" : "  FAIL") << std::endl;
  return pass;
}

int main(int argc, char **argv) {
  std::string meshfile = "grids/coarse.gri";
  if (argc >= 2) meshfile = argv[1];

  int nfail = 0;

  if (!test_flux_jacobian_fd_ping())    nfail++;
  if (!test_jacobian_fd_ping(meshfile)) nfail++;
  if (!test_adjoint_sensitivity(meshfile)) nfail++;

  std::cout << "\n" << (3 - nfail) << "/3 tests passed." << std::endl;
  return nfail;
}
