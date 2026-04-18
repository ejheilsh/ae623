#include "Adjoint.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

AdjointSolver::AdjointSolver(FiniteVolumeSolver &solver)
    : solver_(solver) {}

void AdjointSolver::solve() {
  int Ne = solver_.mesh.E.size();
  int ndof = solver_.ndof_per_elem;
  int N = Ne * ndof * 4;  // total scalar DOFs

  // 1. Get output gradient dJ/dU
  auto dJ = solver_.dCl_dU();

  // 2. Flatten RHS: rhs = -(dCl/dU)
  std::vector<double> rhs(N, 0.0);
  for (int e = 0; e < Ne; ++e)
    for (int j = 0; j < ndof; ++j)
      for (int k = 0; k < 4; ++k)
        rhs[e * ndof * 4 + j * 4 + k] = -dJ[e][j][k];

  // 3. Assemble (dR/dU)^T in sparse CSR format via finite differences.
  //    Row c of the stored matrix is the gradient of R[c] w.r.t. all DOFs,
  //    which is dR/dU column c — exactly the transpose Jacobian row needed.
  //    For p=0 DG, only ~15 neighbours are non-zero per row (0.01% fill at N~10k).
  std::cerr << "Adjoint: assembling sparse Jacobian (" << N << " DOFs) via FD...\n";
  std::vector<int>    row_ptr(N + 1, 0);
  std::vector<int>    col_idx;
  std::vector<double> nz_val;
  col_idx.reserve(N * 20);
  nz_val.reserve(N * 20);
  {
    const double eps = 1e-7, drop_tol = 1e-12;
    auto R0 = solver_.calcResidualDG(solver_.U_dg);
    int stride = std::max(1, N / 10);
    for (int c = 0; c < N; ++c) {
      if (c % stride == 0)
        std::cerr << "  FD: " << (100 * c / N) << "% (" << c << "/" << N << ")\n";
      int elem = c / (ndof * 4), mode = (c / 4) % ndof, comp = c % 4;
      auto U_pert = solver_.U_dg;
      U_pert[elem][mode][comp] += eps;
      auto Rp = solver_.calcResidualDG(U_pert);
      row_ptr[c] = static_cast<int>(col_idx.size());
      for (int e2 = 0; e2 < Ne; ++e2)
        for (int j2 = 0; j2 < ndof; ++j2)
          for (int k2 = 0; k2 < 4; ++k2) {
            double v = (Rp.R[e2 * ndof + j2][k2] - R0.R[e2 * ndof + j2][k2]) / eps;
            if (std::abs(v) > drop_tol) {
              col_idx.push_back(e2 * ndof * 4 + j2 * 4 + k2);
              nz_val.push_back(v);
            }
          }
    }
    row_ptr[N] = static_cast<int>(col_idx.size());
  }
  {
    int nnz = static_cast<int>(nz_val.size());
    std::cerr << "  Sparse: " << nnz << " nnz, avg "
              << std::fixed << std::setprecision(2) << static_cast<double>(nnz) / N
              << "/row, " << 100.0 * nnz / (static_cast<double>(N) * N) << "% fill\n"
              << std::defaultfloat;
  }

  // Sparse CSR matvec: w = A * v  (A = (dR/dU)^T stored row-by-row)
  auto matvec = [&](const std::vector<double>& v) -> std::vector<double> {
    std::vector<double> w(N, 0.0);
    for (int r = 0; r < N; ++r)
      for (int k = row_ptr[r]; k < row_ptr[r + 1]; ++k)
        w[r] += nz_val[k] * v[col_idx[k]];
    return w;
  };

  auto vdot = [&](const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (int i = 0; i < N; ++i) s += a[i] * b[i];
    return s;
  };

  // 4. GMRES(m) with restarts — Saad & Schultz (1986).
  //    Minimises ||b - A x||_2 over successive Krylov subspaces K_m(A, r0).
  //    Unlike BiCGSTAB, the residual is monotonically non-increasing and
  //    there is no breakdown.  Memory: (m+1)*N doubles per restart.
  const int    m            = 150;   // Krylov dimension per restart
  const double gmres_tol    = 1e-10; // relative residual tolerance
  const int    max_restarts = 30;

  std::cerr << "  GMRES(" << m << ")  tol=" << gmres_tol
            << "  max_restarts=" << max_restarts << "\n";

  std::vector<double> x(N, 0.0);
  double b_norm = std::sqrt(vdot(rhs, rhs));

  if (b_norm < 1e-30) {
    std::cerr << "  RHS is zero — psi = 0.\n";
  } else {
    // Arnoldi basis V[0..m], upper Hessenberg H (column-major), Givens (cs,sn), g = Q^T e_1 beta
    std::vector<std::vector<double>> V(m + 1, std::vector<double>(N, 0.0));
    std::vector<double> H((m + 1) * m, 0.0);  // H[i + (m+1)*j]
    std::vector<double> cs(m), sn(m), g(m + 1);

    bool converged = false;
    for (int restart = 0; restart < max_restarts && !converged; ++restart) {
      // Compute initial residual r = b - A x and normalise into V[0]
      {
        auto Ax = matvec(x);
        for (int i = 0; i < N; ++i) V[0][i] = rhs[i] - Ax[i];
      }
      double beta = std::sqrt(vdot(V[0], V[0]));
      if (beta / b_norm < gmres_tol) { converged = true; break; }
      for (int i = 0; i < N; ++i) V[0][i] /= beta;

      std::fill(H.begin(),  H.end(),  0.0);
      std::fill(cs.begin(), cs.end(), 0.0);
      std::fill(sn.begin(), sn.end(), 0.0);
      std::fill(g.begin(),  g.end(),  0.0);
      g[0] = beta;

      int j_done = m;  // number of Arnoldi steps actually taken
      for (int j = 0; j < m; ++j) {
        // w = A * V[j]  (one sparse matvec per Arnoldi step)
        std::vector<double> w = matvec(V[j]);

        // Modified Gram-Schmidt orthogonalisation against V[0..j]
        for (int i = 0; i <= j; ++i) {
          double hij = vdot(w, V[i]);
          H[i + (m + 1) * j] = hij;
          for (int k = 0; k < N; ++k) w[k] -= hij * V[i][k];
        }
        double hnext = std::sqrt(vdot(w, w));
        H[(j + 1) + (m + 1) * j] = hnext;

        // Apply all previous Givens rotations to the new Hessenberg column
        for (int i = 0; i < j; ++i) {
          double h1 = H[i     + (m + 1) * j];
          double h2 = H[i + 1 + (m + 1) * j];
          H[i     + (m + 1) * j] =  cs[i] * h1 + sn[i] * h2;
          H[i + 1 + (m + 1) * j] = -sn[i] * h1 + cs[i] * h2;
        }

        // Compute and apply new Givens rotation to zero H[j+1, j]
        double hjj  = H[j     + (m + 1) * j];
        double hj1j = H[j + 1 + (m + 1) * j];
        double denom = std::hypot(hjj, hj1j);
        if (denom < 1e-300) { j_done = j + 1; break; }  // happy breakdown
        cs[j] = hjj  / denom;
        sn[j] = hj1j / denom;
        H[j     + (m + 1) * j] = denom;  // cs*hjj + sn*hj1j
        H[j + 1 + (m + 1) * j] = 0.0;

        // Update the rotated RHS; |g[j+1]| is the current residual norm
        double gj = g[j];
        g[j]     =  cs[j] * gj;
        g[j + 1] = -sn[j] * gj;

        double relres = std::abs(g[j + 1]) / b_norm;
        if (j % 50 == 0)
          std::cerr << "  GMRES restart " << restart
                    << " iter " << std::setw(4) << j
                    << "  relres=" << relres << "\n";

        if (relres < gmres_tol || hnext < 1e-300) { j_done = j + 1; break; }

        // Extend basis: V[j+1] = w / hnext
        for (int i = 0; i < N; ++i) V[j + 1][i] = w[i] / hnext;
      }

      // Back-substitute: solve upper-triangular H[0:j_done, 0:j_done] * y = g[0:j_done]
      std::vector<double> y(j_done, 0.0);
      for (int i = j_done - 1; i >= 0; --i) {
        y[i] = g[i];
        for (int k = i + 1; k < j_done; ++k)
          y[i] -= H[i + (m + 1) * k] * y[k];
        y[i] /= H[i + (m + 1) * i];
      }

      // Update solution: x += V[:,0:j_done] * y
      for (int j = 0; j < j_done; ++j)
        for (int i = 0; i < N; ++i)
          x[i] += y[j] * V[j][i];

      double relres_final = std::abs(g[j_done]) / b_norm;
      if (relres_final < gmres_tol) {
        converged = true;
        std::cerr << "  GMRES: converged at restart " << restart
                  << "  (relres=" << relres_final << ").\n";
      } else {
        std::cerr << "  GMRES: restarting (relres=" << relres_final << ")\n";
      }
    }
    if (!converged)
      std::cerr << "  GMRES: max restarts reached — using best iterate.\n";
  }

  // 5. Unflatten solution into psi_
  psi_.resize(Ne, std::vector<Vec4>(ndof, Vec4{0, 0, 0, 0}));
  for (int e = 0; e < Ne; ++e)
    for (int j = 0; j < ndof; ++j)
      for (int k = 0; k < 4; ++k)
        psi_[e][j][k] = x[e * ndof * 4 + j * 4 + k];

  std::cerr << "Adjoint: solve complete.\n";
}

double AdjointSolver::sensitivityAlpha(double dalpha) {
  // PSEUDO-CODE:
  //
  // This validates the adjoint by computing dCl/dalpha two independent ways.
  //
  // METHOD 1 (adjoint-based):
  // Compute dR/dalpha by FD: perturb alpha, evaluate residual, difference
  int Ne = solver_.mesh.E.size();
  int ndof = solver_.ndof_per_elem;

  // Baseline residual at current alpha
  auto R0 = solver_.calcResidualDG(solver_.U_dg);

  // Perturb alpha
  double alpha_save = solver_.alpha;
  solver_.alpha = alpha_save + dalpha;
  auto Rp = solver_.calcResidualDG(solver_.U_dg);
  solver_.alpha = alpha_save;  // restore

  // Adjoint-based sensitivity: dCl/dalpha = psi^T * (dR/dalpha)
  // (psi satisfies (dR/dU)^T psi = -(dJ/dU), and dJ/dalpha_direct = 0)
  double sens = 0.0;
  double dRda_norm = 0.0, psi_norm = 0.0;
  for (int e = 0; e < Ne; ++e)
    for (int j = 0; j < ndof; ++j)
      for (int k = 0; k < 4; ++k) {
        double dRda = (Rp.R[e * ndof + j][k] - R0.R[e * ndof + j][k]) / dalpha;
        sens += psi_[e][j][k] * dRda;
        dRda_norm += dRda * dRda;
        psi_norm += psi_[e][j][k] * psi_[e][j][k];
      }
  std::cerr << "  |dR/dalpha| = " << std::sqrt(dRda_norm)
            << "  |psi| = " << std::sqrt(psi_norm) << std::endl;

  // Also verify adjoint solution: check ||A^T psi + dJ/dU|| should be small
  {
    auto blocks = solver_.calcJacobian();
    auto dJ = solver_.dCl_dU();
    int N = Ne * ndof * 4;
    // Flatten psi
    std::vector<double> psi_flat(N, 0.0);
    for (int e = 0; e < Ne; ++e)
      for (int jj = 0; jj < ndof; ++jj)
        for (int k = 0; k < 4; ++k)
          psi_flat[e*ndof*4 + jj*4 + k] = psi_[e][jj][k];

    // Compute A^T * psi  (A^T[ri,ci] = B[j][i] for block(row,col))
    std::vector<double> Atpsi(N, 0.0);
    for (auto &blk : blocks) {
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
          int ri = blk.col * ndof * 4 + i;   // transpose row
          int ci = blk.row * ndof * 4 + j;   // transpose col
          Atpsi[ri] += blk.B[j][i] * psi_flat[ci];
        }
    }
    // Compute residual: A^T psi + dJ/dU (should be zero)
    double res_norm = 0.0, dJ_norm = 0.0;
    for (int e = 0; e < Ne; ++e)
      for (int jj = 0; jj < ndof; ++jj)
        for (int k = 0; k < 4; ++k) {
          double r = Atpsi[e*ndof*4 + jj*4 + k] + dJ[e][jj][k];
          res_norm += r * r;
          dJ_norm += dJ[e][jj][k] * dJ[e][jj][k];
        }
    std::cerr << "  Adjoint residual |A^T psi + dJ/dU| = " << std::sqrt(res_norm)
              << "  |dJ/dU| = " << std::sqrt(dJ_norm) << std::endl;
  }

  return sens;
}

std::vector<double>
AdjointSolver::errorIndicators(FiniteVolumeSolver &fine_solver) {
  int Ne = solver_.mesh.E.size();
  std::vector<double> eps(Ne, 0.0);

  // DWR error estimate: eps_e = |psi_h,e^T * R_h,e(U_H^h)|
  //
  // Steps:
  //   1. Prolong coarse solution U_H into fine (p+1) space: U_H^h
  //   2. Set up fine solver at p+1, assign U_H^h
  //   3. Evaluate fine-space residual R_h(U_H^h)
  //   4. Solve the fine-space adjoint: (dR_h/dU)^T psi_h = -(dJ_h/dU)
  //   5. Element-wise inner product: eps_e = |psi_h,e^T * R_h,e|

  // 1. Prolong coarse solution into fine (p+1) space
  auto U_fine = solver_.prolongP1toP2(solver_.U_dg);

  // 2. Set up fine solver at p+1 with the prolonged coarse solution
  fine_solver.p_order = solver_.p_order + 1;
  fine_solver.initializeDG();
  fine_solver.U_dg = U_fine;

  // 3. Evaluate fine-space residual (single evaluation, no time-stepping)
  auto res = fine_solver.calcResidualDG(U_fine);

  // 4. Solve the fine-space adjoint on fine_solver
  //    This gives psi_h with nonzero higher-order modes that properly
  //    weight the fine-space residual (critical when coarse order is p=0).
  std::cerr << "  Solving fine-space (p=" << fine_solver.p_order
            << ") adjoint for error indicators..." << std::endl;
  AdjointSolver fine_adj(fine_solver);
  fine_adj.solve();
  const auto &psi_fine = fine_adj.psi();

  // 5. Element-wise inner product: eps_e = |psi_h,e^T * R_h,e|
  int ndof_fine = fine_solver.ndof_per_elem;
  for (int e = 0; e < Ne; ++e) {
    double dot = 0.0;
    for (int j = 0; j < ndof_fine; ++j)
      for (int k = 0; k < 4; ++k)
        dot += psi_fine[e][j][k] * res.R[e * ndof_fine + j][k];
    eps[e] = std::abs(dot);
  }

  double total = 0.0;
  for (double v : eps) total += v;
  std::cerr << "  Error indicators: sum |eps_e| = " << total
            << " (estimated |delta_Cl|)" << std::endl;

  return eps;
}
