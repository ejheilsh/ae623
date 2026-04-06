#include "Adjoint.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

AdjointSolver::AdjointSolver(FiniteVolumeSolver &solver)
    : solver_(solver) {}

void AdjointSolver::solve() {
  int Ne = solver_.mesh.E.size();
  int ndof = solver_.ndof_per_elem;
  int N = Ne * ndof * 4;  // total scalar DOFs

  std::cerr << "Adjoint: assembling Jacobian (" << N << " x " << N << ")..." << std::endl;

  // 1. Get Jacobian blocks and output gradient
  auto blocks = solver_.calcJacobian();
  auto dJ = solver_.dCl_dU();

  // 2. Flatten RHS: rhs = -(dCl/dU)
  std::vector<double> rhs(N, 0.0);
  for (int e = 0; e < Ne; ++e)
    for (int j = 0; j < ndof; ++j)
      for (int k = 0; k < 4; ++k)
        rhs[e * ndof * 4 + j * 4 + k] = -dJ[e][j][k];

  // 3. Assemble (dR/dU)^T as a dense matrix
  //    For block (row_elem, col_elem, B):  (dR/dU) has B at (row, col)
  //    The TRANSPOSE puts B^T at (col, row)
  std::vector<double> A(static_cast<size_t>(N) * N, 0.0);
  for (auto &blk : blocks) {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        // Transpose: swap row/col element, swap i/j within block
        int ri = blk.col * ndof * 4 + i;
        int ci = blk.row * ndof * 4 + j;
        A[static_cast<size_t>(ri) * N + ci] += blk.B[j][i];
      }
  }

  // 4. Solve A * x = rhs via Gaussian elimination with partial pivoting
  //    (dense, O(N^3) — fine for N < ~10000)
  std::vector<double> x = rhs;  // will be overwritten with solution

  for (int col = 0; col < N; ++col) {
    // Pivot: find max in column below diagonal
    int piv = col;
    double maxval = std::abs(A[static_cast<size_t>(col) * N + col]);
    for (int row = col + 1; row < N; ++row) {
      double val = std::abs(A[static_cast<size_t>(row) * N + col]);
      if (val > maxval) { maxval = val; piv = row; }
    }
    if (maxval < 1e-14) {
      std::cerr << "Adjoint: singular matrix at column " << col << std::endl;
      break;
    }
    // Swap rows
    if (piv != col) {
      for (int j = 0; j < N; ++j)
        std::swap(A[static_cast<size_t>(col) * N + j],
                  A[static_cast<size_t>(piv) * N + j]);
      std::swap(x[col], x[piv]);
    }
    // Eliminate below
    double diag = A[static_cast<size_t>(col) * N + col];
    for (int row = col + 1; row < N; ++row) {
      double factor = A[static_cast<size_t>(row) * N + col] / diag;
      for (int j = col + 1; j < N; ++j)
        A[static_cast<size_t>(row) * N + j] -= factor * A[static_cast<size_t>(col) * N + j];
      x[row] -= factor * x[col];
    }
  }
  // Back-substitution
  for (int row = N - 1; row >= 0; --row) {
    for (int j = row + 1; j < N; ++j)
      x[row] -= A[static_cast<size_t>(row) * N + j] * x[j];
    x[row] /= A[static_cast<size_t>(row) * N + row];
  }

  // 5. Unflatten solution into psi_
  psi_.resize(Ne, std::vector<Vec4>(ndof, Vec4{0,0,0,0}));
  for (int e = 0; e < Ne; ++e)
    for (int j = 0; j < ndof; ++j)
      for (int k = 0; k < 4; ++k)
        psi_[e][j][k] = x[e * ndof * 4 + j * 4 + k];

  std::cerr << "Adjoint: solve complete." << std::endl;
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

  // PSEUDO-CODE:
  //
  // The error indicator for element e is:
  //   eps_e = |psi_h,e^T * R_h,e(U_H^h)|
  //
  // Steps:
  //
  // 1. Prolong the coarse (p=1) solution into the fine (p=2) space:
  //    auto U_p2 = solver_.prolongP1toP2(solver_.U_dg);
  //
  // 2. Set up the fine solver at p=2 on the same mesh:
  //    fine_solver.p_order = 2;
  //    fine_solver.initializeDG();
  //    fine_solver.U_dg = U_p2;
  //
  // 3. Evaluate the fine-space residual (do NOT time-step, just evaluate once):
  //    auto res = fine_solver.calcResidualDG(U_p2);
  //    // res.R is [nelem * ndof_p2] Vec4 values (the DG residual flattened)
  //
  // 4. Prolong the coarse adjoint into the fine space:
  //    auto psi_p2 = solver_.prolongP1toP2(psi_);
  //    // Same injection as the solution: first 3 modes copied, rest zero.
  //
  // 5. Compute element-wise inner product:
  //    for (int e = 0; e < Ne; ++e) {
  //      double dot = 0.0;
  //      for (int j = 0; j < ndof_p2; ++j)
  //        for (int k = 0; k < 4; ++k)
  //          dot += psi_p2[e][j][k] * res.R[e * ndof_p2 + j][k];
  //      eps[e] = std::abs(dot);
  //    }
  //
  // The sum of eps[e] gives the estimated global output error  |delta_Cl|.

  // 1. Prolong coarse solution and adjoint into fine (p+1) space
  auto U_fine = solver_.prolongP1toP2(solver_.U_dg);
  auto psi_fine = solver_.prolongP1toP2(psi_);

  // 2. Set up fine solver and evaluate residual (no time-stepping)
  fine_solver.p_order = solver_.p_order + 1;
  fine_solver.initializeDG();
  fine_solver.U_dg = U_fine;
  auto res = fine_solver.calcResidualDG(U_fine);

  // 3. Element-wise inner product: eps_e = |psi_h,e^T * R_h,e|
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
