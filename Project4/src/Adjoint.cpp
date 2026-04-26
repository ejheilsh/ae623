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
  std::cerr << "Adjoint: assembling sparse Jacobian (" << N << " DOFs) via graph-colored FD...\n";
  std::vector<int>    row_ptr(N + 1, 0);
  std::vector<int>    col_idx;
  std::vector<double> nz_val;
  col_idx.reserve(N * 20);
  nz_val.reserve(N * 20);

  {
    const double eps_rel = 1e-7, eps_abs = 1e-10, drop_tol = 1e-12;
    auto R0 = solver_.calcResidualDG(solver_.U_dg);

    // Build distance-1 adjacency (elements sharing internal faces)
    std::vector<std::vector<int>> adj(Ne);
    for (const auto& edge : solver_.mesh.IE) {
      adj[edge.elemL].push_back(edge.elemR);
      adj[edge.elemR].push_back(edge.elemL);
    }

    // Build distance-2 adjacency
    std::vector<std::vector<int>> adj2(Ne);
    for (int e = 0; e < Ne; ++e) {
      std::set<int> nbrs2;
      for (int n1 : adj[e]) {
        nbrs2.insert(n1);
        for (int n2 : adj[n1]) {
          nbrs2.insert(n2);
        }
      }
      nbrs2.erase(e);
      adj2[e].assign(nbrs2.begin(), nbrs2.end());
    }

    // Greedy Coloring
    std::vector<int> colors(Ne, -1);
    int num_colors = 0;
    for (int e = 0; e < Ne; ++e) {
      std::set<int> used;
      for (int nbr : adj2[e]) {
        if (colors[nbr] != -1) used.insert(colors[nbr]);
      }
      int c = 0;
      while (used.count(c)) c++;
      colors[e] = c;
      num_colors = std::max(num_colors, c + 1);
    }

    std::vector<std::vector<int>> color_groups(num_colors);
    for (int e = 0; e < Ne; ++e) color_groups[colors[e]].push_back(e);
    std::cerr << "  Graph coloring: " << num_colors << " colors used.\n";

    // Since we don't compute columns sequentially, accumulate into specific rows
    // rows[r] will hold a list of pairs (col, value) for A^T
    std::vector<std::vector<std::pair<int, double>>> csr_rows(N);

    // Evaluate colors
    int total_evals = num_colors * ndof * 4;
    int eval = 0;
    int stride = std::max(1, total_evals / 10);

    for (int c_idx = 0; c_idx < num_colors; ++c_idx) {
      const auto& elems = color_groups[c_idx];
      for (int mode = 0; mode < ndof; ++mode) {
        for (int comp = 0; comp < 4; ++comp) {
          if (eval % stride == 0)
            std::cerr << "  Colored FD: " << (100 * eval / total_evals) << "% (" << eval << "/" << total_evals << ")\n";
          eval++;

          auto U_pert = solver_.U_dg;
          std::vector<double> eps_list(Ne, 0.0);
          
          for (int elem : elems) {
            double u0 = solver_.U_dg[elem][mode][comp];
            double eps = eps_rel * std::max(1.0, std::abs(u0)) + eps_abs;
            eps_list[elem] = eps;
            U_pert[elem][mode][comp] += eps;
          }

          auto Rp = solver_.calcResidualDG(U_pert);

          // Extract isolated footprints
          for (int elem : elems) {
            double eps = eps_list[elem];
            int node_c = elem * ndof * 4 + mode * 4 + comp;
            
            // Perturbations manifest on elem and its dist-1 neighbors ONLY
            std::vector<int> footprint = adj[elem];
            footprint.push_back(elem);
            
            for (int e2 : footprint) {
              for (int j2 = 0; j2 < ndof; ++j2) {
                for (int k2 = 0; k2 < 4; ++k2) {
                  double v = (Rp.R[e2 * ndof + j2][k2] - R0.R[e2 * ndof + j2][k2]) / eps;
                  if (std::abs(v) > drop_tol) {
                    csr_rows[node_c].push_back({e2 * ndof * 4 + j2 * 4 + k2, v});
                  }
                }
              }
            }
          }
        }
      }
    }
    
    // Flatten rows into CSR
    for (int c = 0; c < N; ++c) {
      row_ptr[c] = static_cast<int>(col_idx.size());
      for (const auto& pair : csr_rows[c]) {
        col_idx.push_back(pair.first);
        nz_val.push_back(pair.second);
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

  // Block-Jacobi preconditioner based on the diagonal blocks of A.
  int n_blk = ndof * 4;
  std::vector<std::vector<std::vector<double>>> inv_blocks(Ne, 
      std::vector<std::vector<double>>(n_blk, std::vector<double>(n_blk, 0.0)));
  int n_good_blocks = 0;

  for (int e = 0; e < Ne; ++e) {
    // Extract dense diagonal block
    std::vector<std::vector<double>> B(n_blk, std::vector<double>(n_blk, 0.0));
    for (int r = 0; r < n_blk; ++r) {
      int row = e * n_blk + r;
      for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
        int col = col_idx[k];
        if (col >= e * n_blk && col < (e + 1) * n_blk) {
          B[r][col - e * n_blk] = nz_val[k];
        }
      }
    }

    // Invert block via Gauss-Jordan elimination
    std::vector<std::vector<double>> invB(n_blk, std::vector<double>(n_blk, 0.0));
    for (int i = 0; i < n_blk; ++i) invB[i][i] = 1.0;
    bool singular = false;
    for (int i = 0; i < n_blk; ++i) {
      int pivot = i;
      for (int j = i + 1; j < n_blk; ++j) {
        if (std::abs(B[j][i]) > std::abs(B[pivot][i])) pivot = j;
      }
      if (std::abs(B[pivot][i]) < 1e-13) {
        singular = true;
        break;
      }
      std::swap(B[i], B[pivot]);
      std::swap(invB[i], invB[pivot]);
      double div = B[i][i];
      for (int j = 0; j < n_blk; ++j) {
        B[i][j] /= div;
        invB[i][j] /= div;
      }
      for (int j = 0; j < n_blk; ++j) {
        if (i != j) {
          double mult = B[j][i];
          for (int k = 0; k < n_blk; ++k) {
            B[j][k] -= mult * B[i][k];
            invB[j][k] -= mult * invB[i][k];
          }
        }
      }
    }
    
    if (!singular) {
      inv_blocks[e] = invB;
      n_good_blocks++;
    } else {
      for (int i = 0; i < n_blk; ++i) inv_blocks[e][i][i] = 1.0;
    }
  }
  std::cerr << "  Block-Jacobi preconditioner: " << n_good_blocks << "/" << Ne
            << " usable element blocks\n";

  auto applyLeftPreconditioner = [&](const std::vector<double> &v) {
    std::vector<double> w(N, 0.0);
    for (int e = 0; e < Ne; ++e) {
      for (int r = 0; r < n_blk; ++r) {
        double sum = 0.0;
        for (int c = 0; c < n_blk; ++c) {
          sum += inv_blocks[e][r][c] * v[e * n_blk + c];
        }
        w[e * n_blk + r] = sum;
      }
    }
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
  const int    m            = 100;   // Krylov dimension per restart
  const double gmres_tol    = 1e-5; // relative residual tolerance
  const int    max_restarts = 100;

  std::cerr << "  GMRES(" << m << ")  tol=" << gmres_tol
            << "  max_restarts=" << max_restarts << "\n";

  std::vector<double> x(N, 0.0);
  auto rhs_pc = applyLeftPreconditioner(rhs);
  double b_norm = std::sqrt(vdot(rhs_pc, rhs_pc));

  if (b_norm < 1e-30) {
    std::cerr << "  RHS is zero — psi = 0.\n";
  } else {
    // Arnoldi basis V[0..m], upper Hessenberg H (column-major), Givens (cs,sn), g = Q^T e_1 beta
    std::vector<std::vector<double>> V(m + 1, std::vector<double>(N, 0.0));
    std::vector<double> H((m + 1) * m, 0.0);  // H[i + (m+1)*j]
    std::vector<double> cs(m), sn(m), g(m + 1);

    bool converged = false;
    for (int restart = 0; restart < max_restarts && !converged; ++restart) {
      // Compute initial residual r = M^{-1}(b - A x) and normalise into V[0]
      {
        auto Ax = matvec(x);
        for (int i = 0; i < N; ++i) V[0][i] = rhs[i] - Ax[i];
        V[0] = applyLeftPreconditioner(V[0]);
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
        // w = M^{-1} A * V[j]  (one sparse matvec + Jacobi scaling per step)
        std::vector<double> w = applyLeftPreconditioner(matvec(V[j]));

        // Modified Gram-Schmidt orthogonalisation against V[0..j]
        for (int i = 0; i <= j; ++i) {
          double hij = vdot(w, V[i]);
          H[i + (m + 1) * j] = hij;
          for (int k = 0; k < N; ++k) w[k] -= hij * V[i][k];
        }
        // Double MGS pass for stability
        for (int i = 0; i <= j; ++i) {
          double corr = vdot(w, V[i]);
          H[i + (m + 1) * j] += corr;
          for (int k = 0; k < N; ++k) w[k] -= corr * V[i][k];
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
  if (solver_.p_order == 0) {
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

  // 1. Prolong coarse solution into fine (p+1) space
  auto U_fine = solver_.prolongP1toP2(solver_.U_dg);
  
  // Cache the old coarse adjoint prolongation (psi_H). We DO NOT use this
  // directly for the error anymore; we just need it to find the gradient DIFFERENCE.
  auto psi_prolonged = solver_.prolongP1toP2(psi_);

  // 2. Set up fine solver and evaluate continuous non-linear residual
  fine_solver.p_order = solver_.p_order + 1;
  fine_solver.initializeDG();
  fine_solver.U_dg = U_fine;
  auto res = fine_solver.calcResidualDG(U_fine);

  // 3. TRUE FINE-SPACE ADJOINT SOLVE
  // We explicitly solve the (A_fine^T)*psi_fine = dJ_dU_fine GMRES matrix internally.
  std::cerr << "  Evaluating true fine-space Adjoint for Error Indicators..." << std::endl;
  AdjointSolver fine_adj(fine_solver);
  fine_adj.solve();
  auto psi_fine = fine_adj.psi(); 

  // 4. Element-wise inner product using Dual-Weighted Residual Difference:
  //    eps_e = |(psi_fine - psi_H)^T * R_h,e|
  int ndof_fine = fine_solver.ndof_per_elem;
  for (int e = 0; e < Ne; ++e) {
    double dot = 0.0;
    for (int j = 0; j < ndof_fine; ++j)
      for (int k = 0; k < 4; ++k)
        dot += (psi_fine[e][j][k] - psi_prolonged[e][j][k]) * res.R[e * ndof_fine + j][k];
    eps[e] = std::abs(dot);
  }

  double total = 0.0;
  for (double v : eps) total += v;
  std::cerr << "  Error indicators: sum |eps_e| = " << total
            << " (estimated |delta_Cl|)" << std::endl;

  return eps;
}
