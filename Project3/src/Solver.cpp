#include "Solver.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <limits>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
uint64_t spatialKey(int ix, int iy) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(ix)) << 32) |
         static_cast<uint32_t>(iy);
}

std::vector<Vec4> readStateBinary(const std::string &filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Could not open state file: " + filename);
  }

  int ne_file = 0;
  in.read(reinterpret_cast<char *>(&ne_file), sizeof(int));
  if (!in || ne_file <= 0) {
    throw std::runtime_error("Invalid state header in: " + filename);
  }

  std::vector<Vec4> data(ne_file);
  for (int i = 0; i < ne_file; ++i) {
    in.read(reinterpret_cast<char *>(data[i].v), sizeof(double) * 4);
    if (!in) {
      throw std::runtime_error("Truncated state data in: " + filename);
    }
  }
  return data;
}

std::string residualHistoryFilenameForRestart(const std::string &filename) {
  const std::string dg_tag = "_results_dg.bin";
  const std::string avg_tag = "_results.bin";
  if (filename.size() >= dg_tag.size() &&
      filename.compare(filename.size() - dg_tag.size(), dg_tag.size(), dg_tag) == 0) {
    return filename.substr(0, filename.size() - dg_tag.size()) + "_residual.bin";
  }
  if (filename.size() >= avg_tag.size() &&
      filename.compare(filename.size() - avg_tag.size(), avg_tag.size(), avg_tag) == 0) {
    return filename.substr(0, filename.size() - avg_tag.size()) + "_residual.bin";
  }
  return "";
}

double readBaselineResidualFromHistory(const std::string &filename) {
  const std::string residual_file = residualHistoryFilenameForRestart(filename);
  if (residual_file.empty()) return -1.0;

  std::ifstream in(residual_file, std::ios::binary);
  if (!in) return -1.0;

  int n = 0;
  in.read(reinterpret_cast<char *>(&n), sizeof(int));
  if (!in || n <= 0) return -1.0;

  double baseline = -1.0;
  in.read(reinterpret_cast<char *>(&baseline), sizeof(double));
  if (!in || !std::isfinite(baseline) || baseline <= 0.0) return -1.0;

  return baseline;
}
} // namespace

FiniteVolumeSolver::FiniteVolumeSolver(const std::string &meshfile) {
  if (!mesh.readGRI(meshfile)) {
    throw std::runtime_error("Failed to read mesh file");
  }

  // Don't initialize DG structures yet - p_order will be set after construction
  // DG initialization happens in initializeDG() which is called after p_order is set
  
  p0 = (rho0 * a0 * a0) / gamma;
  pout = 0.7 * p0;
  alpha = alpha * M_PI / 180.0;
  
  // For backward compatibility with p=0 (FV mode), allocate U
  U.resize(mesh.E.size());
  U0.resize(mesh.E.size());
}

void FiniteVolumeSolver::initializeDG() {
  // Call this AFTER p_order is set
  ndof_per_elem = (p_order + 1) * (p_order + 2) / 2;
  std::cout << "Initializing DG solver with p=" << p_order 
            << ", ndof_per_elem=" << ndof_per_elem << std::endl;

  // ── Curved element verification ──────────────────────────────────────────
  // The GRI 'degree' field encodes the geometry order of each element block:
  //   degree 1 → 3 nodes (straight), 2 → 6 nodes (quadratic), 3 → 10 nodes (cubic)
  //
  // Superparametric requirement (from course notes):
  //   p=1 solution  →  needs q >= 2 geometry (superparametric)
  //   p >= 2 solution →  isoparametric (q=p) or subparametric (q<p) acceptable
  {
    int q_geom = mesh.q_order_global;
    std::cout << "  Mesh geometry order: q=" << q_geom;
    if (mesh.has_curved_elements)
      std::cout << " (curved elements present)" << std::endl;
    else
      std::cout << " (straight triangles, no curved elements)" << std::endl;

    if (q_geom >= p_order) {
      std::cout << "  Geometry q=" << q_geom << " >= solution p=" << p_order
                << " — isoparametric or superparametric. Good." << std::endl;
    } else if (p_order == 1 && q_geom < 2) {
      std::cout << "  NOTE: p=1 solution ideally requires superparametric geometry "
                   "(q_geom >= 2) on curved boundaries.  Current mesh is q="
                << q_geom << "; accept some geometric approximation error." << std::endl;
    } else if (p_order >= 2 && q_geom < p_order) {
      std::cout << "  NOTE: p=" << p_order << " solution on q=" << q_geom << " mesh — "
                   "subparametric.  Acceptable per course notes for p>=2." << std::endl;
    }
    // Run the Newton inverse-mapping spot check on element 0
    double xi_c = 0.0, eta_c = 0.0;
    bool ok = mesh.globalToReference(0, mesh.centroids[0], xi_c, eta_c);
    std::cout << "  Inverse mapping check (elem 0 centroid): "
              << (ok ? "converged" : "FAILED")
              << "  xi=" << xi_c << "  eta=" << eta_c
              << "  (expected ~0.333, ~0.333)" << std::endl;
  }
  
  U_dg.resize(mesh.E.size(), std::vector<Vec4>(ndof_per_elem));
  U0_dg.resize(mesh.E.size(), std::vector<Vec4>(ndof_per_elem));
  
  // Compute mass matrix inverse once
  computeMassMatrix();
  
  setInitialCondition();
}

void FiniteVolumeSolver::setInitialCondition() {
  steady_baseline_residual_override = -1.0;
  int Ne = mesh.E.size();
  U_dg.resize(Ne);
  for (int e = 0; e < Ne; ++e) {
    U_dg[e].assign(ndof_per_elem, {0.0, 0.0, 0.0, 0.0});
  }

  double rhou = rho0 * Minf * a0 * std::cos(alpha);
  double rhov = rho0 * Minf * a0 * std::sin(alpha);
  double rhoE = rho0 * (a0 * a0 / ((gamma - 1.0) * gamma) + 0.5 * Minf * Minf);
  Vec4 u_fs = {rho0, rhou, rhov, rhoE};

  // For a nodal Lagrange basis, the L2-projection of a constant u_fs is:
  //   U_dg[e][j] = u_fs  for ALL j
  // because each DOF j corresponds to the nodal value at the j-th Lagrange node,
  // and the reconstruction u(x) = sum_j U_j * phi_j(x) reproduces the constant
  // exactly (partition of unity: sum_j phi_j = 1) only when all U_j = u_fs.
  // Setting only DOF 0 = u_fs and the rest = 0 gives u(x) = phi_0(x) * u_fs,
  // which is NOT constant -- it goes to zero at nodes 1, 2, etc., causing
  // unphysical (zero-density) states at quadrature points on those edges.
  for (int e = 0; e < Ne; ++e) {
    for (int j = 0; j < ndof_per_elem; ++j)
      U_dg[e][j] = u_fs;
  }

  U0_dg = U_dg;

  U.resize(Ne);
  for (int e = 0; e < Ne; ++e) U[e] = U_dg[e][0];
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
  steady_baseline_residual_override = readBaselineResidualFromHistory(filename);

  // Also seed the DG representation: cell-average DOF (index 0) gets the loaded
  // value; higher DOFs are set to the same value so the nodal reconstruction
  // starts as piecewise-constant (the p=0 solution projected into p>0 space).
  if ((int)U_dg.size() == Ne) {
    for (int i = 0; i < Ne; ++i) {
      for (int j = 0; j < ndof_per_elem; ++j)
        U_dg[i][j] = U[i];
    }
    U0_dg = U_dg;
  }

  std::cout << "Successfully loaded initial condition from " << filename << std::endl;
  if (steady_baseline_residual_override > 0.0) {
    std::cout << "Reusing original steady baseline residual from matching history: "
              << std::scientific << std::setprecision(6)
              << steady_baseline_residual_override << std::endl;
  }
}

void FiniteVolumeSolver::loadDGInitialCondition(const std::string &filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) {
    std::cerr << "Warning: Could not open DG initial condition file " << filename
              << ", using default IC instead" << std::endl;
    setInitialCondition();
    return;
  }

  int Ne_file = 0;
  int p_file = -1;
  int ndof_file = -1;
  in.read((char *)&Ne_file, sizeof(int));
  in.read((char *)&p_file, sizeof(int));
  in.read((char *)&ndof_file, sizeof(int));

  int Ne = mesh.E.size();
  if (Ne_file != Ne || p_file != p_order || ndof_file != ndof_per_elem) {
    std::cerr << "Warning: DG initial condition file mismatch: file has (Ne="
              << Ne_file << ", p=" << p_file << ", ndof=" << ndof_file
              << ") but solver expects (Ne=" << Ne << ", p=" << p_order
              << ", ndof=" << ndof_per_elem << "). Using default IC instead."
              << std::endl;
    in.close();
    setInitialCondition();
    return;
  }

  U_dg.assign(Ne, std::vector<Vec4>(ndof_per_elem));
  for (int e = 0; e < Ne; ++e) {
    for (int j = 0; j < ndof_per_elem; ++j) {
      in.read((char *)U_dg[e][j].v, sizeof(double) * 4);
    }
  }
  in.close();
  U0_dg = U_dg;
  steady_baseline_residual_override = readBaselineResidualFromHistory(filename);

  U.resize(Ne);
  for (int e = 0; e < Ne; ++e) {
    U[e] = cellAverage(U_dg[e]);
  }
  U0 = U;

  std::cout << "Successfully loaded DG initial condition from " << filename
            << std::endl;
  if (steady_baseline_residual_override > 0.0) {
    std::cout << "Reusing original steady baseline residual from matching history: "
              << std::scientific << std::setprecision(6)
              << steady_baseline_residual_override << std::endl;
  }
}

void FiniteVolumeSolver::loadMappedInitialCondition(
    const std::string &coarse_meshfile, const std::string &coarse_statefile) {
  steady_baseline_residual_override = -1.0;
  Mesh coarse_mesh;
  if (!coarse_mesh.readGRI(coarse_meshfile)) {
    throw std::runtime_error("Failed to read coarse mesh file: " + coarse_meshfile);
  }

  std::vector<Vec4> coarse_u = readStateBinary(coarse_statefile);
  if (static_cast<int>(coarse_u.size()) != static_cast<int>(coarse_mesh.E.size())) {
    throw std::runtime_error("Coarse state size mismatch: file has " +
                             std::to_string(coarse_u.size()) + " cells, mesh has " +
                             std::to_string(coarse_mesh.E.size()));
  }

  const auto &coarse_centroids = coarse_mesh.centroids;
  if (coarse_centroids.empty()) {
    throw std::runtime_error("Coarse mesh has zero centroids.");
  }

  double xmin = coarse_centroids[0].x, xmax = coarse_centroids[0].x;
  double ymin = coarse_centroids[0].y, ymax = coarse_centroids[0].y;
  for (const auto &c : coarse_centroids) {
    xmin = std::min(xmin, c.x);
    xmax = std::max(xmax, c.x);
    ymin = std::min(ymin, c.y);
    ymax = std::max(ymax, c.y);
  }

  int n_coarse = static_cast<int>(coarse_centroids.size());
  int nx = std::max(8, static_cast<int>(std::sqrt(static_cast<double>(n_coarse))));
  int ny = nx;
  double hx = (xmax - xmin) / static_cast<double>(nx);
  double hy = (ymax - ymin) / static_cast<double>(ny);
  if (hx <= 0.0)
    hx = 1.0;
  if (hy <= 0.0)
    hy = 1.0;

  auto clampIndex = [](int i, int n) { return std::max(0, std::min(n - 1, i)); };
  auto binCoords = [&](const Vec2 &p) {
    int ix = static_cast<int>(std::floor((p.x - xmin) / hx));
    int iy = static_cast<int>(std::floor((p.y - ymin) / hy));
    ix = clampIndex(ix, nx);
    iy = clampIndex(iy, ny);
    return std::pair<int, int>(ix, iy);
  };

  std::unordered_map<uint64_t, std::vector<int>> bins;
  bins.reserve(static_cast<size_t>(n_coarse) * 2);
  for (int i = 0; i < n_coarse; ++i) {
    auto [ix, iy] = binCoords(coarse_centroids[i]);
    bins[spatialKey(ix, iy)].push_back(i);
  }

  int n_fine = static_cast<int>(mesh.E.size());
  U.resize(n_fine);
  int hash_fallback_count = 0;
  int physical_fallback_count = 0;
  const int max_radius = std::max(nx, ny);

  for (int i = 0; i < n_fine; ++i) {
    const Vec2 &cf = mesh.centroids[i];
    auto [ix0, iy0] = binCoords(cf);
    int best = -1;
    double best_d2 = std::numeric_limits<double>::max();

    for (int r = 0; r <= max_radius; ++r) {
      bool found_any = false;
      int x0 = std::max(0, ix0 - r);
      int x1 = std::min(nx - 1, ix0 + r);
      int y0 = std::max(0, iy0 - r);
      int y1 = std::min(ny - 1, iy0 + r);
      for (int bx = x0; bx <= x1; ++bx) {
        for (int by = y0; by <= y1; ++by) {
          auto it = bins.find(spatialKey(bx, by));
          if (it == bins.end())
            continue;
          found_any = true;
          for (int j : it->second) {
            Vec2 d = coarse_centroids[j] - cf;
            double d2 = d.normSq();
            if (d2 < best_d2) {
              best_d2 = d2;
              best = j;
            }
          }
        }
      }
      if (found_any)
        break;
    }

    if (best < 0) {
      hash_fallback_count++;
      for (int j = 0; j < n_coarse; ++j) {
        Vec2 d = coarse_centroids[j] - cf;
        double d2 = d.normSq();
        if (d2 < best_d2) {
          best_d2 = d2;
          best = j;
        }
      }
    }

    Vec4 mapped = coarse_u[best];
    State s(mapped, gamma);
    if (!std::isfinite(s.rho()) || !std::isfinite(s.p()) || s.rho() <= 0.0 ||
        s.p() <= 0.0) {
      physical_fallback_count++;
      mapped = U0[i]; // freestream fallback from constructor IC
    }
    U[i] = mapped;
  }

  U0 = U;
  std::cout << "Mapped IC loaded from coarse solution: " << coarse_statefile
            << " onto mesh with " << n_fine << " cells (coarse cells: "
            << n_coarse << ", hash_fallbacks: " << hash_fallback_count
            << ", physical_fallbacks: " << physical_fallback_count << ")"
            << std::endl;
}

// ═══════════════════════════════════════════════════════════════
// DG BASIS FUNCTIONS (from shape.c)
// ═══════════════════════════════════════════════════════════════
std::vector<double> 
FiniteVolumeSolver::evaluateBasis(double xi, double eta, int p) {
  int ndof = (p+1)*(p+2)/2;
  std::vector<double> phi(ndof);
  
  // Lagrange shape functions on reference triangle
  // Reference coords: (0,0), (1,0), (0,1)
  
  switch (p) {
    case 0:
      phi[0] = 1.0;
      break;
      
    case 1:
      phi[0] = 1.0 - xi - eta;
      phi[1] = xi;
      phi[2] = eta;
      break;
      
    case 2:
      phi[0] = 1.0 - 3.0*xi - 3.0*eta + 2.0*xi*xi + 4.0*xi*eta + 2.0*eta*eta;
      phi[1] = -xi + 2.0*xi*xi;
      phi[2] = -eta + 2.0*eta*eta;
      phi[3] = 4.0*xi*eta;
      phi[4] = 4.0*eta - 4.0*xi*eta - 4.0*eta*eta;
      phi[5] = 4.0*xi - 4.0*xi*xi - 4.0*xi*eta;
      break;
      
    case 3:
      phi[0] = 1.0 - 11.0/2.0*xi - 11.0/2.0*eta + 9.0*xi*xi + 18.0*xi*eta + 9.0*eta*eta 
               - 9.0/2.0*xi*xi*xi - 27.0/2.0*xi*xi*eta - 27.0/2.0*xi*eta*eta - 9.0/2.0*eta*eta*eta;
      phi[1] = xi - 9.0/2.0*xi*xi + 9.0/2.0*xi*xi*xi;
      phi[2] = eta - 9.0/2.0*eta*eta + 9.0/2.0*eta*eta*eta;
      phi[3] = -9.0/2.0*xi*eta + 27.0/2.0*xi*xi*eta;
      phi[4] = -9.0/2.0*xi*eta + 27.0/2.0*xi*eta*eta;
      phi[5] = -9.0/2.0*eta + 9.0/2.0*xi*eta + 18.0*eta*eta - 27.0/2.0*xi*eta*eta - 27.0/2.0*eta*eta*eta;
      phi[6] = 9.0*eta - 45.0/2.0*xi*eta - 45.0/2.0*eta*eta + 27.0/2.0*xi*xi*eta + 27.0*xi*eta*eta + 27.0/2.0*eta*eta*eta;
      phi[7] = 9.0*xi - 45.0/2.0*xi*xi - 45.0/2.0*xi*eta + 27.0/2.0*xi*xi*xi + 27.0*xi*xi*eta + 27.0/2.0*xi*eta*eta;
      phi[8] = -9.0/2.0*xi + 18.0*xi*xi + 9.0/2.0*xi*eta - 27.0/2.0*xi*xi*xi - 27.0/2.0*xi*xi*eta;
      phi[9] = 27.0*xi*eta - 27.0*xi*xi*eta - 27.0*xi*eta*eta;
      break;
      
    default:
      std::cerr << "evaluateBasis: p=" << p << " not implemented!" << std::endl;
      exit(1);
  }
  
  return phi;
}

// ═══════════════════════════════════════════════════════════════
// COMPUTE MASS MATRIX INVERSE
// ═══════════════════════════════════════════════════════════════
void FiniteVolumeSolver::computeMassMatrix() {
  int ndof = ndof_per_elem;
  MassMatrixInv.assign(ndof, std::vector<double>(ndof, 0.0));

  // Build M_ij = integral_T phi_i * phi_j dA on the REFERENCE triangle
  // using the 2-D Gaussian quadrature points from quad2d.c (Dunavant rules).
  // We use enough points to integrate exactly for the given p.
  // Dunavant rule exact for degree d = 2p (flux*basis products).
  // We store the 2-D quadrature as (xi, eta, weight) on the unit right triangle.
  // Here we embed a sufficient set inline for d up to 7 (covers p=0..3).

  // 2D quadrature points (xi, eta, w) on unit right triangle, area=0.5
  // Rule degree 1 (1 pt, exact for deg 1): covers p=0
  // Rule degree 3 (4 pts, exact for deg 3): covers p=1
  // Rule degree 5 (7 pts, exact for deg 5): covers p=2
  // Rule degree 7 (13 pts, exact for deg 7): covers p=3
  struct QP2D { double xi, eta, w; };
  std::vector<QP2D> qpts;

  if (p_order == 0) {
    // 1-point rule, exact degree 1
    qpts = {{1.0/3.0, 1.0/3.0, 0.5}};
  } else if (p_order == 1) {
    // 3-point rule, exact degree 2 (sufficient for M_ij = int phi_i phi_j, max degree 2)
    qpts = {
      {1.0/6.0, 1.0/6.0, 1.0/6.0},
      {2.0/3.0, 1.0/6.0, 1.0/6.0},
      {1.0/6.0, 2.0/3.0, 1.0/6.0},
    };
  } else if (p_order == 2) {
    // 6-point rule, exact degree 4 (needed for phi_i*phi_j at p=2, max degree 4)
    qpts = {
      {0.108103018168070, 0.445948490915965, 0.111690794839005},
      {0.445948490915965, 0.108103018168070, 0.111690794839005},
      {0.445948490915965, 0.445948490915965, 0.111690794839005},
      {0.816847572980459, 0.091576213509771, 0.054975871827661},
      {0.091576213509771, 0.816847572980459, 0.054975871827661},
      {0.091576213509771, 0.091576213509771, 0.054975871827661},
    };
  } else if (p_order == 3) {
    // 12-point rule, exact degree 6 (needed for phi_i*phi_j at p=3, max degree 6)
    qpts = {
      {0.063089014491502, 0.063089014491502, 0.025422453185104},
      {0.063089014491502, 0.873821971016996, 0.025422453185104},
      {0.873821971016996, 0.063089014491502, 0.025422453185104},
      {0.249286745170910, 0.249286745170910, 0.058393137863189},
      {0.249286745170910, 0.501426509658179, 0.058393137863189},
      {0.501426509658179, 0.249286745170910, 0.058393137863189},
      {0.053145049844816, 0.310352451033785, 0.041425537809187},
      {0.053145049844816, 0.636502499121399, 0.041425537809187},
      {0.310352451033785, 0.053145049844816, 0.041425537809187},
      {0.310352451033785, 0.636502499121399, 0.041425537809187},
      {0.636502499121399, 0.053145049844816, 0.041425537809187},
      {0.636502499121399, 0.310352451033785, 0.041425537809187},
    };
  } else {
    std::cerr << "computeMassMatrix: p > 3 not implemented!" << std::endl;
    exit(1);
  }

  // M_ij = sum_q w_q * phi_i(xi_q, eta_q) * phi_j(xi_q, eta_q)
  std::vector<std::vector<double>> M(ndof, std::vector<double>(ndof, 0.0));
  for (const auto &qp : qpts) {
    std::vector<double> phi = evaluateBasis(qp.xi, qp.eta, p_order);
    for (int i = 0; i < ndof; ++i)
      for (int j = 0; j < ndof; ++j)
        M[i][j] += qp.w * phi[i] * phi[j];
  }

  // Invert M using Gauss-Jordan elimination
  // Augment [M | I]
  std::vector<std::vector<double>> aug(ndof, std::vector<double>(2*ndof, 0.0));
  for (int i = 0; i < ndof; ++i) {
    for (int j = 0; j < ndof; ++j) aug[i][j] = M[i][j];
    aug[i][ndof + i] = 1.0;
  }
  for (int col = 0; col < ndof; ++col) {
    // Pivot
    int pivot = col;
    for (int row = col+1; row < ndof; ++row)
      if (std::abs(aug[row][col]) > std::abs(aug[pivot][col])) pivot = row;
    std::swap(aug[col], aug[pivot]);
    double diag = aug[col][col];
    if (std::abs(diag) < 1e-14) {
      std::cerr << "computeMassMatrix: singular mass matrix at p=" << p_order << std::endl;
      exit(1);
    }
    for (int j = 0; j < 2*ndof; ++j) aug[col][j] /= diag;
    for (int row = 0; row < ndof; ++row) {
      if (row == col) continue;
      double factor = aug[row][col];
      for (int j = 0; j < 2*ndof; ++j)
        aug[row][j] -= factor * aug[col][j];
    }
  }
  for (int i = 0; i < ndof; ++i)
    for (int j = 0; j < ndof; ++j)
      MassMatrixInv[i][j] = aug[i][ndof + j];

  // Compute cell-average weights:  w_j = int(phi_j dA_ref) / A_ref
  // These are used by cellAverage() 
  // For p=0: w=[1.0] — DOF 0 is the cell average.
  // For p=1: w=[1/3,1/3,1/3] — average of 3 vertex values.
  // For p=2: w=[0,0,0,1/3,1/3,1/3] — corner DOFs integrate to zero
  cellAvgWeights.assign(ndof, 0.0);
  for (const auto &qp : qpts) {
    std::vector<double> phi = evaluateBasis(qp.xi, qp.eta, p_order);
    for (int j = 0; j < ndof; ++j)
      cellAvgWeights[j] += qp.w;  // int phi_j / A_ref: qp.w already includes the weight
  }
  // w_j = (1/A_ref) * integral phi_j dA_ref = (1/0.5) * sum_q (w_q * phi_j(xi_q,eta_q))
  cellAvgWeights.assign(ndof, 0.0);
  for (const auto &qp : qpts) {
    std::vector<double> phi = evaluateBasis(qp.xi, qp.eta, p_order);
    for (int j = 0; j < ndof; ++j)
      cellAvgWeights[j] += (1.0/0.5) * qp.w * phi[j];
  }

  // Compute spectral radius of M_ref^{-1} via power iteration.
  // This is the factor by which the effective time step is amplified relative to p=0 (where it is 2).
  // Used to scale CFL down so that the nodal DG scheme remains stable.
  {
    std::vector<double> vec(ndof, 1.0 / std::sqrt(static_cast<double>(ndof)));
    double lambda = 0.0;
    for (int iter = 0; iter < 200; ++iter) {
      std::vector<double> Mv(ndof, 0.0);
      for (int i = 0; i < ndof; ++i)
        for (int j = 0; j < ndof; ++j)
          Mv[i] += MassMatrixInv[i][j] * vec[j];
      double norm = 0.0;
      for (int i = 0; i < ndof; ++i) norm += Mv[i] * Mv[i];
      norm = std::sqrt(norm);
      lambda = norm;
      for (int i = 0; i < ndof; ++i) vec[i] = Mv[i] / norm;
    }
    mass_spectral_radius = lambda;
    std::cout << "  M_ref^{-1} spectral radius (p=" << p_order << "): " 
              << mass_spectral_radius << " -> CFL scale factor 2/sr: "
              << 2.0 / mass_spectral_radius << std::endl;
  }
}

// cellAverage — compute the true L2 cell average from nodal DOF vector
// For a nodal Lagrange basis, the cell average is:
//   u_bar = (1/A_ref) * integral u_h dA_ref
//          = sum_j w_j * U_j
// where w_j = (1/A_ref) * integral phi_j dA_ref (pre-computed in cellAvgWeights).
Vec4 FiniteVolumeSolver::cellAverage(const std::vector<Vec4> &dofs) const {
  Vec4 avg = {0,0,0,0};
  for (int j = 0; j < (int)dofs.size(); ++j)
    avg += dofs[j] * cellAvgWeights[j];
  return avg;
}

std::vector<double>
FiniteVolumeSolver::evaluateBasisGrad(double xi, double eta, int p) {
  // Returns [dφ_0/dξ, dφ_1/dξ, ..., dφ_{n-1}/dξ,
  //          dφ_0/dη, dφ_1/dη, ..., dφ_{n-1}/dη]
  // i.e., first ndof entries are ∂φ_i/∂ξ, next ndof entries are ∂φ_i/∂η
  int ndof = (p+1)*(p+2)/2;
  std::vector<double> g(2*ndof, 0.0);
  // dxi part: g[0..ndof-1], deta part: g[ndof..2*ndof-1]

  switch (p) {
    case 0:
      // φ_0 = 1 → all gradients zero
      break;

    case 1:
      // φ_0=1-ξ-η, φ_1=ξ, φ_2=η
      g[0] = -1.0; g[1] = 1.0; g[2] = 0.0;   // dφ_i/dξ
      g[3] = -1.0; g[4] = 0.0; g[5] = 1.0;   // dφ_i/dη
      break;

    case 2:
      // φ_0 = 1-3ξ-3η+2ξ²+4ξη+2η²
      // φ_1 = -ξ+2ξ²
      // φ_2 = -η+2η²
      // φ_3 = 4ξη
      // φ_4 = 4η-4ξη-4η²
      // φ_5 = 4ξ-4ξ²-4ξη
      g[0] = -3.0 + 4.0*xi + 4.0*eta;  g[ndof+0] = -3.0 + 4.0*xi + 4.0*eta;
      g[1] = -1.0 + 4.0*xi;            g[ndof+1] = 0.0;
      g[2] = 0.0;                       g[ndof+2] = -1.0 + 4.0*eta;
      g[3] = 4.0*eta;                   g[ndof+3] = 4.0*xi;
      g[4] = -4.0*eta;                  g[ndof+4] = 4.0 - 4.0*xi - 8.0*eta;
      g[5] = 4.0 - 8.0*xi - 4.0*eta;   g[ndof+5] = -4.0*xi;
      break;

    case 3: {
      // Gradients of the p=3 Lagrange basis from evaluateBasis
      // φ_0 = 1 - (11/2)ξ - (11/2)η + 9ξ²+18ξη+9η² - (9/2)ξ³-(27/2)ξ²η-(27/2)ξη²-(9/2)η³
      g[0]       = -11.0/2.0 + 18.0*xi + 18.0*eta - 27.0/2.0*xi*xi - 27.0*xi*eta - 27.0/2.0*eta*eta;
      g[ndof+0]  = -11.0/2.0 + 18.0*xi + 18.0*eta - 27.0/2.0*xi*xi - 27.0*xi*eta - 27.0/2.0*eta*eta;
      // φ_1 = ξ - (9/2)ξ² + (9/2)ξ³
      g[1]       = 1.0 - 9.0*xi + 27.0/2.0*xi*xi;
      g[ndof+1]  = 0.0;
      // φ_2 = η - (9/2)η² + (9/2)η³
      g[2]       = 0.0;
      g[ndof+2]  = 1.0 - 9.0*eta + 27.0/2.0*eta*eta;
      // φ_3 = -(9/2)ξη + (27/2)ξ²η
      g[3]       = -9.0/2.0*eta + 27.0*xi*eta;
      g[ndof+3]  = -9.0/2.0*xi + 27.0/2.0*xi*xi;
      // φ_4 = -(9/2)ξη + (27/2)ξη²
      g[4]       = -9.0/2.0*eta + 27.0/2.0*eta*eta;
      g[ndof+4]  = -9.0/2.0*xi + 27.0*xi*eta;
      // φ_5 = -(9/2)η + (9/2)ξη + 18η² - (27/2)ξη² - (27/2)η³
      g[5]       = 9.0/2.0*eta - 27.0/2.0*eta*eta;
      g[ndof+5]  = -9.0/2.0 + 9.0/2.0*xi + 36.0*eta - 27.0*xi*eta - 81.0/2.0*eta*eta;
      // φ_6 = 9η - (45/2)ξη - (45/2)η² + (27/2)ξ²η + 27ξη² + (27/2)η³
      g[6]       = -45.0/2.0*eta + 27.0*xi*eta + 27.0*eta*eta;
      g[ndof+6]  = 9.0 - 45.0/2.0*xi - 45.0*eta + 27.0/2.0*xi*xi + 54.0*xi*eta + 81.0/2.0*eta*eta;
      // φ_7 = 9ξ - (45/2)ξ² - (45/2)ξη + (27/2)ξ³ + 27ξ²η + (27/2)ξη²
      g[7]       = 9.0 - 45.0*xi - 45.0/2.0*eta + 81.0/2.0*xi*xi + 54.0*xi*eta + 27.0/2.0*eta*eta;
      g[ndof+7]  = -45.0/2.0*xi + 27.0*xi*xi + 27.0*xi*eta;
      // φ_8 = -(9/2)ξ + 18ξ² + (9/2)ξη - (27/2)ξ³ - (27/2)ξ²η
      g[8]       = -9.0/2.0 + 36.0*xi + 9.0/2.0*eta - 81.0/2.0*xi*xi - 27.0*xi*eta;
      g[ndof+8]  = 9.0/2.0*xi - 27.0/2.0*xi*xi;
      // φ_9 = 27ξη - 27ξ²η - 27ξη²
      g[9]       = 27.0*eta - 54.0*xi*eta - 27.0*eta*eta;
      g[ndof+9]  = 27.0*xi - 27.0*xi*xi - 54.0*xi*eta;
      break;
    }

    default:
      std::cerr << "evaluateBasisGrad: p > 3 not implemented!" << std::endl;
      exit(1);
  }

  return g;
}

// ═══════════════════════════════════════════════════════════════
// QUADRATURE HELPER FUNCTION
// ═══════════════════════════════════════════════════════════════
FiniteVolumeSolver::QuadRule 
FiniteVolumeSolver::getQuadratureRule(int p_order) {
  QuadRule qr;
  
  // Choose number of points based on polynomial order
  // For p-order basis, need to integrate degree ~2p polynomials (flux × basis)
  // Use p+1 or p+2 points to be safe
  
  if (p_order == 0) {
    // 1 point (exact for degree 1)
    qr.n = 1;
    qr.points = {0.5};
    qr.weights = {1.0};
  }
  else if (p_order == 1) {
    // 2 points (exact for degree 3)
    qr.n = 2;
    qr.points = {0.211324865405187, 0.788675134594813};
    qr.weights = {0.5, 0.5};
  }
  else if (p_order == 2) {
    // 3 points (exact for degree 5)
    qr.n = 3;
    qr.points = {0.112701665379258, 0.5, 0.887298334620742};
    qr.weights = {0.277777777777778, 0.444444444444444, 0.277777777777778};
  }
  else if (p_order == 3) {
    // 4 points (exact for degree 7)
    qr.n = 4;
    qr.points = {0.069431844202974, 0.330009478207572, 
                 0.669990521792428, 0.930568155797026};
    qr.weights = {0.173927422568727, 0.326072577431273,
                  0.326072577431273, 0.173927422568727};
  }
  else {
    std::cerr << "Quadrature not implemented for p > 3!" << std::endl;
    exit(1);
  }
  
  return qr;
}

// ═══════════════════════════════════════════════════════════════
// ORIGINAL FV RESIDUAL (for backward compatibility)
// ═══════════════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════════════
// DG RESIDUAL CALCULATION
// ═══════════════════════════════════════════════════════════════

// Given a triangle element with global vertex indices v[0],v[1],v[2]
// and a shared edge defined by global vertex indices (va, vb),
// return the reference-coordinate parameterization of that edge as:
//   xi(t) = xi0 + t*(xi1-xi0),  eta(t) = eta0 + t*(eta1-eta0),  t in [0,1]
// where (xi0,eta0) is the reference position of va and (xi1,eta1) of vb.
// Returns false if neither vertex is found (shouldn't happen).
static bool edgeRefParam(const int ev[3], int va, int vb,
                         double &xi0, double &eta0,
                         double &xi1, double &eta1) {
  // Reference coordinates of the three vertices of the unit right triangle
  // v[0] -> (0,0),  v[1] -> (1,0),  v[2] -> (0,1)
  const double rx[3] = {0.0, 1.0, 0.0};
  const double ry[3] = {0.0, 0.0, 1.0};

  int ia = -1, ib = -1;
  for (int k = 0; k < 3; ++k) {
    if (ev[k] == va) ia = k;
    if (ev[k] == vb) ib = k;
  }
  if (ia < 0 || ib < 0) return false;
  xi0  = rx[ia]; eta0 = ry[ia];
  xi1  = rx[ib]; eta1 = ry[ib];
  return true;
}

FiniteVolumeSolver::ResidualResult
FiniteVolumeSolver::calcResidualDG(const std::vector<std::vector<Vec4>> &Un_dg,
                                    double time, bool use_unsteady_wake) {
  int Ne = mesh.E.size();
  ResidualResult res;
  res.R.resize(Ne * ndof_per_elem, {0.0, 0.0, 0.0, 0.0});
  res.sdl.assign(Ne, 0.0);

  QuadRule qr = getQuadratureRule(p_order);

  // For p>0, several face-time-step and monitoring calculations only need the
  // cell-average state.  Cache it once per residual evaluation instead of
  // recomputing cellAverage(...) repeatedly inside the face loops.
  std::vector<Vec4> Uavg;
  if (p_order > 0) {
    Uavg.resize(Ne);
    for (int e = 0; e < Ne; ++e) {
      Uavg[e] = cellAverage(Un_dg[e]);
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // STEP 4a — VOLUME INTEGRALS (element interiors, loop first per spec)
  // R_j -= ∫_T F(u) · ∇_x φ_j dV   (the "by parts" volume term, Eq. 4.3.12 second term)
  // Uses Dunavant 2D quadrature rules.  For p=0 the gradient is zero so this
  // loop is a no-op, but we always execute it to follow the prescribed order.
  // ═══════════════════════════════════════════════════════════════
  {
    struct QP2D { double xi, eta, w; };
    struct VolumeQuadCache {
      std::vector<QP2D> qpts;
      std::vector<std::vector<double>> phi;
      std::vector<std::vector<double>> gphi;
    };

    auto buildVolumeQuadCache = [&](int p) {
      VolumeQuadCache cache;
      if (p == 0) {
        cache.qpts = {{1.0/3.0, 1.0/3.0, 0.5}};
      } else if (p == 1) {
        cache.qpts = {
          {0.108103018168070, 0.445948490915965, 0.111690794839005},
          {0.445948490915965, 0.108103018168070, 0.111690794839005},
          {0.445948490915965, 0.445948490915965, 0.111690794839005},
          {0.816847572980459, 0.091576213509771, 0.054975871827661},
          {0.091576213509771, 0.816847572980459, 0.054975871827661},
          {0.091576213509771, 0.091576213509771, 0.054975871827661},
        };
      } else if (p == 2) {
        cache.qpts = {
          {0.333333333333333, 0.333333333333333, 0.112500000000000},
          {0.059715871789770, 0.470142064105115, 0.066197076394253},
          {0.470142064105115, 0.059715871789770, 0.066197076394253},
          {0.470142064105115, 0.470142064105115, 0.066197076394253},
          {0.797426985353087, 0.101286507323456, 0.062969590272414},
          {0.101286507323456, 0.797426985353087, 0.062969590272414},
          {0.101286507323456, 0.101286507323456, 0.062969590272414},
        };
      } else {
        cache.qpts = {
          {0.333333333333333, 0.333333333333333, 0.072157803838894},
          {0.081414823414554, 0.459292588292723, 0.047545817133642},
          {0.459292588292723, 0.081414823414554, 0.047545817133642},
          {0.459292588292723, 0.459292588292723, 0.047545817133642},
          {0.658861384496480, 0.170569307751760, 0.051608685267359},
          {0.170569307751760, 0.658861384496480, 0.051608685267359},
          {0.170569307751760, 0.170569307751760, 0.051608685267359},
          {0.898905543365938, 0.050547228317031, 0.016229248811599},
          {0.050547228317031, 0.898905543365938, 0.016229248811599},
          {0.050547228317031, 0.050547228317031, 0.016229248811599},
          {0.008394777409958, 0.263112829634638, 0.013615157087217},
          {0.263112829634638, 0.728492392955404, 0.013615157087217},
          {0.728492392955404, 0.008394777409958, 0.013615157087217},
          {0.263112829634638, 0.008394777409958, 0.013615157087217},
          {0.728492392955404, 0.263112829634638, 0.013615157087217},
          {0.008394777409958, 0.728492392955404, 0.013615157087217},
        };
      }
      cache.phi.reserve(cache.qpts.size());
      cache.gphi.reserve(cache.qpts.size());
      for (const auto &qp : cache.qpts) {
        cache.phi.push_back(evaluateBasis(qp.xi, qp.eta, p));
        cache.gphi.push_back(evaluateBasisGrad(qp.xi, qp.eta, p));
      }
      return cache;
    };

    // These quadrature points are fixed for each polynomial order, so cache the
    // basis data once and reuse it for every element in the volume loop.
    static const VolumeQuadCache vol_cache_p0 = buildVolumeQuadCache(0);
    static const VolumeQuadCache vol_cache_p1 = buildVolumeQuadCache(1);
    static const VolumeQuadCache vol_cache_p2 = buildVolumeQuadCache(2);
    static const VolumeQuadCache vol_cache_p3 = buildVolumeQuadCache(3);
    const VolumeQuadCache *vol_cache =
        (p_order == 0) ? &vol_cache_p0 :
        (p_order == 1) ? &vol_cache_p1 :
        (p_order == 2) ? &vol_cache_p2 : &vol_cache_p3;

    // Volume contributions are element-local, so each thread can safely own one element.
#pragma omp parallel for if (Ne > 64)
    for (int e = 0; e < Ne; ++e) {
      bool affine_geom = (mesh.E[e].q_order == 1);
      double dx1 = 0.0, dx2 = 0.0, dy1 = 0.0, dy2 = 0.0;
      if (affine_geom) {
        Vec2 v0 = mesh.V[mesh.E[e].v[0]];
        Vec2 v1 = mesh.V[mesh.E[e].v[1]];
        Vec2 v2 = mesh.V[mesh.E[e].v[2]];
        dx1 = v1.x - v0.x;
        dx2 = v2.x - v0.x;
        dy1 = v1.y - v0.y;
        dy2 = v2.y - v0.y;
      }

      for (size_t q = 0; q < vol_cache->qpts.size(); ++q) {
        const auto &qp = vol_cache->qpts[q];
        const auto &phi  = vol_cache->phi[q];
        const auto &gphi = vol_cache->gphi[q];

        Vec4 u = {0,0,0,0};
        for (int j = 0; j < ndof_per_elem; ++j)
          u += Un_dg[e][j] * phi[j];

        State s(u, gamma);
        double rho = s.rho(), p_pres = s.p(), e_rhoE = u[3];
        double vel_u = u[1]/rho, vel_v = u[2]/rho;
        Vec4 Fx = {rho*vel_u, rho*vel_u*vel_u + p_pres, rho*vel_u*vel_v, (e_rhoE + p_pres)*vel_u};
        Vec4 Fy = {rho*vel_v, rho*vel_u*vel_v, rho*vel_v*vel_v + p_pres, (e_rhoE + p_pres)*vel_v};

        double wvol = qp.w;
        double J00 = 0.0, J01 = 0.0, J10 = 0.0, J11 = 0.0, detJ = 0.0;
        if (affine_geom) {
          J00 = dx1; J01 = dx2;
          J10 = dy1; J11 = dy2;
          detJ = dx1 * dy2 - dx2 * dy1;
          wvol *= detJ;
        } else {
          ElementGeomEval geom = mesh.evaluateElementGeometry(e, qp.xi, qp.eta);
          J00 = geom.dx_dxi;  J01 = geom.dx_deta;
          J10 = geom.dy_dxi;  J11 = geom.dy_deta;
          detJ = geom.detJ;
          wvol *= detJ;
        }

        for (int j = 0; j < ndof_per_elem; ++j) {
          double dphidxi  = gphi[j];
          double dphideta = gphi[ndof_per_elem + j];
          double dphidx =  ( J11 * dphidxi - J10 * dphideta) / detJ;
          double dphidy =  (-J01 * dphidxi + J00 * dphideta) / detJ;
          res.R[e * ndof_per_elem + j] -= (Fx * dphidx + Fy * dphidy) * wvol;
        }
      }
    }
  }

  // Precompute basis values on each directed local edge at the fixed 1-D face
  // quadrature points.  For straight triangles the reference-space edge points
  // depend only on local endpoint ordering, not on the specific element.
  struct FaceQuadCache {
    std::vector<std::vector<std::vector<std::vector<double>>>> phi;
  };

  auto buildFaceQuadCache = [&](int p, const QuadRule &rule) {
    FaceQuadCache cache;
    int ndof = (p + 1) * (p + 2) / 2;
    cache.phi.assign(3, std::vector<std::vector<std::vector<double>>>(
                            3, std::vector<std::vector<double>>(
                                   rule.n, std::vector<double>(ndof, 0.0))));

    const double rx[3] = {0.0, 1.0, 0.0};
    const double ry[3] = {0.0, 0.0, 1.0};
    for (int ia = 0; ia < 3; ++ia) {
      for (int ib = 0; ib < 3; ++ib) {
        if (ia == ib) continue;
        for (int q = 0; q < rule.n; ++q) {
          double t = rule.points[q];
          double xi  = rx[ia] + t * (rx[ib] - rx[ia]);
          double eta = ry[ia] + t * (ry[ib] - ry[ia]);
          cache.phi[ia][ib][q] = evaluateBasis(xi, eta, p);
        }
      }
    }
    return cache;
  };

  const FaceQuadCache *face_cache = nullptr;
  if (p_order > 0) {
    static const FaceQuadCache face_cache_p1 = buildFaceQuadCache(1, getQuadratureRule(1));
    static const FaceQuadCache face_cache_p2 = buildFaceQuadCache(2, getQuadratureRule(2));
    static const FaceQuadCache face_cache_p3 = buildFaceQuadCache(3, getQuadratureRule(3));
    face_cache = (p_order == 1) ? &face_cache_p1 :
                 (p_order == 2) ? &face_cache_p2 : &face_cache_p3;
  }

  // ═══════════════════════════════════════════════════════════════
  // STEP 4b — INTERIOR INTERFACE FLUXES (Eq. 4.3.12 third term)
  // ═══════════════════════════════════════════════════════════════
  auto accumulateInteriorFace = [&](int i, std::vector<Vec4> &Racc,
                                    std::vector<double> &sdlacc) {
    int eL = mesh.IE[i].elemL;
    int eR = mesh.IE[i].elemR;
    int va  = mesh.IE[i].v[0];
    int vb  = mesh.IE[i].v[1];
    Vec2 normal = mesh.inormals[i];
    double len   = mesh.ilengths[i];

    if (p_order == 0) {
      FluxResult fr;
      if (fluxname == "hlle") fr = fluxHLLE(Un_dg[eL][0], Un_dg[eR][0], normal, gamma);
      else                     fr = fluxRoe (Un_dg[eL][0], Un_dg[eR][0], normal, gamma);
      Racc[eL * ndof_per_elem] += fr.F * len;
      Racc[eR * ndof_per_elem] -= fr.F * len;
      sdlacc[eL] += fr.smax * len;
      sdlacc[eR] += fr.smax * len;
    } else {
      // Find reference parameterizations for this edge in each element.
      // For periodic edges, the right element's edge nodes differ from the
      // left element's (top vs bottom periodic nodes).  mesh.IE[i].vR[]
      // stores the correct right-side vertex indices.
      int vaR = mesh.IE[i].vR[0];
      int vbR = mesh.IE[i].vR[1];
      double xiL0, etaL0, xiL1, etaL1;
      double xiR0, etaR0, xiR1, etaR1;
      edgeRefParam(mesh.E[eL].v, va, vb, xiL0, etaL0, xiL1, etaL1);
      edgeRefParam(mesh.E[eR].v, vaR, vbR, xiR0, etaR0, xiR1, etaR1);

      int iaL = -1, ibL = -1, iaR = -1, ibR = -1;
      for (int k = 0; k < 3; ++k) {
        if (mesh.E[eL].v[k] == va)  iaL = k;
        if (mesh.E[eL].v[k] == vb)  ibL = k;
        if (mesh.E[eR].v[k] == vaR) iaR = k;
        if (mesh.E[eR].v[k] == vbR) ibR = k;
      }

      for (int q = 0; q < qr.n; ++q) {
        double t = qr.points[q];
        const auto &phiL =
            (iaL >= 0 && ibL >= 0) ? face_cache->phi[iaL][ibL][q]
                                   : evaluateBasis(xiL0 + t * (xiL1 - xiL0),
                                                   etaL0 + t * (etaL1 - etaL0), p_order);
        const auto &phiR =
            (iaR >= 0 && ibR >= 0) ? face_cache->phi[iaR][ibR][q]
                                   : evaluateBasis(xiR0 + t * (xiR1 - xiR0),
                                                   etaR0 + t * (etaR1 - etaR0), p_order);

        Vec4 u_L = {0,0,0,0}, u_R = {0,0,0,0};
        for (int j = 0; j < ndof_per_elem; ++j) {
          u_L += Un_dg[eL][j] * phiL[j];
          u_R += Un_dg[eR][j] * phiR[j];
        }

        FluxResult fr;
        if (fluxname == "hlle") fr = fluxHLLE(u_L, u_R, normal, gamma);
        else                     fr = fluxRoe (u_L, u_R, normal, gamma);

        double w = qr.weights[q] * len;
        for (int j = 0; j < ndof_per_elem; ++j) {
          Racc[eL * ndof_per_elem + j] += fr.F * (w * phiL[j]);
          Racc[eR * ndof_per_elem + j] -= fr.F * (w * phiR[j]);
        }
      }
      // Use the true cell-average state to compute sdl for time step sizing.
      // For p>=1 DOF 0 is the vertex value, not the integral mean — use cellAverage().
      {
        const Vec4 &uavgL = Uavg[eL];
        const Vec4 &uavgR = Uavg[eR];
        FluxResult fr0;
        if (fluxname == "hlle") fr0 = fluxHLLE(uavgL, uavgR, normal, gamma);
        else                     fr0 = fluxRoe (uavgL, uavgR, normal, gamma);
        sdlacc[eL] += fr0.smax * len;
        sdlacc[eR] += fr0.smax * len;
      }
    }
  };

  int ie_threads = 1;
#ifdef _OPENMP
  ie_threads = omp_get_max_threads();
#endif
  if (ie_threads > 1 && (int)mesh.IE.size() > 128 &&
      !mesh.ie_faces_by_color.empty()) {
    // Interior faces are edge-colored so faces within one color never update
    // the same element. That allows direct shared accumulation without large
    // thread-local residual arrays or a global reduction.
    for (const auto &faces : mesh.ie_faces_by_color) {
#pragma omp parallel for if (faces.size() > 16)
      for (int k = 0; k < (int)faces.size(); ++k) {
        accumulateInteriorFace(faces[k], res.R, res.sdl);
      }
    }
  } else {
    for (int i = 0; i < (int)mesh.IE.size(); ++i) {
      accumulateInteriorFace(i, res.R, res.sdl);
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // STEP 4c — BOUNDARY INTERFACE FLUXES (special treatment per spec)
  // ═══════════════════════════════════════════════════════════════
  auto accumulateBoundaryFace = [&](int i, std::vector<Vec4> &Racc,
                                    std::vector<double> &sdlacc) {
    int eL   = mesh.BE[i].elemL;
    int va   = mesh.BE[i].v[0];
    int vb   = mesh.BE[i].v[1];
    std::string bName = mesh.Bname[mesh.BE[i].bIndex];
    Vec2 n   = mesh.bnormals[i];
    double len = mesh.blengths[i];
    QuadRule bqr = (mesh.E[eL].q_order > p_order)
                       ? getQuadratureRule(mesh.E[eL].q_order)
                       : qr;

    if (p_order == 0) {
      Vec4 u_int = Un_dg[eL][0];
      if (mesh.E[eL].q_order == 1) {
        Vec2 edge_midpoint = (mesh.V[va] + mesh.V[vb]) * 0.5;
        FluxResult fr;
        if (bName == "inflow") {
          Vec4 Ub = subsonicInflow(u_int, n, rho0, a0, alpha, gamma,
                                   edge_midpoint.y, time, use_unsteady_wake);
          if (fluxname == "hlle") fr = fluxHLLE(u_int, Ub, n, gamma);
          else                     fr = fluxRoe (u_int, Ub, n, gamma);
        } else if (bName == "outflow") {
          Vec4 Ub = subsonicOutflow(u_int, n, pout, gamma);
          if (fluxname == "hlle") fr = fluxHLLE(u_int, Ub, n, gamma);
          else                     fr = fluxRoe (u_int, Ub, n, gamma);
        } else {
          fr = inviscidWallFlux(u_int, n, gamma);
        }
        Racc[eL * ndof_per_elem] += fr.F * len;
        sdlacc[eL] += fr.smax * len;
      } else {
        static bool printed_curved_wall_debug = false;
        for (int q = 0; q < bqr.n; ++q) {
          double t = bqr.points[q];
          EdgeGeomEval edge_geom = mesh.evaluateEdgeGeometry(eL, va, vb, t);
          Vec2 n_q = edge_geom.normal;
          if (!printed_curved_wall_debug && bName == "wall") {
            std::cout << "Curved wall edge debug:"
                      << " elem=" << eL
                      << " edge=(" << va << "," << vb << ")"
                      << " qpt=" << q
                      << " t=" << t
                      << " x=(" << edge_geom.x.x << "," << edge_geom.x.y << ")"
                      << " tan=(" << edge_geom.tangent.x << "," << edge_geom.tangent.y << ")"
                      << " n=(" << n_q.x << "," << n_q.y << ")"
                      << " ds_dt=" << edge_geom.ds_dt
                      << std::endl;
            if (q == qr.n - 1) {
              printed_curved_wall_debug = true;
            }
          }
          FluxResult fr;
          if (bName == "inflow") {
            Vec4 Ub = subsonicInflow(u_int, n_q, rho0, a0, alpha, gamma,
                                     edge_geom.x.y, time, use_unsteady_wake);
            if (fluxname == "hlle") fr = fluxHLLE(u_int, Ub, n_q, gamma);
            else                     fr = fluxRoe (u_int, Ub, n_q, gamma);
          } else if (bName == "outflow") {
            Vec4 Ub = subsonicOutflow(u_int, n_q, pout, gamma);
            if (fluxname == "hlle") fr = fluxHLLE(u_int, Ub, n_q, gamma);
            else                     fr = fluxRoe (u_int, Ub, n_q, gamma);
          } else {
            fr = inviscidWallFlux(u_int, n_q, gamma);
          }
          double w = bqr.weights[q] * edge_geom.ds_dt;
          Racc[eL * ndof_per_elem] += fr.F * w;
          sdlacc[eL] += fr.smax * w;
        }
      }
    } else {
      double xi0, eta0, xi1, eta1;
      edgeRefParam(mesh.E[eL].v, va, vb, xi0, eta0, xi1, eta1);
      int ia = -1, ib = -1;
      for (int k = 0; k < 3; ++k) {
        if (mesh.E[eL].v[k] == va) ia = k;
        if (mesh.E[eL].v[k] == vb) ib = k;
      }

      bool use_cached_face_basis =
          (bqr.n == qr.n && ia >= 0 && ib >= 0);
      for (int q = 0; q < bqr.n; ++q) {
        double t    = bqr.points[q];
        std::vector<double> phi_eval;
        const std::vector<double> *phi_ptr = nullptr;
        if (use_cached_face_basis) {
          phi_ptr = &face_cache->phi[ia][ib][q];
        } else {
          phi_eval = evaluateBasis(xi0 + t * (xi1 - xi0),
                                   eta0 + t * (eta1 - eta0), p_order);
          phi_ptr = &phi_eval;
        }
        const auto &phi = *phi_ptr;
        EdgeGeomEval edge_geom = mesh.evaluateEdgeGeometry(eL, va, vb, t);
        Vec2 n_q = edge_geom.normal;

        Vec4 u_int = {0,0,0,0};
        for (int j = 0; j < ndof_per_elem; ++j)
          u_int += Un_dg[eL][j] * phi[j];

        FluxResult fr;
        if (bName == "inflow") {
          Vec4 Ub = subsonicInflow(u_int, n_q, rho0, a0, alpha, gamma,
                                   edge_geom.x.y, time, use_unsteady_wake);
          if (fluxname == "hlle") fr = fluxHLLE(u_int, Ub, n_q, gamma);
          else                     fr = fluxRoe (u_int, Ub, n_q, gamma);
        } else if (bName == "outflow") {
          Vec4 Ub = subsonicOutflow(u_int, n_q, pout, gamma);
          if (fluxname == "hlle") fr = fluxHLLE(u_int, Ub, n_q, gamma);
          else                     fr = fluxRoe (u_int, Ub, n_q, gamma);
        } else {
          fr = inviscidWallFlux(u_int, n_q, gamma);
        }

        double w = bqr.weights[q] * edge_geom.ds_dt;
        for (int j = 0; j < ndof_per_elem; ++j)
          Racc[eL * ndof_per_elem + j] += fr.F * (w * phi[j]);
        sdlacc[eL] += fr.smax * w;
      }
      // Use the true cell-average state for sdl computation.
      // For p>=1 DOF 0 is the vertex value at (0,0), not the integral mean.
      if (mesh.E[eL].q_order == 1) {
        const Vec4 &u_int0 = Uavg[eL];
        FluxResult fr0;
        if (bName == "inflow") {
          Vec2 edge_mid = (mesh.V[va] + mesh.V[vb]) * 0.5;
          Vec4 Ub0 = subsonicInflow(u_int0, n, rho0, a0, alpha, gamma, edge_mid.y, time, use_unsteady_wake);
          fr0 = (fluxname == "hlle") ? fluxHLLE(u_int0, Ub0, n, gamma) : fluxRoe(u_int0, Ub0, n, gamma);
        } else if (bName == "outflow") {
          Vec4 Ub0 = subsonicOutflow(u_int0, n, pout, gamma);
          fr0 = (fluxname == "hlle") ? fluxHLLE(u_int0, Ub0, n, gamma) : fluxRoe(u_int0, Ub0, n, gamma);
        } else {
          fr0 = inviscidWallFlux(u_int0, n, gamma);
        }
        sdlacc[eL] += fr0.smax * len;
      }
    }
  };

  int be_threads = 1;
#ifdef _OPENMP
  be_threads = omp_get_max_threads();
#endif
  if (be_threads > 1 && (int)mesh.BE.size() > 128) {
    // Boundary faces only write into their owning element, but multiple faces
    // may still hit the same element, so use thread-local buffers here as well.
    std::vector<std::vector<Vec4>> Rpriv(
        be_threads, std::vector<Vec4>(Ne * ndof_per_elem, {0.0, 0.0, 0.0, 0.0}));
    std::vector<std::vector<double>> sdlpriv(
        be_threads, std::vector<double>(Ne, 0.0));
#pragma omp parallel
    {
#ifdef _OPENMP
      int tid = omp_get_thread_num();
#else
      int tid = 0;
#endif
#pragma omp for
      for (int i = 0; i < (int)mesh.BE.size(); ++i) {
        accumulateBoundaryFace(i, Rpriv[tid], sdlpriv[tid]);
      }
    }
    for (int t = 0; t < be_threads; ++t) {
      for (int k = 0; k < Ne * ndof_per_elem; ++k) res.R[k] += Rpriv[t][k];
      for (int e = 0; e < Ne; ++e) res.sdl[e] += sdlpriv[t][e];
    }
  } else {
    for (int i = 0; i < (int)mesh.BE.size(); ++i) {
      accumulateBoundaryFace(i, res.R, res.sdl);
    }
  }

  return res;
}

void FiniteVolumeSolver::solveSteady(int itercap) {
  int Ne = mesh.E.size();
  std::cout << "Beginning DG solver loop for " << itercap << " iterations (p=" 
            << p_order << ")..." << std::endl;

  last_steady_converged = false;
  last_steady_failed_nonphysical = false;
  last_steady_hit_itercap = false;

  double baseline_residual = steady_baseline_residual_override;
  double target_residual = (baseline_residual > 0.0)
                               ? baseline_residual * 1.0e-5
                               : -1.0;
  bool printed_baseline = false;

  for (int niter = 0; niter < itercap; ++niter) {
    // STEP 4: Compute R(U) once — used for both convergence monitoring and
    //         time advancement (Step 5).  This is Stage 1 of SSP-RK2.
    ResidualResult res = calcResidualDG(U_dg, 0.0, false);

    // Compute residual norm
    double Rnorm = 0;
    for (const auto &r : res.R) {
      Rnorm += std::abs(r[0]) + std::abs(r[1]) + std::abs(r[2]) + std::abs(r[3]);
    }

    if (baseline_residual < 0.0) {
      baseline_residual = Rnorm;
      target_residual = baseline_residual * 1.0e-5;
    }
    if (!printed_baseline) {
      std::cout << "Baseline residual: " << std::scientific << std::setprecision(6)
                << baseline_residual
                << " | Target residual (1e-5 * baseline): " << target_residual;
      if (steady_baseline_residual_override > 0.0) {
        std::cout << " | Restart initial residual: " << Rnorm;
      }
      std::cout
                << std::endl;
      printed_baseline = true;
    }
    res_history.push_back(Rnorm);

    if (niter % 1000 == 0 || Rnorm < target_residual) {
      double minRho = 1e10, minP = 1e10;
      cell_residuals.resize(Ne);
      
      // For residual monitoring, check the cell-average (DOF 0 for p=0, or average for p>0)
      for (int i = 0; i < Ne; ++i) {
        // Cell residual: sum over all DOFs in the element
        double cell_res = 0;
        for (int j = 0; j < ndof_per_elem; ++j) {
          const auto &r = res.R[i * ndof_per_elem + j];
          cell_res += std::abs(r[0]) + std::abs(r[1]) + std::abs(r[2]) + std::abs(r[3]);
        }
        cell_residuals[i] = cell_res;

        // Check physical state using the true cell average (not DOF 0)
        State s(cellAverage(U_dg[i]), gamma);
        minRho = std::min(minRho, s.rho());
        minP = std::min(minP, s.p());
      }
      
      std::cout << "Iter: " << std::setw(6) << niter
                << " | Residual: " << std::scientific << std::setprecision(6)
                << Rnorm
                << " | RelRes: " << (baseline_residual > 0.0 ? Rnorm / baseline_residual : 0.0)
                << " | Min Rho: " << minRho << " | Min P: " << minP
                << std::endl;
    }

    if (Rnorm < target_residual) {
      last_steady_converged = true;
      std::cout << "Converged in " << niter << " iterations." << std::endl;
      break;
    }

    // STEP 5: Advance in time using RK4, reusing the residual already
    //         computed above as Stage 1 (avoids a redundant calcResidualDG call).
    U_dg = rk4_DG(U_dg, 0.0, false, &res);

    // Check for non-physical states
    if (!isPhysicalDG(U_dg)) {
      last_steady_failed_nonphysical = true;
      std::cerr << "Non-physical state detected at iteration " << niter
                << std::endl;
      break;
    }
  }

  if (!last_steady_converged && !last_steady_failed_nonphysical) {
    last_steady_hit_itercap = true;
  }
  
  // After solving, update U (for output compatibility) with true cell averages
  U.resize(Ne);
  for (int e = 0; e < Ne; ++e) {
    U[e] = cellAverage(U_dg[e]);  // Correct: weighted mean over all DOFs
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

// Overload: time-accurate unsteady SSP-RK2 with a single global dt for all cells
std::vector<Vec4> FiniteVolumeSolver::sspRK2(const std::vector<Vec4> &Un,
                                             double dt_global,
                                             bool secondOrder, bool limited,
                                             double time,
                                             bool use_unsteady_wake) {
  int Ne = Un.size();

  // Stage 1: U1 = Un - dt * L(Un)
  ResidualResult res1 =
      secondOrder ? calcResidualSecondOrder(Un, limited, time, use_unsteady_wake)
                  : calcResidual(Un, time, use_unsteady_wake);
  std::vector<Vec4> U1(Ne);
  for (int i = 0; i < Ne; ++i) {
    U1[i] = Un[i] - (res1.R[i] / mesh.areas[i]) * dt_global;
  }

  // Stage 2: Unp1 = 0.5*Un + 0.5*(U1 - dt * L(U1))
  ResidualResult res2 =
      secondOrder ? calcResidualSecondOrder(U1, limited, time, use_unsteady_wake)
                  : calcResidual(U1, time, use_unsteady_wake);
  std::vector<Vec4> Unp1(Ne);
  for (int i = 0; i < Ne; ++i) {
    Unp1[i] = Un[i] * 0.5 + (U1[i] - (res2.R[i] / mesh.areas[i]) * dt_global) * 0.5;
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
    double dx = std::sqrt(mesh.areas[i]);
    double eps2 = std::pow(5.0 * dx, 3.0); // Venkatakrishnan parameter K=5.0

    for (int v = 0; v < 3; ++v) {
      Vec2 r = mesh.V[mesh.E[i].v[v]] - mesh.centroids[i];
      Vec4 dU = gradX[i] * r.x + gradY[i] * r.y;
      for (int k = 0; k < 4; ++k) {
        if (std::abs(dU[k]) > 1e-12) {
          double D1 = (dU[k] > 0) ? (Umax[k] - Un[i][k]) : (Umin[k] - Un[i][k]);
          double D2 = dU[k];
          double num = D1 * D1 + eps2 + 2.0 * D1 * D2;
          double den = D1 * D1 + 2.0 * D2 * D2 + D1 * D2 + eps2;
          phi[k] = std::min(phi[k], num / den);
        }
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

void FiniteVolumeSolver::solveUnsteady(int itercap, double t_end) {
  int Ne = mesh.E.size();
  std::cout << "Beginning unsteady DG solver loop for " << itercap 
            << " iterations (p=" << p_order << ")..." << std::endl;
  
  // For unsteady, use a global time step (smallest over all cells)
  current_time = 0.0;
  std::vector<std::vector<Vec4>> U_prev_dg = U_dg;  // Store previous solution
  
  // Snapshot saving parameters
  int snapshot_interval = 100;  // Save every N iterations
  int snapshot_count = 0;

  // Create snapshot directory if it doesn't exist
  std::filesystem::create_directories(unsteady_output_dir);
  
  for (int niter = 0; niter < itercap; ++niter) {
    // STEP 4: Compute R(U) once — used for dt sizing and as Stage 1 of SSP-RK2.
    ResidualResult res_sdl = calcResidualDG(U_dg, current_time, true);

    // Global time step: minimum CFL-limited dt over all cells, with DG spectral radius scaling
    double cfl_dg = CFL * (2.0 / mass_spectral_radius);
    double dt_global = 1e10;
    for (int i = 0; i < Ne; ++i) {
      double sdl = std::max(res_sdl.sdl[i], 1e-12);
      double dt_local = cfl_dg * 2.0 * mesh.areas[i] / sdl;
      dt_global = std::min(dt_global, dt_local);
    }

    // STEP 5: Advance in time using RK4 with the residual already computed
    //         above reused as Stage 1 (avoids a redundant calcResidualDG call).
    std::vector<std::vector<Vec4>> U_new_dg = rk4_DG(U_dg, dt_global, current_time, true, &res_sdl);

    U_dg = U_new_dg;
    
    // Advance physical time
    current_time += dt_global;
    
    // Update U for compatibility (true cell averages)
    for (int e = 0; e < Ne; ++e) {
      U[e] = cellAverage(U_dg[e]);
    }
    
    bool save_allowed = (current_time >= unsteady_save_after);

    // Stop if t_end reached
    if (t_end > 0.0 && current_time >= t_end) {
      if (save_allowed) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/results_%.6f_%04d.bin",
                 unsteady_output_dir.c_str(), current_time, snapshot_count);
        saveSnapshot(filename);
        std::string dg_filename(filename);
        dg_filename.replace(dg_filename.size() - 4, 4, "_dg.bin");
        saveDGSnapshot(dg_filename);
        std::cout << "Saved snapshot: " << filename << std::endl;
        std::cout << "Saved DG snapshot: " << dg_filename << std::endl;
      }
      std::cout << "Reached t_end = " << t_end << " at iteration " << niter << std::endl;
      break;
    }
    
    // Calculate residual norm (time derivative magnitude)
    double Rnorm = 0;
    for (const auto &r : res_sdl.R) {
      Rnorm += std::abs(r[0]) + std::abs(r[1]) + std::abs(r[2]) + std::abs(r[3]);
    }
    res_history.push_back(Rnorm);
    
    // Calculate solution change from previous time step
    double dU_norm = 0;
    for (int i = 0; i < Ne; ++i) {
      for (int j = 0; j < ndof_per_elem; ++j) {
        Vec4 dU = U_dg[i][j] - U_prev_dg[i][j];
        dU_norm += std::abs(dU[0]) + std::abs(dU[1]) + std::abs(dU[2]) + std::abs(dU[3]);
      }
    }

    // Save periodic snapshots
    if (save_allowed && niter % snapshot_interval == 0) {
      char filename[256];
      snprintf(filename, sizeof(filename), "%s/results_%.6f_%04d.bin",
               unsteady_output_dir.c_str(), current_time, snapshot_count);
      saveSnapshot(filename);
      std::string dg_filename(filename);
      dg_filename.replace(dg_filename.size() - 4, 4, "_dg.bin");
      saveDGSnapshot(dg_filename);
      snapshot_count++;
      
      std::cout << "Iter: " << std::setw(6) << niter
                << " | Time: " << std::fixed << std::setprecision(6) << current_time
                << " | dt: " << std::scientific << dt_global
                << " | Residual: " << Rnorm
                << " | dU: " << dU_norm
                << " | Saved snapshot: " << filename
                << " | Saved DG snapshot: " << dg_filename
                << std::endl;
    }

    // Check if solution has reached periodic steady state
    if (dU_norm < rtol) {
      std::cout << "Reached steady state (solution stopped changing) at iteration " 
                << niter << std::endl;
      break;
    }
    
    // Check for NaN/Inf
    if (!isPhysicalDG(U_dg)) {
      std::cerr << "Non-physical state detected at iteration " << niter
                << std::endl;
      break;
    }
    
    // Store current solution for next iteration comparison
    U_prev_dg = U_dg;
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

void FiniteVolumeSolver::saveDGSnapshot(const std::string &filename) {
  std::ofstream out(filename, std::ios::binary);
  int Ne = U_dg.size();
  out.write((char *)&Ne, sizeof(int));
  out.write((char *)&p_order, sizeof(int));
  out.write((char *)&ndof_per_elem, sizeof(int));
  for (const auto &elem_dofs : U_dg) {
    for (const auto &u : elem_dofs) {
      out.write((char *)u.v, sizeof(double) * 4);
    }
  }
  out.close();
}

// ═══════════════════════════════════════════════════════════════
// DG TIME STEPPING FUNCTIONS
// ═══════════════════════════════════════════════════════════════

// Helper: apply M^{-1} * R for one element, return the result vector
static void applyMassInv(const std::vector<std::vector<double>> &Minv,
                         const Vec4 *R_elem, int ndof, double area_scale,
                         Vec4 *out) {
  for (int i = 0; i < ndof; ++i) {
    Vec4 rhs = {0,0,0,0};
    for (int j = 0; j < ndof; ++j)
      rhs += R_elem[j] * (Minv[i][j] / area_scale);
    out[i] = rhs;
  }
}

// DG RK4 with local time stepping (for steady state)
std::vector<std::vector<Vec4>> FiniteVolumeSolver::rk4_DG(
    const std::vector<std::vector<Vec4>> &Un_dg,
    double time,
    bool use_unsteady_wake,
    const ResidualResult *res1_precomputed) {
  
  int Ne = Un_dg.size();
  
  // CFL scaling: for nodal DG with mass-matrix lumping, the spectral radius of
  // M_ref^{-1} governs the max stable CFL.  Use 2/spectral_radius(M_ref^{-1}).
  // For p=0: sr=2, scale=1.0.  p=1: sr=24, scale=1/12.  p=2: sr≈96, scale≈1/48.
  double cfl_eff = CFL * (2.0 / mass_spectral_radius);

  // --- Stage 1: k1 = M^{-1} R(Un) ---
  ResidualResult res1 = res1_precomputed ? *res1_precomputed
                                         : calcResidualDG(Un_dg, time, use_unsteady_wake);

  // Compute local dt for each element (used for all stages)
  std::vector<double> dt_elem(Ne);
  // Per-element CFL sizing is independent across elements.
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e) {
    double sdl = std::max(res1.sdl[e], 1e-12);
    dt_elem[e] = cfl_eff * 2.0 * mesh.areas[e] / sdl;
  }

  // k1 = M^{-1} R(Un)
  std::vector<std::vector<Vec4>> k1(Ne, std::vector<Vec4>(ndof_per_elem));
  // Mass-matrix application is element-local and writes only into k1[e].
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e) {
    double area_scale = mesh.areas[e] / 0.5;
    applyMassInv(MassMatrixInv, &res1.R[e * ndof_per_elem], ndof_per_elem, area_scale, k1[e].data());
  }

  // --- Stage 2: k2 = M^{-1} R(Un - 0.5*dt*k1) ---
  std::vector<std::vector<Vec4>> Utmp(Ne, std::vector<Vec4>(ndof_per_elem));
  // Build the stage-2 state independently for each element.
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e)
    for (int i = 0; i < ndof_per_elem; ++i)
      Utmp[e][i] = Un_dg[e][i] - k1[e][i] * (0.5 * dt_elem[e]);

  ResidualResult res2 = calcResidualDG(Utmp, time, use_unsteady_wake);
  std::vector<std::vector<Vec4>> k2(Ne, std::vector<Vec4>(ndof_per_elem));
  // Mass-matrix application is element-local and writes only into k2[e].
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e) {
    double area_scale = mesh.areas[e] / 0.5;
    applyMassInv(MassMatrixInv, &res2.R[e * ndof_per_elem], ndof_per_elem, area_scale, k2[e].data());
  }

  // --- Stage 3: k3 = M^{-1} R(Un - 0.5*dt*k2) ---
  // Build the stage-3 state independently for each element.
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e)
    for (int i = 0; i < ndof_per_elem; ++i)
      Utmp[e][i] = Un_dg[e][i] - k2[e][i] * (0.5 * dt_elem[e]);

  ResidualResult res3 = calcResidualDG(Utmp, time, use_unsteady_wake);
  std::vector<std::vector<Vec4>> k3(Ne, std::vector<Vec4>(ndof_per_elem));
  // Mass-matrix application is element-local and writes only into k3[e].
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e) {
    double area_scale = mesh.areas[e] / 0.5;
    applyMassInv(MassMatrixInv, &res3.R[e * ndof_per_elem], ndof_per_elem, area_scale, k3[e].data());
  }

  // --- Stage 4: k4 = M^{-1} R(Un - dt*k3) ---
  // Build the stage-4 state independently for each element.
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e)
    for (int i = 0; i < ndof_per_elem; ++i)
      Utmp[e][i] = Un_dg[e][i] - k3[e][i] * dt_elem[e];

  ResidualResult res4 = calcResidualDG(Utmp, time, use_unsteady_wake);
  std::vector<std::vector<Vec4>> k4(Ne, std::vector<Vec4>(ndof_per_elem));
  // Mass-matrix application is element-local and writes only into k4[e].
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e) {
    double area_scale = mesh.areas[e] / 0.5;
    applyMassInv(MassMatrixInv, &res4.R[e * ndof_per_elem], ndof_per_elem, area_scale, k4[e].data());
  }

  // --- Update: Un+1 = Un - dt/6 * (k1 + 2*k2 + 2*k3 + k4) ---
  std::vector<std::vector<Vec4>> Unp1(Ne, std::vector<Vec4>(ndof_per_elem));
  // Final RK update is element-local.
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e)
    for (int i = 0; i < ndof_per_elem; ++i)
      Unp1[e][i] = Un_dg[e][i] - (k1[e][i] + k2[e][i]*2.0 + k3[e][i]*2.0 + k4[e][i]) * (dt_elem[e] / 6.0);

  return Unp1;
}

// DG RK4 with global time step (for time-accurate unsteady)
std::vector<std::vector<Vec4>> FiniteVolumeSolver::rk4_DG(
    const std::vector<std::vector<Vec4>> &Un_dg,
    double dt_global,
    double time,
    bool use_unsteady_wake,
    const ResidualResult *res1_precomputed) {
  
  int Ne = Un_dg.size();
  
  // --- Stage 1: k1 = M^{-1} R(Un) ---
  ResidualResult res1 = res1_precomputed ? *res1_precomputed
                                         : calcResidualDG(Un_dg, time, use_unsteady_wake);

  std::vector<std::vector<Vec4>> k1(Ne, std::vector<Vec4>(ndof_per_elem));
  // Mass-matrix application is element-local and writes only into k1[e].
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e) {
    double area_scale = mesh.areas[e] / 0.5;
    applyMassInv(MassMatrixInv, &res1.R[e * ndof_per_elem], ndof_per_elem, area_scale, k1[e].data());
  }

  // --- Stage 2: k2 = M^{-1} R(Un - 0.5*dt*k1) ---
  std::vector<std::vector<Vec4>> Utmp(Ne, std::vector<Vec4>(ndof_per_elem));
  // Build the stage-2 state independently for each element.
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e)
    for (int i = 0; i < ndof_per_elem; ++i)
      Utmp[e][i] = Un_dg[e][i] - k1[e][i] * (0.5 * dt_global);

  ResidualResult res2 = calcResidualDG(Utmp, time, use_unsteady_wake);
  std::vector<std::vector<Vec4>> k2(Ne, std::vector<Vec4>(ndof_per_elem));
  // Mass-matrix application is element-local and writes only into k2[e].
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e) {
    double area_scale = mesh.areas[e] / 0.5;
    applyMassInv(MassMatrixInv, &res2.R[e * ndof_per_elem], ndof_per_elem, area_scale, k2[e].data());
  }

  // --- Stage 3: k3 = M^{-1} R(Un - 0.5*dt*k2) ---
  // Build the stage-3 state independently for each element.
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e)
    for (int i = 0; i < ndof_per_elem; ++i)
      Utmp[e][i] = Un_dg[e][i] - k2[e][i] * (0.5 * dt_global);

  ResidualResult res3 = calcResidualDG(Utmp, time, use_unsteady_wake);
  std::vector<std::vector<Vec4>> k3(Ne, std::vector<Vec4>(ndof_per_elem));
  // Mass-matrix application is element-local and writes only into k3[e].
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e) {
    double area_scale = mesh.areas[e] / 0.5;
    applyMassInv(MassMatrixInv, &res3.R[e * ndof_per_elem], ndof_per_elem, area_scale, k3[e].data());
  }

  // --- Stage 4: k4 = M^{-1} R(Un - dt*k3) ---
  // Build the stage-4 state independently for each element.
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e)
    for (int i = 0; i < ndof_per_elem; ++i)
      Utmp[e][i] = Un_dg[e][i] - k3[e][i] * dt_global;

  ResidualResult res4 = calcResidualDG(Utmp, time, use_unsteady_wake);
  std::vector<std::vector<Vec4>> k4(Ne, std::vector<Vec4>(ndof_per_elem));
  // Mass-matrix application is element-local and writes only into k4[e].
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e) {
    double area_scale = mesh.areas[e] / 0.5;
    applyMassInv(MassMatrixInv, &res4.R[e * ndof_per_elem], ndof_per_elem, area_scale, k4[e].data());
  }

  // --- Update: Un+1 = Un - dt/6 * (k1 + 2*k2 + 2*k3 + k4) ---
  std::vector<std::vector<Vec4>> Unp1(Ne, std::vector<Vec4>(ndof_per_elem));
  // Final RK update is element-local.
#pragma omp parallel for if (Ne > 64)
  for (int e = 0; e < Ne; ++e)
    for (int i = 0; i < ndof_per_elem; ++i)
      Unp1[e][i] = Un_dg[e][i] - (k1[e][i] + k2[e][i]*2.0 + k3[e][i]*2.0 + k4[e][i]) * (dt_global / 6.0);

  return Unp1;
}

// Check if DG solution is physical
bool FiniteVolumeSolver::isPhysicalDG(const std::vector<std::vector<Vec4>> &Un_dg) {
  for (const auto &u_elem : Un_dg) {
    State s(cellAverage(u_elem), gamma);
    if (!(s.rho() > 0) || !(s.p() > 0))   // catches NaN as well
      return false;
  }
  return true;
}
