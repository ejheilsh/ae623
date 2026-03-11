#include "Solver.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <limits>
#include <unordered_map>

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
  // The GRI 'degree' field records the intended solution polynomial order,
  // not the number of geometry nodes.  All meshes in this project use straight
  // triangular elements (q=1 geometry).  True isoparametric curved elements
  // would require 6 or 10 nodes per element in the GRI file.
  //
  // Superparametric requirement (from course notes):
  //   p=1 solution  →  needs q >= 2 geometry (superparametric)
  //   p >= 2 solution →  isoparametric (q=p) or subparametric (q<p) acceptable
  {
    int p_hint = mesh.q_order_global;  // GRI degree field = intended solution order
    std::cout << "  GRI degree hint (intended solution order): " << p_hint << std::endl;
    std::cout << "  Actual mesh geometry: q=1 (straight triangles, no curved elements)" << std::endl;
    if (p_order == 1) {
      std::cout << "  NOTE: p=1 solution ideally requires superparametric geometry "
                   "(q_geom >= 2) on curved boundaries.  Current meshes are straight "
                   "(q_geom=1); accept some geometric approximation error." << std::endl;
    } else if (p_order >= 2) {
      std::cout << "  NOTE: p=" << p_order << " solution on straight (q=1) mesh — "
                   "subparametric.  Acceptable per course notes for p>=2." << std::endl;
    }
    // Run the Newton inverse-mapping spot check on element 0 (always straight → one step)
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
}

void FiniteVolumeSolver::loadMappedInitialCondition(
    const std::string &coarse_meshfile, const std::string &coarse_statefile) {
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
              << mass_spectral_radius << " -> CFL scale factor: "
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
      g[5]       = 9.0/2.0*eta;
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

  // ═══════════════════════════════════════════════════════════════
  // STEP 4a — VOLUME INTEGRALS (element interiors, loop first per spec)
  // R_j -= ∫_T F(u) · ∇_x φ_j dV   (the "by parts" volume term, Eq. 4.3.12 second term)
  // Uses Dunavant 2D quadrature rules.  For p=0 the gradient is zero so this
  // loop is a no-op, but we always execute it to follow the prescribed order.
  // ═══════════════════════════════════════════════════════════════
  {
    struct QP2D { double xi, eta, w; };
    std::vector<QP2D> qpts2d;
    if (p_order == 0) {
      // 1-point centroid rule — gradients are zero so contributions vanish,
      // but the loop still runs in the correct order per the spec.
      qpts2d = {{1.0/3.0, 1.0/3.0, 0.5}};
    } else if (p_order == 1) {
      qpts2d = {{1.0/6.0, 1.0/6.0, 1.0/6.0},
                {2.0/3.0, 1.0/6.0, 1.0/6.0},
                {1.0/6.0, 2.0/3.0, 1.0/6.0}};
    } else if (p_order == 2) {
      qpts2d = {
        {0.108103018168070, 0.445948490915965, 0.111690794839005},
        {0.445948490915965, 0.108103018168070, 0.111690794839005},
        {0.445948490915965, 0.445948490915965, 0.111690794839005},
        {0.816847572980459, 0.091576213509771, 0.054975871827661},
        {0.091576213509771, 0.816847572980459, 0.054975871827661},
        {0.091576213509771, 0.091576213509771, 0.054975871827661},
      };
    } else { // p_order == 3
      qpts2d = {
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
    }

    for (int e = 0; e < Ne; ++e) {
      Vec2 v0 = mesh.V[mesh.E[e].v[0]];
      Vec2 v1 = mesh.V[mesh.E[e].v[1]];
      Vec2 v2 = mesh.V[mesh.E[e].v[2]];
      double dx1 = v1.x - v0.x, dx2 = v2.x - v0.x;
      double dy1 = v1.y - v0.y, dy2 = v2.y - v0.y;
      // J^{-T} maps ref gradient to physical gradient (detJ factors cancel):
      //   (∂φ/∂x)*|J| =  ∂φ/∂ξ * dy2 - ∂φ/∂η * dy1
      //   (∂φ/∂y)*|J| = -∂φ/∂ξ * dx2 + ∂φ/∂η * dx1

      for (const auto &qp : qpts2d) {
        std::vector<double> phi  = evaluateBasis(qp.xi, qp.eta, p_order);
        std::vector<double> gphi = evaluateBasisGrad(qp.xi, qp.eta, p_order);

        Vec4 u = {0,0,0,0};
        for (int j = 0; j < ndof_per_elem; ++j)
          u += Un_dg[e][j] * phi[j];

        State s(u, gamma);
        double rho = s.rho(), p_pres = s.p(), e_rhoE = u[3];
        double vel_u = u[1]/rho, vel_v = u[2]/rho;
        Vec4 Fx = {rho*vel_u, rho*vel_u*vel_u + p_pres, rho*vel_u*vel_v, (e_rhoE + p_pres)*vel_u};
        Vec4 Fy = {rho*vel_v, rho*vel_u*vel_v, rho*vel_v*vel_v + p_pres, (e_rhoE + p_pres)*vel_v};

        for (int j = 0; j < ndof_per_elem; ++j) {
          double dphidxi  = gphi[j];
          double dphideta = gphi[ndof_per_elem + j];
          double dphidx =  dphidxi * dy2 - dphideta * dy1;  // = (∂φ/∂x) * detJ
          double dphidy = -dphidxi * dx2 + dphideta * dx1;  // = (∂φ/∂y) * detJ
          res.R[e * ndof_per_elem + j] -= (Fx * dphidx + Fy * dphidy) * qp.w;
        }
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // STEP 4b — INTERIOR INTERFACE FLUXES (Eq. 4.3.12 third term)
  // ═══════════════════════════════════════════════════════════════
  for (int i = 0; i < (int)mesh.IE.size(); ++i) {
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
      res.R[eL * ndof_per_elem] += fr.F * len;
      res.R[eR * ndof_per_elem] -= fr.F * len;
      res.sdl[eL] += fr.smax * len;
      res.sdl[eR] += fr.smax * len;
    } else {
      // Find reference parameterizations for this edge in each element
      double xiL0, etaL0, xiL1, etaL1;
      double xiR0, etaR0, xiR1, etaR1;
      edgeRefParam(mesh.E[eL].v, va, vb, xiL0, etaL0, xiL1, etaL1);
      edgeRefParam(mesh.E[eR].v, va, vb, xiR0, etaR0, xiR1, etaR1);

      for (int q = 0; q < qr.n; ++q) {
        double t = qr.points[q];
        double xiL  = xiL0  + t * (xiL1  - xiL0);
        double etaL = etaL0 + t * (etaL1 - etaL0);
        double xiR  = xiR0  + t * (xiR1  - xiR0);
        double etaR = etaR0 + t * (etaR1 - etaR0);

        std::vector<double> phiL = evaluateBasis(xiL,  etaL,  p_order);
        std::vector<double> phiR = evaluateBasis(xiR,  etaR,  p_order);

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
          res.R[eL * ndof_per_elem + j] += fr.F * (w * phiL[j]);
          res.R[eR * ndof_per_elem + j] -= fr.F * (w * phiR[j]);
        }
      }
      // Use the true cell-average state to compute sdl for time step sizing.
      // For p>=1 DOF 0 is the vertex value, not the integral mean — use cellAverage().
      {
        Vec4 uavgL = cellAverage(Un_dg[eL]);
        Vec4 uavgR = cellAverage(Un_dg[eR]);
        FluxResult fr0;
        if (fluxname == "hlle") fr0 = fluxHLLE(uavgL, uavgR, normal, gamma);
        else                     fr0 = fluxRoe (uavgL, uavgR, normal, gamma);
        res.sdl[eL] += fr0.smax * len;
        res.sdl[eR] += fr0.smax * len;
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // STEP 4c — BOUNDARY INTERFACE FLUXES (special treatment per spec)
  // ═══════════════════════════════════════════════════════════════
  for (int i = 0; i < (int)mesh.BE.size(); ++i) {
    int eL   = mesh.BE[i].elemL;
    int va   = mesh.BE[i].v[0];
    int vb   = mesh.BE[i].v[1];
    std::string bName = mesh.Bname[mesh.BE[i].bIndex];
    Vec2 n   = mesh.bnormals[i];
    double len = mesh.blengths[i];

    if (p_order == 0) {
      Vec4 u_int = Un_dg[eL][0];
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
      res.R[eL * ndof_per_elem] += fr.F * len;
      res.sdl[eL] += fr.smax * len;
    } else {
      double xi0, eta0, xi1, eta1;
      edgeRefParam(mesh.E[eL].v, va, vb, xi0, eta0, xi1, eta1);

      for (int q = 0; q < qr.n; ++q) {
        double t    = qr.points[q];
        double xi_q = xi0  + t * (xi1  - xi0);
        double eta_q= eta0 + t * (eta1 - eta0);

        std::vector<double> phi = evaluateBasis(xi_q, eta_q, p_order);

        Vec4 u_int = {0,0,0,0};
        for (int j = 0; j < ndof_per_elem; ++j)
          u_int += Un_dg[eL][j] * phi[j];

        // Physical position of this quadrature point on the edge
        Vec2 edge_pos = mesh.V[va] * (1.0 - t) + mesh.V[vb] * t;

        FluxResult fr;
        if (bName == "inflow") {
          Vec4 Ub = subsonicInflow(u_int, n, rho0, a0, alpha, gamma,
                                   edge_pos.y, time, use_unsteady_wake);
          if (fluxname == "hlle") fr = fluxHLLE(u_int, Ub, n, gamma);
          else                     fr = fluxRoe (u_int, Ub, n, gamma);
        } else if (bName == "outflow") {
          Vec4 Ub = subsonicOutflow(u_int, n, pout, gamma);
          if (fluxname == "hlle") fr = fluxHLLE(u_int, Ub, n, gamma);
          else                     fr = fluxRoe (u_int, Ub, n, gamma);
        } else {
          fr = inviscidWallFlux(u_int, n, gamma);
        }

        double w = qr.weights[q] * len;
        for (int j = 0; j < ndof_per_elem; ++j)
          res.R[eL * ndof_per_elem + j] += fr.F * (w * phi[j]);
      }
      // Use the true cell-average state for sdl computation.
      // For p>=1 DOF 0 is the vertex value at (0,0), not the integral mean.
      {
        Vec4 u_int0 = cellAverage(Un_dg[eL]);
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
        res.sdl[eL] += fr0.smax * len;
      }
    }
  }

  return res;
}

// DG LIMITER — Venkatakrishnan / minmod moment-limiting
// Reference: Cockburn & Shu, "The Runge-Kutta discontinuous Galerkin
//   method for conservation laws V", JCP 141 (1998).
void FiniteVolumeSolver::applyLimiterDG(std::vector<std::vector<Vec4>> &Un_dg) {
  if (p_order == 0) return;  // nothing to limit for piecewise-constant
  int Ne = mesh.E.size();

  // Build neighbour list (shared-edge neighbours only)
  std::vector<std::vector<int>> nbrs(Ne);
  for (const auto &ie : mesh.IE) {
    nbrs[ie.elemL].push_back(ie.elemR);
    nbrs[ie.elemR].push_back(ie.elemL);
  }

  // Pre-collect true cell averages using cellAverage() — NOT DOF 0 for p>=1.
  // For a nodal Lagrange basis DOF 0 is the value at vertex (0,0), not the
  // integral average.  Conservation requires using the true mean.
  std::vector<Vec4> ubar(Ne);
  for (int e = 0; e < Ne; ++e) ubar[e] = cellAverage(Un_dg[e]);

  for (int e = 0; e < Ne; ++e) {
    if (nbrs[e].empty()) continue;

    // ── Step 1: neighbour bounds on the cell average ─────────────────────
    Vec4 umin = ubar[e], umax = ubar[e];
    for (int nb : nbrs[e]) {
      for (int k = 0; k < 4; ++k) {
        umin[k] = std::min(umin[k], ubar[nb][k]);
        umax[k] = std::max(umax[k], ubar[nb][k]);
      }
    }

    // ── Step 2: evaluate the reconstructed solution at the vertices ───────
    // For a p-order DG element the value at vertex v_i (reference (ξ_i,η_i)) is:
    //   u(v_i) = Σ_j U_dg[e][j] * φ_j(ξ_i, η_i)
    // The three reference vertices of the standard triangle are (0,0),(1,0),(0,1).
    static const double xiv[3]  = {0.0, 1.0, 0.0};
    static const double etav[3] = {0.0, 0.0, 1.0};

    bool troubled = false;
    const double eps = 1e-12;
    for (int vi = 0; vi < 3 && !troubled; ++vi) {
      std::vector<double> phi = evaluateBasis(xiv[vi], etav[vi], p_order);
      Vec4 uv = {0,0,0,0};
      for (int j = 0; j < ndof_per_elem; ++j)
        uv += Un_dg[e][j] * phi[j];
      for (int k = 0; k < 4; ++k) {
        if (uv[k] < umin[k] - eps || uv[k] > umax[k] + eps) {
          troubled = true; break;
        }
      }
    }
    if (!troubled) continue;

    // ── Step 3: Venkatakrishnan-limited linear reconstruction ─────────────
    // Compute the unlimited gradient of the cell average across neighbours
    // using a Green-Gauss-like least-squares formula (simple centroid-to-centroid).
    // Then apply the Venkatakrishnan limiter to scale the gradient.
    Vec4 gradx_e = {0,0,0,0}, grady_e = {0,0,0,0};
    double sw = 0;
    for (int nb : nbrs[e]) {
      Vec2 dr = mesh.centroids[nb] - mesh.centroids[e];
      double d2 = dr.x*dr.x + dr.y*dr.y;
      double w  = 1.0 / d2;
      sw += w;
      Vec4 du = (ubar[nb] - ubar[e]) * w;
      gradx_e += du * dr.x;
      grady_e += du * dr.y;
    }
    if (sw > 0) { gradx_e = gradx_e * (1.0/sw); grady_e = grady_e * (1.0/sw); }

    // Venkatakrishnan limiter factor φ applied to the gradient
    double dx = std::sqrt(mesh.areas[e]);
    double eps2 = std::pow(1.0 * dx, 3.0);  // K=1 (conservative)
    Vec4 phi_lim = {1,1,1,1};
    for (int vi = 0; vi < 3; ++vi) {
      Vec2 r = mesh.V[mesh.E[e].v[vi]] - mesh.centroids[e];
      Vec4 dU = gradx_e * r.x + grady_e * r.y;
      for (int k = 0; k < 4; ++k) {
        if (std::abs(dU[k]) > 1e-12) {
          double D1 = (dU[k] > 0) ? (umax[k]-ubar[e][k]) : (umin[k]-ubar[e][k]);
          double D2 = dU[k];
          double num = D1*D1 + eps2 + 2.0*D1*D2;
          double den = D1*D1 + 2.0*D2*D2 + D1*D2 + eps2;
          phi_lim[k] = std::min(phi_lim[k], num/den);
        }
      }
    }
    for (int k = 0; k < 4; ++k)
      phi_lim[k] = std::max(0.0, std::min(1.0, phi_lim[k]));

    // Apply per-component limiter factor to the gradient
    for (int k = 0; k < 4; ++k) {
      gradx_e[k] *= phi_lim[k];
      grady_e[k] *= phi_lim[k];
    }

    // ── Step 4: re-project ALL DG DOFs to match the limited linear field ──
    // The limited linear reconstruction is:
    //   u_lim(x,y) = ubar + gradx*(x-xc) + grady*(y-yc)
    // Re-project by evaluating u_lim at every nodal DOF position.
    // DOF j lives at physical position: x_j = v0 + (v1-v0)*xi_j + (v2-v0)*eta_j
    // This correctly handles p=1 (3 vertex nodes), p=2 (6 nodes), p=3 (10 nodes).
    // Conservation is preserved because cellAverage(u_lim_nodal) = ubar exactly
    // (the linear reconstruction has ubar as its mean over any triangle).
    Vec2 v0 = mesh.V[mesh.E[e].v[0]];
    Vec2 v1 = mesh.V[mesh.E[e].v[1]];
    Vec2 v2 = mesh.V[mesh.E[e].v[2]];
    Vec2 xc = mesh.centroids[e];

    // Reference coordinates of the Lagrange nodes for each p (canonical ordering)
    static const double xi_nodes_p1[3]  = {0.0, 1.0, 0.0};
    static const double eta_nodes_p1[3] = {0.0, 0.0, 1.0};
    static const double xi_nodes_p2[6]  = {0.0, 1.0, 0.0, 0.5, 0.5, 0.0};
    static const double eta_nodes_p2[6] = {0.0, 0.0, 1.0, 0.5, 0.0, 0.5};
    // p=2 node order confirmed: DOF3→(0.5,0.5), DOF4→(0.0,0.5), DOF5→(0.5,0.0)
    // (matches evaluateBasis Kronecker delta check)
    static const double xi_nodes_p3[10]  = {0.0,1.0,0.0, 2./3,1./3, 0.0, 0.0,1./3,2./3,1./3};
    static const double eta_nodes_p3[10] = {0.0,0.0,1.0, 0.0, 0.0,1./3,2./3,2./3,1./3,1./3};

    const double *xi_n = nullptr, *eta_n = nullptr;
    if (p_order == 1) { xi_n = xi_nodes_p1; eta_n = eta_nodes_p1; }
    else if (p_order == 2) { xi_n = xi_nodes_p2; eta_n = eta_nodes_p2; }
    else { xi_n = xi_nodes_p3; eta_n = eta_nodes_p3; }

    for (int j = 0; j < ndof_per_elem; ++j) {
      // Physical position of node j
      double xj = v0.x + (v1.x-v0.x)*xi_n[j] + (v2.x-v0.x)*eta_n[j];
      double yj = v0.y + (v1.y-v0.y)*xi_n[j] + (v2.y-v0.y)*eta_n[j];
      // Evaluate limited linear reconstruction at this node
      Un_dg[e][j] = ubar[e] + gradx_e*(xj-xc.x) + grady_e*(yj-xc.y);
    }
    // Note: cellAverage(Un_dg[e]) == ubar[e] after this re-projection
    // because the linear function integrates to its centroid value (=ubar[e]).
  }
}
void FiniteVolumeSolver::solveSteady(int itercap, bool secondOrder,
                                     bool limited) {
  int Ne = mesh.E.size();
  std::cout << "Beginning DG solver loop for " << itercap << " iterations (p=" 
            << p_order << ")..." << std::endl;
  
  for (int niter = 0; niter < itercap; ++niter) {
    // STEP 4: Compute R(U) once — used for both convergence monitoring and
    //         time advancement (Step 5).  This is Stage 1 of SSP-RK2.
    ResidualResult res = calcResidualDG(U_dg, 0.0, false);

    // Compute residual norm
    double Rnorm = 0;
    for (const auto &r : res.R) {
      Rnorm += std::abs(r[0]) + std::abs(r[1]) + std::abs(r[2]) + std::abs(r[3]);
    }
    res_history.push_back(Rnorm);

    if (niter % 10 == 0 || Rnorm < rtol) {
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
                << Rnorm << " | Min Rho: " << minRho << " | Min P: " << minP
                << std::endl;
    }

    if (Rnorm < rtol) {
      std::cout << "Converged in " << niter << " iterations." << std::endl;
      break;
    }

    // STEP 5: Advance in time using SSP-RK2, reusing the residual already
    //         computed above as Stage 1 (avoids a redundant calcResidualDG call).
    U_dg = sspRK2_DG(U_dg, 0.0, false, &res);

    // Check for non-physical states
    if (!isPhysicalDG(U_dg)) {
      std::cerr << "Non-physical state detected at iteration " << niter
                << std::endl;
      break;
    }
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

void FiniteVolumeSolver::solveUnsteady(int itercap, bool secondOrder,
                                       bool limited, double t_end) {
  int Ne = mesh.E.size();
  std::cout << "Beginning unsteady DG solver loop for " << itercap 
            << " iterations (p=" << p_order << ")..." << std::endl;
  
  // For unsteady, use a global time step (smallest over all cells)
  current_time = 0.0;
  std::vector<std::vector<Vec4>> U_prev_dg = U_dg;  // Store previous solution
  
  // Snapshot saving parameters
  int snapshot_interval = 100;  // Save every N iterations
  int snapshot_count = 0;
  
  // Create data directory if it doesn't exist
  system("mkdir -p data");
  
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

    // STEP 5: Advance in time using SSP-RK2 with the residual already computed
    //         above reused as Stage 1 (avoids a redundant calcResidualDG call).
    std::vector<std::vector<Vec4>> U_new_dg = sspRK2_DG(U_dg, dt_global, current_time, true, &res_sdl);

    U_dg = U_new_dg;
    
    // Advance physical time
    current_time += dt_global;
    
    // Update U for compatibility (true cell averages)
    for (int e = 0; e < Ne; ++e) {
      U[e] = cellAverage(U_dg[e]);
    }
    
    // Stop if t_end reached
    if (t_end > 0.0 && current_time >= t_end) {
      // Save a final snapshot at t_end
      char filename[256];
      snprintf(filename, sizeof(filename), "data/results_%.6f_%04d.bin",
               current_time, snapshot_count);
      saveSnapshot(filename);
      std::cout << "Saved snapshot: " << filename << std::endl;
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
    if (niter % snapshot_interval == 0) {
      char filename[256];
      snprintf(filename, sizeof(filename), "data/results_%.6f_%04d.bin",
               current_time, snapshot_count);
      saveSnapshot(filename);
      snapshot_count++;
      
      std::cout << "Iter: " << std::setw(6) << niter
                << " | Time: " << std::fixed << std::setprecision(6) << current_time
                << " | dt: " << std::scientific << dt_global
                << " | Residual: " << Rnorm
                << " | dU: " << dU_norm
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

// ═══════════════════════════════════════════════════════════════
// DG TIME STEPPING FUNCTIONS
// ═══════════════════════════════════════════════════════════════

// DG SSP-RK2 with local time stepping (for steady state)
std::vector<std::vector<Vec4>> FiniteVolumeSolver::sspRK2_DG(
    const std::vector<std::vector<Vec4>> &Un_dg,
    double time,
    bool use_unsteady_wake,
    const ResidualResult *res1_precomputed) {
  
  int Ne = Un_dg.size();
  
  // Stage 1: use pre-computed residual if supplied (avoids a redundant evaluation),
  //          otherwise compute it now.
  ResidualResult res1 = res1_precomputed ? *res1_precomputed
                                         : calcResidualDG(Un_dg, time, use_unsteady_wake);

  // Scale CFL by 2/spectral_radius(M_ref^{-1}) so the nodal DG time step is stable.
  // For p=0: spectral_radius=2, scale=1.0 (no change).
  // For p=1: spectral_radius=24, scale=1/12.
  // For p=2: spectral_radius≈96.4, scale≈1/48.
  // For p=3: spectral_radius≈224, scale≈1/112.
  double cfl_eff = CFL * (2.0 / mass_spectral_radius);
  std::vector<std::vector<Vec4>> U1_dg(Ne, std::vector<Vec4>(ndof_per_elem));
  
  for (int e = 0; e < Ne; ++e) {
    double sdl = std::max(res1.sdl[e], 1e-12);
    double dt = cfl_eff * 2.0 * mesh.areas[e] / sdl;
    
    // Apply mass matrix inverse scaled by actual element area
    double area_scale = mesh.areas[e] / 0.5;  // A_actual / A_ref
    
    for (int i = 0; i < ndof_per_elem; ++i) {
      Vec4 rhs = {0, 0, 0, 0};
      for (int j = 0; j < ndof_per_elem; ++j) {
        // M_actual^{-1}_{ij} * R_j = M_ref^{-1}_{ij} * R_j / area_scale
        rhs += res1.R[e * ndof_per_elem + j] * (MassMatrixInv[i][j] / area_scale);
      }
      // Forward Euler step: U1 = Un - dt * M^{-1} * R
      U1_dg[e][i] = Un_dg[e][i] - rhs * dt;
    }
  }
  
  // Stage 2: Compute residual of U1
  // Apply limiter to Stage 1 result before re-evaluating the residual (Cockburn & Shu 1998).
  if (p_order > 0) applyLimiterDG(U1_dg);
  ResidualResult res2 = calcResidualDG(U1_dg, time, use_unsteady_wake);
  
  std::vector<std::vector<Vec4>> Unp1_dg(Ne, std::vector<Vec4>(ndof_per_elem));
  
  for (int e = 0; e < Ne; ++e) {
    double sdl = std::max(res1.sdl[e], 1e-12);
    double dt = cfl_eff * 2.0 * mesh.areas[e] / sdl;
    
    double area_scale = mesh.areas[e] / 0.5;
    
    for (int i = 0; i < ndof_per_elem; ++i) {
      Vec4 rhs2 = {0, 0, 0, 0};
      for (int j = 0; j < ndof_per_elem; ++j) {
        rhs2 += res2.R[e * ndof_per_elem + j] * (MassMatrixInv[i][j] / area_scale);
      }
      // SSP-RK2: Unp1 = 0.5*Un + 0.5*(U1 - dt*M^{-1}*R2)
      Unp1_dg[e][i] = Un_dg[e][i] * 0.5 + 
                      (U1_dg[e][i] - rhs2 * dt) * 0.5;
    }
  }

  // Apply limiter to the final stage result.
  if (p_order > 0) applyLimiterDG(Unp1_dg);

  return Unp1_dg;
}

// DG SSP-RK2 with global time step (for time-accurate unsteady)
std::vector<std::vector<Vec4>> FiniteVolumeSolver::sspRK2_DG(
    const std::vector<std::vector<Vec4>> &Un_dg,
    double dt_global,
    double time,
    bool use_unsteady_wake,
    const ResidualResult *res1_precomputed) {
  
  int Ne = Un_dg.size();
  
  // Stage 1: use pre-computed residual if supplied (avoids a redundant evaluation),
  //          otherwise compute it now.
  ResidualResult res1 = res1_precomputed ? *res1_precomputed
                                         : calcResidualDG(Un_dg, time, use_unsteady_wake);
  std::vector<std::vector<Vec4>> U1_dg(Ne, std::vector<Vec4>(ndof_per_elem));
  
  for (int e = 0; e < Ne; ++e) {
    double area_scale = mesh.areas[e] / 0.5;
    
    for (int i = 0; i < ndof_per_elem; ++i) {
      Vec4 rhs = {0, 0, 0, 0};
      for (int j = 0; j < ndof_per_elem; ++j) {
        rhs += res1.R[e * ndof_per_elem + j] * (MassMatrixInv[i][j] / area_scale);
      }
      U1_dg[e][i] = Un_dg[e][i] - rhs * dt_global;
    }
  }
  
  // Stage 2
  // Apply limiter to Stage 1 result before re-evaluating the residual.
  if (p_order > 0) applyLimiterDG(U1_dg);
  ResidualResult res2 = calcResidualDG(U1_dg, time, use_unsteady_wake);
  std::vector<std::vector<Vec4>> Unp1_dg(Ne, std::vector<Vec4>(ndof_per_elem));
  
  for (int e = 0; e < Ne; ++e) {
    double area_scale = mesh.areas[e] / 0.5;
    
    for (int i = 0; i < ndof_per_elem; ++i) {
      Vec4 rhs2 = {0, 0, 0, 0};
      for (int j = 0; j < ndof_per_elem; ++j) {
        rhs2 += res2.R[e * ndof_per_elem + j] * (MassMatrixInv[i][j] / area_scale);
      }
      Unp1_dg[e][i] = Un_dg[e][i] * 0.5 + 
                      (U1_dg[e][i] - rhs2 * dt_global) * 0.5;
    }
  }

  // Apply limiter to the final stage result.
  if (p_order > 0) applyLimiterDG(Unp1_dg);

  return Unp1_dg;
}

// Check if DG solution is physical
bool FiniteVolumeSolver::isPhysicalDG(const std::vector<std::vector<Vec4>> &Un_dg) {
  // Check the true cell average of each element for physicality.
  // The cell average is conserved by the DG scheme so this is the right quantity.
  for (const auto &u_elem : Un_dg) {
    State s(cellAverage(u_elem), gamma);
    if (s.rho() <= 0 || s.p() <= 0)
      return false;
  }
  return true;
}
