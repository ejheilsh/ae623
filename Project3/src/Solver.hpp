#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "Fluxes.hpp"
#include "Mesh.hpp"
#include <string>
#include <vector>

class FiniteVolumeSolver {
public:
  Mesh mesh;
  // (DG): Multi-DOF representation
  int p_order = 0;                              // polynomial order (0,1,2,3)
  int ndof_per_elem;                            // (p+1)*(p+2)/2
  std::vector<std::vector<Vec4>> U_dg;          // [nelem][ndof][4]
  std::vector<std::vector<Vec4>> U0_dg;         // for baseline
  
  // Keep old U for compatibility from the VF (p=0 case can use it)
  std::vector<Vec4> U;     
  std::vector<Vec4> U0;

  std::vector<double> res_history;
  std::vector<double> cell_residuals;
  std::string fluxname = "roe";

  double gamma = 1.4;
  double CFL = 1.0;
  double rtol = 1e-6;
  double rho0 = 1.0;
  double a0 = 1.0;
  double alpha = 50.0;
  double Minf = 0.1;
  double p0;
  double pout;
  double current_time = 0.0;  // Track physical time for unsteady simulations

  FiniteVolumeSolver(const std::string &meshfile);

  void initializeDG();  // Call after setting p_order
  void setInitialCondition();
  void loadInitialCondition(const std::string &filename);
  void loadMappedInitialCondition(const std::string &coarse_meshfile,
                                  const std::string &coarse_statefile);
  void solveSteady(int itercap = 1000000, bool secondOrder = false,
                   bool limited = false);
  void solveUnsteady(int itercap = 1000000, bool secondOrder = false,
                     bool limited = false, double t_end = -1.0);
  
  void saveSnapshot(const std::string &filename);

private:
  struct ResidualResult {
    std::vector<Vec4> R;
    std::vector<double> sdl;
  };

  // Mass matrix storage (pre-computed per element or global)
  std::vector<std::vector<double>> MassMatrixInv;  // [ndof][ndof] - inverted mass matrix
  double mass_spectral_radius = 2.0;               // max eigenvalue of M_ref^{-1} (=2 for p=0)
  
  // Basis function helper
  void computeMassMatrix();                         // compute M^{-1} once
  std::vector<double> evaluateBasis(double xi, double eta, int p);  // φ_i(ξ,η)
  std::vector<double> evaluateBasisGrad(double xi, double eta, int p); // ∂φ_i/∂ξ, ∂φ_i/∂η
  
  // Quadrature helpers
  struct QuadRule {
    int n;                            // number of points
    std::vector<double> points;       // quadrature points on [0,1]
    std::vector<double> weights;      // quadrature weights
  };
  QuadRule getQuadratureRule(int p_order);  // Get appropriate quadrature for polynomial order p

  // Original FV residual (for backward compatibility)
  ResidualResult calcResidual(const std::vector<Vec4> &Un, double time = 0.0, 
                              bool use_unsteady_wake = false);
  
  // DG residual
  ResidualResult calcResidualDG(const std::vector<std::vector<Vec4>> &Un_dg, 
    double time = 0.0, 
    bool use_unsteady_wake = false);

  // BR2 diffusion residual (removed — not needed for inviscid Euler).

  // Weights w_j = int(phi_j dA_ref) / A_ref  for cell-average computation.
  std::vector<double> cellAvgWeights;  // length ndof_per_elem

  // Compute the L2 cell average: u_bar = sum_j w_j * U_dg[e][j]
  Vec4 cellAverage(const std::vector<Vec4> &dofs) const;

  // Venkatakrishnan DG limiter (Cockburn & Shu 1998 minmod-based projection).
  // Projects higher-order DOFs onto a limited linear reconstruction to suppress
  // spurious oscillations near shocks.  Applied after each RK stage when
  // p_order > 0.
  void applyLimiterDG(std::vector<std::vector<Vec4>> &Un_dg);
    
  ResidualResult calcResidualSecondOrder(const std::vector<Vec4> &Un,
                                         bool limited, double time = 0.0,
                                         bool use_unsteady_wake = false);

  std::vector<Vec4> applyLimiter(const std::vector<Vec4> &Un,
                                 std::vector<Vec4> &gradX,
                                 std::vector<Vec4> &gradY);
  std::vector<Vec4> sspRK2(const std::vector<Vec4> &Un, bool secondOrder,
                           bool limited, double time = 0.0,
                           bool use_unsteady_wake = false);
  // Fixed-dt overload for time-accurate unsteady (single global dt)
  std::vector<Vec4> sspRK2(const std::vector<Vec4> &Un, double dt_global,
                           bool secondOrder, bool limited, double time,
                           bool use_unsteady_wake);

  // DG time stepping functions
  // Local time-stepping (steady): accepts an optional pre-computed Stage-1 residual
  // to avoid recomputing it when the caller already has it.
  std::vector<std::vector<Vec4>> sspRK2_DG(const std::vector<std::vector<Vec4>> &Un_dg,
                                            double time = 0.0,
                                            bool use_unsteady_wake = false,
                                            const ResidualResult *res1_precomputed = nullptr);
  // Global time-stepping (unsteady): accepts an optional pre-computed Stage-1 residual.
  std::vector<std::vector<Vec4>> sspRK2_DG(const std::vector<std::vector<Vec4>> &Un_dg,
                                            double dt_global,
                                            double time,
                                            bool use_unsteady_wake,
                                            const ResidualResult *res1_precomputed = nullptr);

  std::vector<double> calcDt(const std::vector<double> &sdl);
  bool isPhysical(const std::vector<Vec4> &Un);
  bool isPhysicalDG(const std::vector<std::vector<Vec4>> &Un_dg);
};

#endif
