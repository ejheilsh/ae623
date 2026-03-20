#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "Fluxes.hpp"
#include "Mesh.hpp"
#include <string>
#include <vector>

class FiniteVolumeSolver {
public:
  Mesh mesh;
  int p_order = 0;                              // polynomial order (0,1,2,3)
  int ndof_per_elem;                            // (p+1)*(p+2)/2
  std::vector<std::vector<Vec4>> U_dg;          // [nelem][ndof][4]
  std::vector<std::vector<Vec4>> U0_dg;

  // p=0 / FV compatibility
  std::vector<Vec4> U;
  std::vector<Vec4> U0;

  std::vector<double> res_history;
  std::vector<double> cell_residuals;
  std::string fluxname = "roe";

  double gamma = 1.4;
  double CFL   = 1.0;
  double rtol  = 1e-6;
  double rho0  = 1.0;
  double a0    = 1.0;
  double alpha = 50.0;
  double Minf  = 0.1;
  double p0;
  double pout;
  double current_time = 0.0;

  FiniteVolumeSolver(const std::string &meshfile);

  std::string run_output_dir = "";
  double baseline_residual = -1.0;

  // raw integrated forces on the blade
  double Fx_raw = 0.0;
  double Fy_raw = 0.0;

  struct WallSample {
    int be_index;
    int elem;
    int qpt;
    double x;
    double y;
    double nx;
    double ny;
    double ds;
    double p;
    double cp;
    double dFx;
    double dFy;
  };

  std::vector<WallSample> wall_samples;

  // helper methods
  void syncCellAveragesFromDG();
  void collectWallSurfaceData();
  void writeAverageStateBinary(const std::string &filename) const;
  void writeDGStateBinary(const std::string &filename) const;
  void writeResidualBinary(const std::string &filename) const;
  void writeCellResidualBinary(const std::string &filename) const;
  void writeWallSurfaceCSV(const std::string &filename) const;
  void writeSummaryText(const std::string &filename) const;
  void writeSteadyOutputs(const std::string &outdir);

  void initializeDG();
  void setInitialCondition();
  void loadInitialCondition(const std::string &filename);
  void loadMappedInitialCondition(const std::string &coarse_meshfile,
                                  const std::string &coarse_statefile);
  void solveSteady(int itercap = 1000000);
  void solveUnsteady(int itercap = 1000000, double t_end = -1.0);
  void saveSnapshot(const std::string &filename);

private:
  struct ResidualResult {
    std::vector<Vec4>   R;
    std::vector<double> sdl;
  };

  // Mass matrix (reference triangle, inverted once at init)
  std::vector<std::vector<double>> MassMatrixInv;
  double mass_spectral_radius = 2.0;

  // Cell-average weights: w_j = (1/A_ref) * int phi_j dA_ref
  std::vector<double> cellAvgWeights;

  void computeMassMatrix();
  std::vector<double> evaluateBasis    (double xi, double eta, int p);
  std::vector<double> evaluateBasisGrad(double xi, double eta, int p);

  struct QuadRule {
    int n;
    std::vector<double> points;
    std::vector<double> weights;
  };
  QuadRule getQuadratureRule(int p_order);

  // Residuals
  ResidualResult calcResidual  (const std::vector<Vec4> &Un,
                                double time = 0.0, bool use_unsteady_wake = false);
  ResidualResult calcResidualSecondOrder(const std::vector<Vec4> &Un,
                                         bool limited, double time = 0.0,
                                         bool use_unsteady_wake = false);
  ResidualResult calcResidualDG(const std::vector<std::vector<Vec4>> &Un_dg,
                                double time = 0.0, bool use_unsteady_wake = false);

  // Cell average
  Vec4 cellAverage(const std::vector<Vec4> &dofs) const;

  // FV helpers (still defined in Solver.cpp)
  std::vector<Vec4> sspRK2(const std::vector<Vec4> &Un,
                            bool secondOrder, bool limited,
                            double time = 0.0, bool use_unsteady_wake = false);
  std::vector<Vec4> sspRK2(const std::vector<Vec4> &Un, double dt_global,
                            bool secondOrder, bool limited,
                            double time, bool use_unsteady_wake);
  std::vector<Vec4> applyLimiter(const std::vector<Vec4> &Un,
                                 std::vector<Vec4> &gradX,
                                 std::vector<Vec4> &gradY);
  bool isPhysical(const std::vector<Vec4> &Un);

  // DG time stepping (classical RK4)
  std::vector<std::vector<Vec4>> rk4_DG(
      const std::vector<std::vector<Vec4>> &Un_dg,
      double time = 0.0, bool use_unsteady_wake = false,
      const ResidualResult *res1_precomputed = nullptr);
  std::vector<std::vector<Vec4>> rk4_DG(
      const std::vector<std::vector<Vec4>> &Un_dg,
      double dt_global, double time, bool use_unsteady_wake,
      const ResidualResult *res1_precomputed = nullptr);

  std::vector<double> calcDt(const std::vector<double> &sdl);
  bool isPhysicalDG(const std::vector<std::vector<Vec4>> &Un_dg);
};

#endif
