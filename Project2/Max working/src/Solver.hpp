#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "Fluxes.hpp"
#include "Mesh.hpp"
#include <string>
#include <vector>

class FiniteVolumeSolver {
public:
  Mesh mesh;
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

  ResidualResult calcResidual(const std::vector<Vec4> &Un, double time = 0.0, 
                              bool use_unsteady_wake = false);
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

  std::vector<double> calcDt(const std::vector<double> &sdl);
  bool isPhysical(const std::vector<Vec4> &Un);
};

#endif
